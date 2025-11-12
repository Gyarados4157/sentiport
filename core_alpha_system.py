"""
Core Alpha Factor System - NLP-based Quantitative Trading Signal Generation
Focuses on 5 core text factors: sentiment momentum, sentiment reversal, news anomaly, text momentum, sentiment divergence
"""

import pandas as pd
import numpy as np
import sqlite3
try:
    import yfinance as yf
except ImportError:  # pragma: no cover - fallback when yfinance缺失
    yf = None
from datetime import datetime, timedelta
import requests
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# NLP dependencies
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover
    nltk = None
    SentimentIntensityAnalyzer = None
try:
    from transformers import BertTokenizer, BertForSequenceClassification
except ImportError:  # pragma: no cover
    BertTokenizer = None
    BertForSequenceClassification = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging


if nltk is not None and SentimentIntensityAnalyzer is not None:
    nltk.download('vader_lexicon', quiet=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """SQLite数据库管理器"""
    
    def __init__(self, db_path: str = "financial_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        
        # 价格数据表
        conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            date TEXT,
            ticker TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            adj_close REAL,
            PRIMARY KEY (date, ticker)
        )
        """)
        
        # 新闻数据表
        conn.execute("""
        CREATE TABLE IF NOT EXISTS news_data (
            id INTEGER PRIMARY KEY,
            date TEXT,
            ticker TEXT,
            title TEXT,
            summary TEXT,
            url TEXT,
            source TEXT,
            sentiment_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Alpha因子表
        conn.execute("""
        CREATE TABLE IF NOT EXISTS alpha_factors (
            date TEXT,
            ticker TEXT,
            sentiment_momentum REAL,
            sentiment_reversal REAL,
            news_volume_anomaly REAL,
            text_momentum REAL,
            sentiment_divergence REAL,
            combined_alpha REAL,
            PRIMARY KEY (date, ticker)
        )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")

class DataCollector:
    """数据收集器 - 整合多源数据"""

    def __init__(self, db_manager: DatabaseManager, alpha_vantage_key: str = "CR9P7L6SGO9W1L8V"):
        self.db = db_manager
        self.alpha_vantage_key = alpha_vantage_key

    # ---- 辅助函数 -------------------------------------------------

    @staticmethod
    def _normalize_price_dates(df: pd.DataFrame) -> pd.DataFrame:
        """将日期列统一为无时区的YYYY-MM-DD文本"""
        if 'date' in df.columns:
            date_series = pd.to_datetime(df['date'], errors='coerce')
        elif 'Date' in df.columns:
            date_series = pd.to_datetime(df['Date'], errors='coerce')
        else:
            return df

        date_series = date_series.dt.tz_localize(None)
        df['date'] = date_series.dt.strftime('%Y-%m-%d')
        return df

    @staticmethod
    def _period_to_sessions(period: str) -> int:
        """将yfinance周期字符串转换为大致的交易日数量"""
        mapping = {
            '1y': 252,
            '2y': 252 * 2,
            '3y': 252 * 3,
            '6mo': 126,
            '1mo': 21,
        }
        return mapping.get(period.lower(), 252)

    def _generate_synthetic_price_data(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """在网络不可用时生成离线演示数据"""
        try:
            sessions = self._period_to_sessions(period)
            end_date = datetime.utcnow().date()
            start_sessions = max(sessions, 60)
            # 使用交易日频率
            date_index = pd.bdate_range(end=end_date, periods=start_sessions)

            rng = np.random.default_rng(abs(hash(ticker)) % (2 ** 32))
            # 产生随机对数收益，控制波动在2%
            log_returns = rng.normal(loc=0.0003, scale=0.02, size=len(date_index))
            prices = 100 * np.exp(np.cumsum(log_returns))

            df = pd.DataFrame({
                'Date': date_index,
                'Open': prices * (1 + rng.normal(0, 0.002, len(prices))),
                'High': prices * (1 + rng.normal(0.003, 0.002, len(prices))).clip(min=0),
                'Low': prices * (1 + rng.normal(-0.003, 0.002, len(prices))).clip(min=0),
                'Close': prices,
                'Volume': rng.integers(low=1_000_000, high=10_000_000, size=len(prices))
            })

            df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
            df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)

            df = self._normalize_price_dates(df)
            df['adj_close'] = df['Close']
            df['ticker'] = ticker

            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            logger.warning(f"⚠️ 使用模拟价格数据用于 {ticker} ({period})")
            return df[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'adj_close']]
        except Exception as e:
            logger.error(f"无法生成模拟价格数据: {e}")
            return None
    
    def get_sp500_tickers(self, limit: int = 30) -> List[str]:
        """获取S&P 500成分股代码（demo版本限制数量）"""
        # 预定义的大盘股列表（可扩展）
        major_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
            'JNJ', 'V', 'PG', 'JPM', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC',
            'NFLX', 'ADBE', 'CRM', 'CMCSA', 'XOM', 'VZ', 'ABT', 'PFE', 'T',
            'WMT', 'CSCO', 'PEP'
        ]
        return major_stocks[:limit]
    
    def collect_stock_data(self, tickers: List[str], period: str = "2y") -> bool:
        """批量收集股票价格数据 - 增强版带重试和错误处理"""
        logger.info(f"Collecting price data for {len(tickers)} stocks")
        
        # 使用WAL模式提高并发性能
        conn = sqlite3.connect(self.db.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        collected_frames = []
        successful_tickers = []
        failed_tickers = []

        if yf is None:
            logger.warning("yfinance 未安装，所有价格数据将使用模拟数据")

        for ticker in tickers:
            retries = 3 if yf is not None else 0
            success = False

            if yf is not None:
                for attempt in range(retries):
                    try:
                        logger.info(f"Fetching {ticker} (attempt {attempt + 1}/{retries})")
                        stock = yf.Ticker(ticker)
                        hist = stock.history(period=period, timeout=30)

                        if hist.empty or len(hist) < 50:
                            logger.warning(f"No sufficient data returned for {ticker} (records={len(hist)})")
                            break

                        data = hist.reset_index()
                        data['ticker'] = ticker
                        data = data.rename(columns={
                            'Date': 'date',
                            'Open': 'open',
                            'High': 'high',
                            'Low': 'low',
                            'Close': 'close',
                            'Volume': 'volume'
                        })
                        data = self._normalize_price_dates(data)
                        data['adj_close'] = data['close']
                        data = data.dropna()

                        if data.empty:
                            logger.warning(f"No valid data after cleaning for {ticker}")
                            break

                        collected_frames.append(data[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'adj_close']])
                        successful_tickers.append(ticker)
                        success = True
                        break

                    except Exception as e:
                        logger.error(f"❌ Attempt {attempt + 1} failed for {ticker}: {e}")
                        if attempt < retries - 1:
                            wait_time = (attempt + 1) * 2
                            logger.info(f"Waiting {wait_time}s before retry...")
                            time.sleep(wait_time)

            if not success:
                synthetic = self._generate_synthetic_price_data(ticker, period)
                if synthetic is not None and not synthetic.empty:
                    collected_frames.append(synthetic)
                    successful_tickers.append(ticker)
                else:
                    failed_tickers.append(ticker)

            time.sleep(0.5)

        try:
            if collected_frames:
                combined_df = pd.concat(collected_frames, ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['date', 'ticker'], keep='last')
                combined_df.to_sql('stock_prices', conn, if_exists='replace', index=False)
            conn.commit()
            logger.info(f"✅ Data collection completed: {len(successful_tickers)} successful, {len(failed_tickers)} failed")
        except Exception as e:
            logger.error(f"❌ Failed to commit data: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

        if successful_tickers:
            logger.info(f"✅ Successfully collected: {', '.join(successful_tickers)}")
        if failed_tickers:
            logger.warning(f"⚠️ Failed to collect: {', '.join(failed_tickers)}")

        return len(successful_tickers) > 0
    
    def check_news_cache(self, ticker: str, days: int = 7) -> bool:
        """检查是否已有最近的新闻缓存"""
        conn = sqlite3.connect(self.db.db_path)
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            query = """
            SELECT COUNT(*) as count FROM news_data
            WHERE ticker = ? AND date >= ?
            """
            result = pd.read_sql(query, conn, params=[ticker, cutoff_date])
            return result['count'].iloc[0] > 0
        finally:
            conn.close()

    def collect_news_data(self, ticker: str, days: int = 30, force_refresh: bool = False) -> List[Dict]:
        """收集单个股票的新闻数据（带缓存）"""
        # 检查缓存
        if not force_refresh and self.check_news_cache(ticker, days=7):
            logger.info(f"✅ 使用缓存的新闻数据: {ticker}")
            return []

        try:
            logger.info(f"📰 正在从 Alpha Vantage 获取 {ticker} 的新闻...")
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'apikey': self.alpha_vantage_key,
                'limit': 200
            }

            response = requests.get(url, params=params, timeout=15)

            if response.status_code != 200:
                logger.error(f"API 请求失败: {response.status_code}")
                return []

            data = response.json()

            # 检查 API 限制
            if 'Note' in data:
                logger.warning(f"⚠️ API 限制: {data['Note']}")
                return []

            if 'Information' in data:
                logger.warning(f"⚠️ API 信息: {data['Information']}")
                return []

            if 'feed' not in data:
                logger.error(f"未找到新闻数据: {data}")
                return []

            news_items = []
            for item in data['feed'][:100]:  # 获取更多新闻
                # 解析时间戳: 20231215T153000 -> 2023-12-15
                time_published = item.get('time_published', '')
                if len(time_published) >= 8:
                    date_str = f"{time_published[:4]}-{time_published[4:6]}-{time_published[6:8]}"
                else:
                    date_str = datetime.now().strftime('%Y-%m-%d')

                # 获取情感分数
                ticker_sentiments = item.get('ticker_sentiment', [])
                sentiment_score = 0.0
                for ts in ticker_sentiments:
                    if ts.get('ticker') == ticker:
                        try:
                            sentiment_score = float(ts.get('ticker_sentiment_score', 0.0))
                        except (ValueError, TypeError):
                            sentiment_score = 0.0
                        break

                news_items.append({
                    'date': date_str,
                    'ticker': ticker,
                    'title': item.get('title', '')[:500],  # 限制长度
                    'summary': item.get('summary', '')[:1000],
                    'url': item.get('url', ''),
                    'source': item.get('source', 'Unknown'),
                    'sentiment_score': sentiment_score
                })

            logger.info(f"✅ 成功获取 {len(news_items)} 条新闻: {ticker}")
            return news_items

        except requests.exceptions.Timeout:
            logger.error(f"⏰ API 请求超时: {ticker}")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ 网络请求错误 {ticker}: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ 收集新闻时发生错误 {ticker}: {e}")
            return []

    def batch_collect_news(self, tickers: List[str], force_refresh: bool = False) -> bool:
        """批量收集多个股票的新闻数据"""
        logger.info(f"📰 开始收集 {len(tickers)} 只股票的新闻数据...")

        conn = sqlite3.connect(self.db.db_path, timeout=30.0)
        all_news = []
        successful_count = 0

        for i, ticker in enumerate(tickers):
            logger.info(f"[{i+1}/{len(tickers)}] 正在处理: {ticker}")

            news_items = self.collect_news_data(ticker, force_refresh=force_refresh)

            if news_items:
                all_news.extend(news_items)
                successful_count += 1

            # API 限制：每分钟最多 5 次请求（免费版）
            if i < len(tickers) - 1:
                time.sleep(12)  # 等待 12 秒，确保不超过限制

        # 保存到数据库
        if all_news:
            try:
                news_df = pd.DataFrame(all_news)

                # 去重：基于 ticker + title
                news_df = news_df.drop_duplicates(subset=['ticker', 'title'], keep='first')

                # 检查已存在的新闻
                existing_query = """
                SELECT ticker, title FROM news_data
                """
                existing_df = pd.read_sql(existing_query, conn)

                if not existing_df.empty:
                    # 过滤掉已存在的新闻
                    merged = news_df.merge(
                        existing_df,
                        on=['ticker', 'title'],
                        how='left',
                        indicator=True
                    )
                    news_df = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

                if not news_df.empty:
                    news_df.to_sql('news_data', conn, if_exists='append', index=False)
                    logger.info(f"✅ 成功保存 {len(news_df)} 条新闻到数据库")
                else:
                    logger.info("ℹ️ 没有新的新闻需要保存（全部已存在）")

                conn.commit()
            except Exception as e:
                logger.error(f"❌ 保存新闻数据失败: {e}")
                conn.rollback()
                return False
            finally:
                conn.close()

        logger.info(f"✅ 新闻收集完成: {successful_count}/{len(tickers)} 成功")
        return successful_count > 0

class NLPProcessor:
    """NLP处理器 - 文本分析和情感计算"""
    
    def __init__(self):
        # 初始化VADER
        if SentimentIntensityAnalyzer is not None:
            self.vader = SentimentIntensityAnalyzer()
        else:
            self.vader = None
            logger.warning("NLTK VADER 未安装，情感分析将退化为0")
        
        # 初始化FinBERT
        if torch is None or BertTokenizer is None or BertForSequenceClassification is None:
            self.finbert_available = False
            self.finbert_tokenizer = None
            self.finbert_model = None
            logger.warning("FinBERT依赖未满足（需要torch和transformers），将使用简化情感分析")
        else:
            try:
                self.finbert_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
                self.finbert_model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
                self.finbert_available = True
                logger.info("FinBERT model loaded successfully")
            except Exception as exc:
                self.finbert_available = False
                self.finbert_tokenizer = None
                self.finbert_model = None
                logger.warning(f"FinBERT model not available ({exc}); using fallback sentiment")
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """综合情感分析"""
        if not text:
            return {'compound': 0.0, 'confidence': 0.0}
        
        # VADER分析
        if self.vader is not None:
            vader_scores = self.vader.polarity_scores(text)
        else:
            vader_scores = {'compound': 0.0}
        
        # FinBERT分析（如果可用）
        if self.finbert_available and self.finbert_model is not None and torch is not None:
            try:
                inputs = self.finbert_tokenizer(text, return_tensors='pt', 
                                              truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = self.finbert_model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    
                # FinBERT输出：[negative, neutral, positive]
                finbert_score = probabilities[0][2].item() - probabilities[0][0].item()
                
                # 结合两个分数
                combined_score = (vader_scores['compound'] + finbert_score) / 2
                confidence = max(abs(vader_scores['compound']), abs(finbert_score))
                
            except:
                combined_score = vader_scores['compound']
                confidence = abs(vader_scores['compound'])
        else:
            combined_score = vader_scores['compound']
            confidence = abs(vader_scores['compound'])
        
        return {
            'compound': combined_score,
            'confidence': confidence
        }
    
    def extract_text_features(self, texts: List[str]) -> Dict[str, List[float]]:
        """提取文本特征用于因子计算"""
        features = {
            'sentiment_scores': [],
            'confidence_scores': [],
            'text_lengths': [],
            'keyword_counts': [],
            'urgency_scores': []
        }
        
        # 金融关键词
        financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'growth', 'decline',
            'bullish', 'bearish', 'buy', 'sell', 'upgrade', 'downgrade',
            'merger', 'acquisition', 'dividend', 'split', 'breakthrough'
        ]
        
        urgency_words = [
            'breaking', 'urgent', 'alert', 'emergency', 'crisis', 'surge',
            'plunge', 'spike', 'crash', 'soar', 'tumble'
        ]
        
        for text in texts:
            if not text:
                # 填充默认值
                features['sentiment_scores'].append(0.0)
                features['confidence_scores'].append(0.0)
                features['text_lengths'].append(0)
                features['keyword_counts'].append(0)
                features['urgency_scores'].append(0.0)
                continue
            
            # 情感分析
            sentiment = self.analyze_sentiment(text)
            features['sentiment_scores'].append(sentiment['compound'])
            features['confidence_scores'].append(sentiment['confidence'])
            
            # 文本长度
            features['text_lengths'].append(len(text))
            
            # 关键词计数
            text_lower = text.lower()
            keyword_count = sum(1 for keyword in financial_keywords if keyword in text_lower)
            features['keyword_counts'].append(keyword_count)
            
            # 紧急性评分
            urgency_count = sum(1 for word in urgency_words if word in text_lower)
            urgency_score = min(urgency_count / 5.0, 1.0)  # 标准化到0-1
            features['urgency_scores'].append(urgency_score)
        
        return features

class AlphaFactorEngine:
    """Alpha因子引擎 - 生成交易信号"""
    
    def __init__(self, db_manager: DatabaseManager, nlp_processor: NLPProcessor):
        self.db = db_manager
        self.nlp = nlp_processor
        self.scaler = StandardScaler()
    
    def calculate_sentiment_momentum(self, ticker: str, lookback: int = 20) -> float:
        """情感动量因子 - 情感变化率"""
        conn = sqlite3.connect(self.db.db_path)
        
        query = """
        SELECT date, sentiment_score 
        FROM news_data 
        WHERE ticker = ? AND sentiment_score IS NOT NULL
        ORDER BY date DESC 
        LIMIT ?
        """
        
        df = pd.read_sql(query, conn, params=[ticker, lookback])
        conn.close()
        
        if len(df) < 5:
            return 0.0
        
        # 计算情感动量（近期vs历史平均）
        recent_sentiment = df['sentiment_score'].head(5).mean()
        historical_sentiment = df['sentiment_score'].tail(15).mean()
        
        momentum = (recent_sentiment - historical_sentiment) / (abs(historical_sentiment) + 0.01)
        
        return np.clip(momentum, -3, 3)  # 限制极端值
    
    def calculate_sentiment_reversal(self, ticker: str, threshold: float = 2.0) -> float:
        """情感反转因子 - 极端情感后的反转信号"""
        conn = sqlite3.connect(self.db.db_path)
        
        query = """
        SELECT sentiment_score 
        FROM news_data 
        WHERE ticker = ? AND sentiment_score IS NOT NULL
        ORDER BY date DESC 
        LIMIT 10
        """
        
        df = pd.read_sql(query, conn, params=[ticker])
        conn.close()
        
        if len(df) < 3:
            return 0.0
        
        recent_sentiment = df['sentiment_score'].iloc[0]
        avg_sentiment = df['sentiment_score'].mean()
        std_sentiment = df['sentiment_score'].std() + 0.01
        
        # Z-score标准化
        z_score = (recent_sentiment - avg_sentiment) / std_sentiment
        
        # 反转信号：极端正面->负信号，极端负面->正信号
        if abs(z_score) > threshold:
            reversal_signal = -np.sign(z_score) * (abs(z_score) - threshold)
            return np.clip(reversal_signal, -2, 2)
        
        return 0.0
    
    def calculate_news_volume_anomaly(self, ticker: str, lookback: int = 30) -> float:
        """新闻量异常因子 - 新闻流量激增检测"""
        conn = sqlite3.connect(self.db.db_path)
        
        # 获取每日新闻数量
        query = """
        SELECT DATE(date) as day, COUNT(*) as news_count
        FROM news_data 
        WHERE ticker = ?
        GROUP BY DATE(date)
        ORDER BY day DESC 
        LIMIT ?
        """
        
        df = pd.read_sql(query, conn, params=[ticker, lookback])
        conn.close()
        
        if len(df) < 7:
            return 0.0
        
        recent_volume = df['news_count'].head(3).mean()
        historical_volume = df['news_count'].tail(20).mean()
        
        if historical_volume == 0:
            return 0.0
        
        volume_ratio = recent_volume / historical_volume
        
        # 异常检测：超过历史平均2倍认为是异常
        if volume_ratio > 2.0:
            anomaly_score = min((volume_ratio - 2.0) / 2.0, 1.0)
            return anomaly_score
        
        return 0.0
    
    def calculate_text_momentum(self, ticker: str, lookback: int = 15) -> float:
        """文本动量因子 - 关键词频率变化"""
        conn = sqlite3.connect(self.db.db_path)
        
        query = """
        SELECT date, title, summary 
        FROM news_data 
        WHERE ticker = ?
        ORDER BY date DESC 
        LIMIT ?
        """
        
        df = pd.read_sql(query, conn, params=[ticker, lookback * 2])
        conn.close()
        
        if len(df) < 10:
            return 0.0
        
        # 分割为近期和历史
        recent_texts = df.head(lookback)['summary'].fillna('').tolist()
        historical_texts = df.tail(lookback)['summary'].fillna('').tolist()
        
        # 提取特征
        recent_features = self.nlp.extract_text_features(recent_texts)
        historical_features = self.nlp.extract_text_features(historical_texts)
        
        # 计算关键词频率变化
        recent_keyword_freq = np.mean(recent_features['keyword_counts'])
        historical_keyword_freq = np.mean(historical_features['keyword_counts']) + 0.01
        
        momentum = (recent_keyword_freq - historical_keyword_freq) / historical_keyword_freq
        
        return np.clip(momentum, -2, 2)
    
    def calculate_sentiment_divergence(self, ticker: str, lookback: int = 15) -> float:
        """情感分歧度因子 - 市场意见分歧程度"""
        conn = sqlite3.connect(self.db.db_path)
        
        query = """
        SELECT sentiment_score 
        FROM news_data 
        WHERE ticker = ? AND sentiment_score IS NOT NULL
        ORDER BY date DESC 
        LIMIT ?
        """
        
        df = pd.read_sql(query, conn, params=[ticker, lookback])
        conn.close()
        
        if len(df) < 5:
            return 0.0
        
        sentiments = df['sentiment_score'].values
        
        # 计算情感分散度
        sentiment_std = np.std(sentiments)
        sentiment_range = np.max(sentiments) - np.min(sentiments)
        
        # 分歧度：标准差和范围的综合
        divergence = (sentiment_std + sentiment_range / 4.0) / 2.0
        
        return min(divergence, 2.0)  # 限制最大值
    
    def generate_combined_alpha(self, ticker: str) -> Dict[str, float]:
        """生成综合Alpha信号"""
        factors = {
            'sentiment_momentum': self.calculate_sentiment_momentum(ticker),
            'sentiment_reversal': self.calculate_sentiment_reversal(ticker),
            'news_volume_anomaly': self.calculate_news_volume_anomaly(ticker),
            'text_momentum': self.calculate_text_momentum(ticker),
            'sentiment_divergence': self.calculate_sentiment_divergence(ticker)
        }
        
        # 权重配置（可调优）
        weights = {
            'sentiment_momentum': 0.3,
            'sentiment_reversal': 0.25,
            'news_volume_anomaly': 0.2,
            'text_momentum': 0.15,
            'sentiment_divergence': 0.1
        }
        
        # 计算加权组合
        combined_alpha = sum(factors[k] * weights[k] for k in factors.keys())
        factors['combined_alpha'] = combined_alpha
        
        return factors

class BacktestEngine:
    """简化回测引擎"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def calculate_factor_ic(self, lookback_days: int = 250) -> pd.DataFrame:
        """计算因子信息系数（IC）"""
        conn = sqlite3.connect(self.db.db_path)
        
        # 获取因子和收益数据
        query = """
        SELECT a.date, a.ticker, a.combined_alpha,
               p1.close as current_price,
               p2.close as next_price
        FROM alpha_factors a
        JOIN stock_prices p1 ON a.date = p1.date AND a.ticker = p1.ticker
        JOIN stock_prices p2 ON DATE(a.date, '+1 day') = p2.date AND a.ticker = p2.ticker
        ORDER BY a.date DESC
        LIMIT ?
        """
        
        df = pd.read_sql(query, conn, params=[lookback_days * 50])
        conn.close()
        
        if df.empty:
            return pd.DataFrame()
        
        # 计算未来收益
        df['future_return'] = (df['next_price'] - df['current_price']) / df['current_price']
        
        # 按日期分组计算IC
        ic_results = []
        for date, group in df.groupby('date'):
            if len(group) > 10:  # 至少需要10只股票
                ic = np.corrcoef(group['combined_alpha'], group['future_return'])[0, 1]
                if not np.isnan(ic):
                    ic_results.append({'date': date, 'ic': ic})
        
        ic_df = pd.DataFrame(ic_results)
        if not ic_df.empty:
            ic_df['date'] = pd.to_datetime(ic_df['date'])
        
        return ic_df
    
    def get_performance_summary(self) -> Dict[str, float]:
        """获取性能摘要"""
        ic_df = self.calculate_factor_ic()
        
        if ic_df.empty:
            return {'ic_mean': 0, 'ic_std': 0, 'ir': 0, 'hit_rate': 0}
        
        ic_mean = ic_df['ic'].mean()
        ic_std = ic_df['ic'].std()
        ir = ic_mean / (ic_std + 0.001)  # Information Ratio
        hit_rate = (ic_df['ic'] > 0).mean()
        
        return {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ir': ir,
            'hit_rate': hit_rate
        }

def main_demo():
    """主演示函数"""
    logger.info("=== Alpha因子系统Demo启动 ===")
    
    # 初始化组件
    db = DatabaseManager()
    collector = DataCollector(db)
    nlp = NLPProcessor()
    alpha_engine = AlphaFactorEngine(db, nlp)
    backtest = BacktestEngine(db)
    
    # 1. 数据收集
    logger.info("Step 1: 收集数据")
    tickers = collector.get_sp500_tickers(limit=10)  # Demo用10只股票
    logger.info(f"目标股票: {tickers}")
    
    # 收集价格数据
    collector.collect_stock_data(tickers)

    # 收集真实新闻数据
    logger.info("收集真实新闻数据...")
    collector.batch_collect_news(tickers[:5], force_refresh=False)  # 前5只股票
    
    # 2. 生成Alpha因子
    logger.info("Step 2: 生成Alpha因子")
    conn = sqlite3.connect(db.db_path)
    
    alpha_results = []
    for ticker in tickers[:3]:  # Demo只计算前3只
        logger.info(f"计算 {ticker} 的Alpha因子")
        factors = alpha_engine.generate_combined_alpha(ticker)
        
        result = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'ticker': ticker,
            **factors
        }
        alpha_results.append(result)
        
        logger.info(f"{ticker} Alpha信号: {factors['combined_alpha']:.4f}")
    
    # 保存Alpha因子
    pd.DataFrame(alpha_results).to_sql('alpha_factors', conn, if_exists='replace', index=False)
    conn.close()
    
    # 3. 简单回测分析
    logger.info("Step 3: 性能分析")
    performance = backtest.get_performance_summary()
    
    logger.info("=== 系统性能摘要 ===")
    for key, value in performance.items():
        logger.info(f"{key}: {value:.4f}")
    
    # 4. 生成信号报告
    logger.info("Step 4: 生成交易信号")
    alpha_df = pd.DataFrame(alpha_results)
    
    # 标准化信号
    alpha_df['signal'] = np.where(alpha_df['combined_alpha'] > 0.1, 'BUY',
                        np.where(alpha_df['combined_alpha'] < -0.1, 'SELL', 'HOLD'))
    
    logger.info("=== 交易信号 ===")
    for _, row in alpha_df.iterrows():
        logger.info(f"{row['ticker']}: {row['signal']} (Alpha: {row['combined_alpha']:.4f})")
    
    logger.info("=== Demo完成 ===")
    logger.info(f"数据库文件: {db.db_path}")
    logger.info("可以通过Streamlit界面查看详细结果")
    
    return alpha_df, performance

if __name__ == "__main__":
    # 运行Demo
    results, perf = main_demo()
    print("\nDemo运行完成！")
    print(f"生成了 {len(results)} 个交易信号")
    print(f"平均IC: {perf['ic_mean']:.4f}")
