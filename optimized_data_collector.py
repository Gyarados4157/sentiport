"""
优化数据收集器 - 高性能并发数据获取
解决Yahoo Finance API限制和性能问题
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import sqlite3
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
from functools import partial
import pickle
import hashlib

logger = logging.getLogger(__name__)

class OptimizedDataCollector:
    """优化的数据收集器 - 支持并发、缓存、智能重试"""
    
    def __init__(self, db_path: str = "financial_data.db"):
        self.db_path = db_path
        
        # 统计追踪
        self.success_count = 0
        self.error_count = 0
        self.failed_tickers = set()
        
        # 并发控制
        self.max_concurrent_requests = 5
        self.request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        # 重试策略
        self.max_retries = 3
        self.base_retry_delay = 1.0
        
        # 缓存配置
        self.cache_dir = "data_cache"
        self._init_cache_dir()
        
        # 批处理配置
        self.batch_size = 10
        
        # 速率限制追踪
        self.request_times = []
        self.rate_limit_window = 60  # 1分钟窗口
        self.max_requests_per_window = 30
    
    def _init_cache_dir(self):
        """初始化缓存目录"""
        import os
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _get_cache_key(self, ticker: str, period: str, data_type: str = "price") -> str:
        """生成缓存键"""
        today = datetime.now().strftime("%Y-%m-%d")
        key = f"{ticker}_{period}_{data_type}_{today}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """从缓存加载数据"""
        cache_file = f"{self.cache_dir}/{cache_key}.pkl"
        try:
            import os
            if os.path.exists(cache_file):
                # 检查缓存是否过期（24小时）
                file_age = time.time() - os.path.getmtime(cache_file)
                if file_age < 86400:  # 24小时
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                        logger.debug(f"Cache hit for {cache_key}")
                        return data
        except Exception as e:
            logger.debug(f"Cache read error: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """保存数据到缓存"""
        cache_file = f"{self.cache_dir}/{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
                logger.debug(f"Data cached with key {cache_key}")
        except Exception as e:
            logger.debug(f"Cache write error: {e}")
    
    async def _check_rate_limit(self):
        """检查并执行速率限制"""
        current_time = time.time()
        
        # 清理过期的请求记录
        self.request_times = [t for t in self.request_times 
                             if t > current_time - self.rate_limit_window]
        
        # 如果超过速率限制，等待
        if len(self.request_times) >= self.max_requests_per_window:
            oldest_request = min(self.request_times)
            wait_time = self.rate_limit_window - (current_time - oldest_request) + 1
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                return await self._check_rate_limit()  # 递归检查
        
        # 记录新请求
        self.request_times.append(current_time)
    
    async def fetch_stock_data_async(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """异步获取单个股票数据"""
        # 检查缓存
        cache_key = self._get_cache_key(ticker, period)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        async with self.request_semaphore:
            # 速率限制
            await self._check_rate_limit()
            
            retries = 0
            delay = self.base_retry_delay
            
            while retries < self.max_retries:
                try:
                    # 使用线程池执行同步yfinance调用
                    loop = asyncio.get_event_loop()
                    stock = yf.Ticker(ticker)
                    
                    # 异步执行
                    hist = await loop.run_in_executor(
                        None, 
                        partial(stock.history, period=period, timeout=10)
                    )
                    
                    if hist.empty:
                        logger.warning(f"No data for {ticker}")
                        return None
                    
                    # 数据预处理
                    hist = hist.reset_index()
                    hist['ticker'] = ticker
                    
                    # 缓存数据
                    self._save_to_cache(cache_key, hist)
                    
                    self.success_count += 1
                    logger.info(f"✅ Fetched {ticker}: {len(hist)} records")
                    return hist
                    
                except Exception as e:
                    retries += 1
                    self.error_count += 1
                    
                    error_str = str(e)
                    if "429" in error_str or "rate" in error_str.lower():
                        # 速率限制错误 - 增加延迟
                        delay = min(delay * 2, 60)
                        logger.warning(f"Rate limit hit for {ticker}, retry {retries}/{self.max_retries} after {delay}s")
                    elif "404" in error_str or "not found" in error_str.lower():
                        # 股票不存在
                        logger.error(f"Ticker {ticker} not found")
                        self.failed_tickers.add(ticker)
                        return None
                    else:
                        logger.error(f"Error fetching {ticker}: {e}, retry {retries}/{self.max_retries}")
                    
                    if retries < self.max_retries:
                        await asyncio.sleep(delay)
                        delay *= 1.5  # 指数退避
            
            # 所有重试失败
            logger.error(f"❌ Failed to fetch {ticker} after {self.max_retries} retries")
            self.failed_tickers.add(ticker)
            return None
    
    async def fetch_multiple_stocks_async(self, tickers: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """并发获取多个股票数据"""
        logger.info(f"Starting async fetch for {len(tickers)} stocks")
        
        # 创建任务
        tasks = []
        for ticker in tickers:
            task = asyncio.create_task(self.fetch_stock_data_async(ticker, period))
            tasks.append((ticker, task))
        
        # 等待所有任务完成
        results = {}
        for ticker, task in tasks:
            try:
                data = await task
                if data is not None and not data.empty:
                    results[ticker] = data
            except Exception as e:
                logger.error(f"Failed to process {ticker}: {e}")
                self.failed_tickers.add(ticker)
        
        logger.info(f"Fetched {len(results)}/{len(tickers)} stocks successfully")
        return results
    
    def fetch_stocks_batch(self, tickers: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """批量获取股票数据（同步接口）"""
        # 创建事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # 分批处理避免过载
            all_results = {}
            for i in range(0, len(tickers), self.batch_size):
                batch = tickers[i:i+self.batch_size]
                logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(tickers)-1)//self.batch_size + 1}")
                
                # 运行异步任务
                batch_results = loop.run_until_complete(
                    self.fetch_multiple_stocks_async(batch, period)
                )
                all_results.update(batch_results)
                
                # 批次间延迟
                if i + self.batch_size < len(tickers):
                    time.sleep(2)  # 批次间休息2秒
            
            return all_results
            
        finally:
            loop.close()
    
    def save_to_database(self, data: Dict[str, pd.DataFrame]):
        """简化的数据库保存"""
        try:
            # 准备批量数据
            all_data = []
            for ticker, df in data.items():
                df_clean = df.copy()
                df_clean['ticker'] = ticker
                
                # 重命名列
                df_clean = df_clean.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # 添加调整后收盘价
                if 'adj_close' not in df_clean.columns:
                    df_clean['adj_close'] = df_clean['close']
                
                # 选择需要的列
                columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
                df_clean = df_clean[columns]
                
                all_data.append(df_clean)
            
            # 合并所有数据
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                
                # 简单的数据库操作
                conn = sqlite3.connect(self.db_path, timeout=30)
                try:
                    # 删除旧数据并插入新数据
                    conn.execute("DELETE FROM stock_prices")
                    combined_df.to_sql('stock_prices', conn, if_exists='append', index=False)
                    conn.commit()
                    logger.info(f"Saved {len(combined_df)} records to database")
                finally:
                    conn.close()
            
        except Exception as e:
            logger.error(f"Database save failed: {e}")
            raise
    
    def get_statistics(self) -> Dict:
        """获取收集统计"""
        return {
            'success_count': self.success_count,
            'error_count': self.error_count,
            'failed_tickers': list(self.failed_tickers),
            'success_rate': self.success_count / (self.success_count + self.error_count) 
                          if (self.success_count + self.error_count) > 0 else 0,
            'current_rate_limit': len(self.request_times)
        }
    
    def collect_with_fallback(self, tickers: List[str], period: str = "1y") -> bool:
        """带降级策略的数据收集"""
        logger.info(f"Starting optimized data collection for {len(tickers)} stocks")
        
        # 第一轮：并发获取
        results = self.fetch_stocks_batch(tickers, period)
        
        # 第二轮：重试失败的股票（降低并发度）
        if self.failed_tickers:
            logger.info(f"Retrying {len(self.failed_tickers)} failed stocks with reduced concurrency")
            
            # 降低并发度重试
            self.max_concurrent_requests = 2
            retry_tickers = list(self.failed_tickers)
            self.failed_tickers.clear()
            
            retry_results = self.fetch_stocks_batch(retry_tickers, period)
            results.update(retry_results)
        
        # 保存到数据库
        if results:
            self.save_to_database(results)
            
            # 显示统计
            stats = self.get_statistics()
            logger.info(f"""
            =====================================
            Data Collection Complete
            =====================================
            Success: {stats['success_count']} stocks
            Failed: {stats['error_count']} attempts
            Success Rate: {stats['success_rate']:.1%}
            Failed Tickers: {', '.join(stats['failed_tickers']) if stats['failed_tickers'] else 'None'}
            =====================================
            """)
            
            return True
        
        return False

class IncrementalDataUpdater:
    """增量数据更新器 - 只更新最新数据"""
    
    def __init__(self, db_path: str = "financial_data.db"):
        self.db_path = db_path
        self.collector = OptimizedDataCollector(db_path)
    
    def get_last_update_dates(self) -> Dict[str, datetime]:
        """获取每只股票的最后更新日期"""
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT ticker, MAX(date) as last_date
        FROM stock_prices
        GROUP BY ticker
        """
        df = pd.read_sql(query, conn)
        conn.close()
        
        result = {}
        for _, row in df.iterrows():
            result[row['ticker']] = pd.to_datetime(row['last_date'])
        
        return result
    
    def update_incremental(self, tickers: List[str]) -> bool:
        """增量更新数据"""
        logger.info("Starting incremental data update")
        
        # 获取最后更新日期
        last_updates = self.get_last_update_dates()
        
        # 计算需要更新的天数
        today = datetime.now()
        update_tasks = []
        
        for ticker in tickers:
            if ticker in last_updates:
                days_since_update = (today - last_updates[ticker]).days
                if days_since_update > 0:
                    # 只获取缺失的数据
                    period = f"{min(days_since_update + 1, 30)}d"
                    update_tasks.append((ticker, period))
            else:
                # 新股票，获取1年数据
                update_tasks.append((ticker, "1y"))
        
        if not update_tasks:
            logger.info("All data is up to date")
            return True
        
        logger.info(f"Updating {len(update_tasks)} stocks incrementally")
        
        # 执行增量更新
        results = {}
        for ticker, period in update_tasks:
            data = self.collector.fetch_stocks_batch([ticker], period)
            if ticker in data:
                results[ticker] = data[ticker]
        
        # 合并到数据库（追加模式）
        if results:
            self._merge_incremental_data(results)
            return True
        
        return False
    
    def _merge_incremental_data(self, new_data: Dict[str, pd.DataFrame]):
        """合并增量数据到数据库"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            for ticker, df in new_data.items():
                # 准备数据
                df_clean = df.copy()
                df_clean['ticker'] = ticker
                df_clean = df_clean.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                if 'adj_close' not in df_clean.columns:
                    df_clean['adj_close'] = df_clean['close']
                
                # 删除该股票的重复日期数据
                dates = df_clean['date'].tolist()
                placeholders = ','.join(['?' for _ in dates])
                conn.execute(f"""
                    DELETE FROM stock_prices 
                    WHERE ticker = ? AND date IN ({placeholders})
                """, [ticker] + dates)
                
                # 插入新数据
                columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
                df_clean[columns].to_sql('stock_prices', conn, if_exists='append', index=False)
            
            conn.commit()
            logger.info(f"Merged incremental data for {len(new_data)} stocks")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to merge incremental data: {e}")
            raise
        finally:
            conn.close()

if __name__ == "__main__":
    # 测试优化的数据收集器
    collector = OptimizedDataCollector()
    
    # 测试股票列表
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # 执行收集
    success = collector.collect_with_fallback(test_tickers, period="1m")
    
    if success:
        print("Data collection completed successfully!")
        stats = collector.get_statistics()
        print(f"Statistics: {stats}")
    else:
        print("Data collection failed!")