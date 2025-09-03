"""
性能优化器 - SentiPort系统性能优化模块
包含缓存、并发、批处理、连接池等优化策略
"""

import asyncio
import aiohttp
import threading
import queue
import time
import hashlib
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps, lru_cache
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sqlite3
from contextlib import contextmanager
import redis
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheManager:
    """高性能缓存管理器 - 支持多级缓存"""
    
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        # L1缓存：内存缓存（快速）
        self.memory_cache = {}
        self.memory_cache_size = 0
        self.memory_cache_limit = 500 * 1024 * 1024  # 500MB
        
        # L2缓存：Redis缓存（持久化）
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=False,
                socket_connect_timeout=1
            )
            self.redis_client.ping()
            self.redis_available = True
            logger.info("Redis cache connected successfully")
        except:
            self.redis_available = False
            logger.warning("Redis not available, using memory cache only")
        
        # L3缓存：SQLite缓存（大数据持久化）
        self.sqlite_cache_path = "cache.db"
        self._init_sqlite_cache()
        
        # 缓存统计
        self.stats = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'redis_hits': 0,
            'sqlite_hits': 0
        }
    
    def _init_sqlite_cache(self):
        """初始化SQLite缓存表"""
        conn = sqlite3.connect(self.sqlite_cache_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB,
                expires_at REAL,
                created_at REAL,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)")
        conn.commit()
        conn.close()
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """多级缓存获取"""
        # L1: 内存缓存
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if entry['expires_at'] > time.time():
                self.stats['memory_hits'] += 1
                self.stats['hits'] += 1
                return entry['value']
            else:
                del self.memory_cache[key]
        
        # L2: Redis缓存
        if self.redis_available:
            try:
                data = self.redis_client.get(key)
                if data:
                    value = pickle.loads(data)
                    # 提升到L1缓存
                    self._set_memory(key, value, ttl=300)
                    self.stats['redis_hits'] += 1
                    self.stats['hits'] += 1
                    return value
            except Exception as e:
                logger.debug(f"Redis get error: {e}")
        
        # L3: SQLite缓存
        try:
            conn = sqlite3.connect(self.sqlite_cache_path)
            cursor = conn.execute(
                "SELECT value, expires_at FROM cache WHERE key = ? AND expires_at > ?",
                (key, time.time())
            )
            row = cursor.fetchone()
            conn.close()
            
            if row:
                value = pickle.loads(row[0])
                # 提升到L1和L2缓存
                self._set_memory(key, value, ttl=300)
                if self.redis_available:
                    self._set_redis(key, value, ttl=3600)
                self.stats['sqlite_hits'] += 1
                self.stats['hits'] += 1
                return value
        except Exception as e:
            logger.debug(f"SQLite get error: {e}")
        
        self.stats['misses'] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """多级缓存设置"""
        # L1: 内存缓存
        self._set_memory(key, value, ttl)
        
        # L2: Redis缓存
        if self.redis_available:
            self._set_redis(key, value, ttl)
        
        # L3: SQLite缓存（仅大数据或长期缓存）
        if ttl > 3600:  # 超过1小时的缓存写入SQLite
            self._set_sqlite(key, value, ttl)
    
    def _set_memory(self, key: str, value: Any, ttl: int):
        """设置内存缓存"""
        self.memory_cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl,
            'size': len(pickle.dumps(value))
        }
        # 简单的LRU逻辑
        if len(self.memory_cache) > 1000:
            # 删除最旧的10%
            sorted_keys = sorted(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k]['expires_at']
            )
            for k in sorted_keys[:100]:
                del self.memory_cache[k]
    
    def _set_redis(self, key: str, value: Any, ttl: int):
        """设置Redis缓存"""
        try:
            self.redis_client.setex(
                key,
                ttl,
                pickle.dumps(value)
            )
        except Exception as e:
            logger.debug(f"Redis set error: {e}")
    
    def _set_sqlite(self, key: str, value: Any, ttl: int):
        """设置SQLite缓存"""
        try:
            conn = sqlite3.connect(self.sqlite_cache_path)
            conn.execute("""
                INSERT OR REPLACE INTO cache 
                (key, value, expires_at, created_at, access_count, last_accessed)
                VALUES (?, ?, ?, ?, 1, ?)
            """, (
                key,
                pickle.dumps(value),
                time.time() + ttl,
                time.time(),
                time.time()
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"SQLite set error: {e}")
    
    def cache_decorator(self, ttl: int = 3600):
        """缓存装饰器"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = self._generate_key(func.__name__, args, kwargs)
                
                # 尝试从缓存获取
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_value
                
                # 执行函数
                result = func(*args, **kwargs)
                
                # 缓存结果
                self.set(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'hit_rate': f"{hit_rate:.2%}",
            'total_hits': self.stats['hits'],
            'total_misses': self.stats['misses'],
            'memory_hits': self.stats['memory_hits'],
            'redis_hits': self.stats['redis_hits'],
            'sqlite_hits': self.stats['sqlite_hits'],
            'memory_cache_size': len(self.memory_cache)
        }

class RateLimiter:
    """智能速率限制器 - 避免API 429错误"""
    
    def __init__(self, max_requests: int = 5, window_seconds: int = 1):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
        self.lock = threading.Lock()
        
        # 自适应参数
        self.adaptive_enabled = True
        self.error_count = 0
        self.success_count = 0
        self.current_delay = 0.1
    
    def acquire(self):
        """获取请求许可"""
        with self.lock:
            now = time.time()
            # 清理过期请求记录
            self.requests = [r for r in self.requests 
                           if r > now - self.window_seconds]
            
            # 检查是否超限
            if len(self.requests) >= self.max_requests:
                # 计算需要等待的时间
                oldest = min(self.requests)
                wait_time = oldest + self.window_seconds - now
                if wait_time > 0:
                    time.sleep(wait_time)
                    return self.acquire()  # 递归重试
            
            # 记录请求
            self.requests.append(now)
            
            # 添加自适应延迟
            if self.adaptive_enabled:
                time.sleep(self.current_delay)
    
    def report_success(self):
        """报告成功请求"""
        self.success_count += 1
        # 逐渐减少延迟
        if self.success_count > 10 and self.current_delay > 0.05:
            self.current_delay *= 0.95
            logger.debug(f"Reduced delay to {self.current_delay:.3f}s")
    
    def report_error(self, error_code: int = None):
        """报告错误请求"""
        self.error_count += 1
        
        if error_code == 429:  # Too Many Requests
            # 显著增加延迟
            self.current_delay = min(self.current_delay * 2, 5.0)
            self.max_requests = max(1, self.max_requests - 1)
            logger.warning(f"Rate limit hit! Increased delay to {self.current_delay}s")
        elif error_code in [500, 502, 503, 504]:  # Server errors
            # 适度增加延迟
            self.current_delay = min(self.current_delay * 1.5, 3.0)
            logger.warning(f"Server error! Increased delay to {self.current_delay}s")

class ConnectionPool:
    """HTTP连接池管理器"""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.session = None
        self._init_session()
    
    def _init_session(self):
        """初始化aiohttp会话"""
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=5,
            ttl_dns_cache=300,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=30,
            connect=5,
            sock_read=10
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
    
    async def get(self, url: str, **kwargs) -> Dict:
        """异步GET请求"""
        async with self.session.get(url, **kwargs) as response:
            return {
                'status': response.status,
                'data': await response.text(),
                'headers': dict(response.headers)
            }
    
    async def post(self, url: str, **kwargs) -> Dict:
        """异步POST请求"""
        async with self.session.post(url, **kwargs) as response:
            return {
                'status': response.status,
                'data': await response.json(),
                'headers': dict(response.headers)
            }
    
    async def close(self):
        """关闭连接池"""
        if self.session:
            await self.session.close()

class DatabaseOptimizer:
    """数据库性能优化器"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection_pool = queue.Queue(maxsize=10)
        self._init_pool()
    
    def _init_pool(self):
        """初始化连接池"""
        for _ in range(5):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            # 优化设置
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB
            self.connection_pool.put(conn)
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接"""
        conn = self.connection_pool.get()
        try:
            yield conn
        finally:
            self.connection_pool.put(conn)
    
    def create_indexes(self):
        """创建优化索引"""
        with self.get_connection() as conn:
            # 股票价格表索引
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_stock_prices_ticker_date 
                ON stock_prices(ticker, date DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_stock_prices_date 
                ON stock_prices(date DESC)
            """)
            
            # 新闻数据表索引
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_news_ticker_date 
                ON news_data(ticker, date DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_news_sentiment 
                ON news_data(sentiment_score)
            """)
            
            # Alpha因子表索引
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alpha_ticker_date 
                ON alpha_factors(ticker, date DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alpha_combined 
                ON alpha_factors(combined_alpha DESC)
            """)
            
            conn.commit()
            logger.info("Database indexes created successfully")
    
    def batch_insert(self, table: str, data: pd.DataFrame, chunk_size: int = 1000):
        """批量插入优化"""
        with self.get_connection() as conn:
            # 使用事务批量插入
            conn.execute("BEGIN TRANSACTION")
            try:
                for i in range(0, len(data), chunk_size):
                    chunk = data.iloc[i:i+chunk_size]
                    chunk.to_sql(table, conn, if_exists='append', index=False)
                conn.execute("COMMIT")
                logger.info(f"Batch inserted {len(data)} rows into {table}")
            except Exception as e:
                conn.execute("ROLLBACK")
                logger.error(f"Batch insert failed: {e}")
                raise

class AsyncDataFetcher:
    """异步数据获取器 - 并发获取股票数据"""
    
    def __init__(self, cache_manager: CacheManager, rate_limiter: RateLimiter):
        self.cache = cache_manager
        self.rate_limiter = rate_limiter
        self.connection_pool = ConnectionPool()
    
    async def fetch_stock_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """异步获取单个股票数据"""
        # 检查缓存
        cache_key = f"stock_{ticker}_{period}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
        
        # 速率限制
        self.rate_limiter.acquire()
        
        try:
            # 使用yfinance的异步接口（模拟）
            import yfinance as yf
            stock = yf.Ticker(ticker)
            hist = await asyncio.get_event_loop().run_in_executor(
                None, stock.history, period
            )
            
            if not hist.empty:
                # 缓存数据
                self.cache.set(cache_key, hist, ttl=3600)  # 1小时缓存
                self.rate_limiter.report_success()
                return hist
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            self.rate_limiter.report_error(429 if "429" in str(e) else 500)
            return pd.DataFrame()
    
    async def fetch_multiple_stocks(self, tickers: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """并发获取多个股票数据"""
        tasks = []
        for ticker in tickers:
            task = asyncio.create_task(self.fetch_stock_data(ticker, period))
            tasks.append((ticker, task))
        
        results = {}
        for ticker, task in tasks:
            try:
                data = await task
                if not data.empty:
                    results[ticker] = data
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
        
        return results

class ModelOptimizer:
    """NLP模型优化器 - 批处理和缓存"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.model = None
        self.tokenizer = None
        self.batch_size = 32
        self.max_length = 512
    
    def load_model_lazy(self):
        """延迟加载模型"""
        if self.model is None:
            from transformers import BertTokenizer, BertForSequenceClassification
            import torch
            
            # 使用量化模型减少内存
            self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
            self.model = BertForSequenceClassification.from_pretrained(
                'yiyanghkust/finbert-tone',
                torch_dtype=torch.float16  # 使用半精度
            )
            
            # 使用GPU如果可用
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Model loaded on GPU")
            else:
                logger.info("Model loaded on CPU")
            
            self.model.eval()  # 设置为推理模式
    
    def batch_predict(self, texts: List[str]) -> List[float]:
        """批量预测"""
        self.load_model_lazy()
        
        # 检查缓存
        cache_key = hashlib.md5(str(texts).encode()).hexdigest()
        cached_result = self.cache.get(f"nlp_{cache_key}")
        if cached_result is not None:
            return cached_result
        
        import torch
        
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # 批量编码
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length,
                padding=True
            )
            
            # 移动到GPU如果可用
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                
                # 计算情感分数
                for prob in probabilities:
                    # FinBERT: [negative, neutral, positive]
                    sentiment_score = prob[2].item() - prob[0].item()
                    results.append(sentiment_score)
        
        # 缓存结果
        self.cache.set(f"nlp_{cache_key}", results, ttl=7200)  # 2小时缓存
        
        return results

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {
            'api_latency': [],
            'db_latency': [],
            'nlp_latency': [],
            'cache_hit_rate': 0,
            'memory_usage': 0,
            'cpu_usage': 0
        }
        self.start_time = time.time()
    
    @contextmanager
    def measure_time(self, metric_name: str):
        """测量执行时间"""
        start = time.time()
        yield
        elapsed = time.time() - start
        
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(elapsed)
        
        # 记录慢查询
        if elapsed > 1.0:
            logger.warning(f"Slow operation {metric_name}: {elapsed:.2f}s")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取性能统计"""
        import psutil
        
        stats = {
            'uptime': time.time() - self.start_time,
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_percent': psutil.Process().cpu_percent(),
        }
        
        # 计算各项延迟统计
        for metric, values in self.metrics.items():
            if isinstance(values, list) and values:
                stats[f'{metric}_avg'] = np.mean(values)
                stats[f'{metric}_p50'] = np.percentile(values, 50)
                stats[f'{metric}_p95'] = np.percentile(values, 95)
                stats[f'{metric}_p99'] = np.percentile(values, 99)
        
        return stats
    
    def generate_report(self) -> str:
        """生成性能报告"""
        stats = self.get_statistics()
        
        report = ["=" * 50]
        report.append("PERFORMANCE REPORT")
        report.append("=" * 50)
        report.append(f"Uptime: {stats['uptime']:.0f} seconds")
        report.append(f"Memory Usage: {stats['memory_usage_mb']:.1f} MB")
        report.append(f"CPU Usage: {stats['cpu_percent']:.1f}%")
        report.append("")
        
        report.append("LATENCY METRICS (seconds):")
        for metric in ['api_latency', 'db_latency', 'nlp_latency']:
            if f'{metric}_avg' in stats:
                report.append(f"  {metric}:")
                report.append(f"    Average: {stats[f'{metric}_avg']:.3f}")
                report.append(f"    P50: {stats[f'{metric}_p50']:.3f}")
                report.append(f"    P95: {stats[f'{metric}_p95']:.3f}")
                report.append(f"    P99: {stats[f'{metric}_p99']:.3f}")
        
        report.append("=" * 50)
        
        return "\n".join(report)

# 全局实例
cache_manager = CacheManager()
rate_limiter = RateLimiter(max_requests=5, window_seconds=1)
performance_monitor = PerformanceMonitor()

def optimize_system():
    """系统优化入口函数"""
    logger.info("Starting system optimization...")
    
    # 1. 初始化缓存
    logger.info("Initializing cache system...")
    cache_stats = cache_manager.get_stats()
    logger.info(f"Cache stats: {cache_stats}")
    
    # 2. 优化数据库
    logger.info("Optimizing database...")
    db_optimizer = DatabaseOptimizer("financial_data.db")
    db_optimizer.create_indexes()
    
    # 3. 初始化模型优化器
    logger.info("Initializing model optimizer...")
    model_optimizer = ModelOptimizer(cache_manager)
    
    # 4. 启动性能监控
    logger.info("Starting performance monitor...")
    
    logger.info("System optimization completed!")
    
    return {
        'cache_manager': cache_manager,
        'rate_limiter': rate_limiter,
        'db_optimizer': db_optimizer,
        'model_optimizer': model_optimizer,
        'performance_monitor': performance_monitor
    }

if __name__ == "__main__":
    # 运行优化
    components = optimize_system()
    
    # 显示性能报告
    print("\n" + components['performance_monitor'].generate_report())
    print("\nCache Statistics:")
    print(components['cache_manager'].get_stats())