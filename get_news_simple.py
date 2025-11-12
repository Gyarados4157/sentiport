#!/usr/bin/env python3
"""
简单的新闻获取脚本 - 最小依赖版本
"""

import sqlite3
import json
import time
from datetime import datetime

# 检查 requests 库
try:
    import requests
except ImportError:
    print("❌ 缺少 requests 库")
    print("请运行: pip install requests")
    exit(1)

# 配置
API_KEY = "CR9P7L6SGO9W1L8V"
DB_PATH = "financial_data.db"
TICKERS = ['AAPL', 'MSFT', 'GOOGL']  # 只获取3只股票

print("=" * 60)
print("📰 获取真实新闻数据（简化版）")
print("=" * 60)
print()

# 连接数据库
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 确保表存在
cursor.execute("""
CREATE TABLE IF NOT EXISTS news_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
conn.commit()

print(f"📊 将获取以下股票的新闻: {', '.join(TICKERS)}")
print()

total_added = 0

for i, ticker in enumerate(TICKERS, 1):
    print(f"[{i}/{len(TICKERS)}] 正在获取 {ticker} 的新闻...")

    try:
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'apikey': API_KEY,
            'limit': 50
        }

        response = requests.get(url, params=params, timeout=15)

        if response.status_code != 200:
            print(f"  ❌ HTTP 错误: {response.status_code}")
            continue

        data = response.json()

        # 检查错误
        if 'Note' in data:
            print(f"  ⚠️ API 限制: {data['Note']}")
            continue

        if 'feed' not in data:
            print(f"  ⚠️ 没有新闻数据")
            continue

        # 处理新闻
        added = 0
        for item in data['feed'][:30]:  # 限制每只股票30条
            # 解析日期
            time_published = item.get('time_published', '')
            if len(time_published) >= 8:
                date_str = f"{time_published[:4]}-{time_published[4:6]}-{time_published[6:8]}"
            else:
                date_str = datetime.now().strftime('%Y-%m-%d')

            # 获取情感分数
            sentiment_score = 0.0
            for ts in item.get('ticker_sentiment', []):
                if ts.get('ticker') == ticker:
                    try:
                        sentiment_score = float(ts.get('ticker_sentiment_score', 0.0))
                    except:
                        pass
                    break

            title = item.get('title', '')[:500]
            summary = item.get('summary', '')[:1000]
            url = item.get('url', '')
            source = item.get('source', 'Unknown')

            # 检查是否已存在
            cursor.execute("""
                SELECT COUNT(*) FROM news_data
                WHERE ticker = ? AND title = ?
            """, (ticker, title))

            if cursor.fetchone()[0] == 0:
                # 插入新闻
                cursor.execute("""
                    INSERT INTO news_data (date, ticker, title, summary, url, source, sentiment_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (date_str, ticker, title, summary, url, source, sentiment_score))
                added += 1

        conn.commit()
        print(f"  ✅ 成功添加 {added} 条新闻")
        total_added += added

        # API 限流：等待12秒
        if i < len(TICKERS):
            print(f"  ⏳ 等待 12 秒（API 限流）...")
            time.sleep(12)

    except Exception as e:
        print(f"  ❌ 错误: {e}")
        continue

# 显示统计
print()
print("=" * 60)
cursor.execute("""
    SELECT ticker, COUNT(*) as count
    FROM news_data
    WHERE source != 'Simulated Dataset'
    GROUP BY ticker
""")

stats = cursor.fetchall()
if stats:
    print("📊 数据库新闻统计:")
    for ticker, count in stats:
        print(f"  {ticker}: {count} 条")
else:
    print("⚠️ 数据库中没有新闻数据")

conn.close()

print()
print(f"✅ 总共添加了 {total_added} 条新闻")
print()
print("现在可以运行: streamlit run demo_streamlit.py")
print("=" * 60)
