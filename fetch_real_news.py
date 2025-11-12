#!/usr/bin/env python3
"""
快速获取真实新闻数据
"""

import sqlite3
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from core_alpha_system import DatabaseManager, DataCollector
    import pandas as pd

    print("=" * 60)
    print("📰 获取真实新闻数据")
    print("=" * 60)
    print()

    # 初始化
    db = DatabaseManager()
    collector = DataCollector(db)

    # 获取股票列表
    conn = sqlite3.connect(db.db_path)
    try:
        tickers_df = pd.read_sql("SELECT DISTINCT ticker FROM stock_prices LIMIT 10", conn)
        if tickers_df.empty:
            print("⚠️ 数据库中没有股票数据，使用默认列表")
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        else:
            tickers = tickers_df['ticker'].tolist()
    finally:
        conn.close()

    print(f"📊 将获取以下股票的新闻: {', '.join(tickers)}")
    print()

    # 获取新闻（只获取前3个，避免API限制）
    print(f"🚀 开始获取新闻（前3只股票）...")
    print()

    success = collector.batch_collect_news(tickers[:3], force_refresh=True)

    if success:
        print()
        print("=" * 60)
        print("✅ 新闻数据获取成功！")

        # 显示统计
        conn = sqlite3.connect(db.db_path)
        stats = pd.read_sql("""
            SELECT ticker, COUNT(*) as count
            FROM news_data
            WHERE source != 'Simulated Dataset'
            GROUP BY ticker
        """, conn)
        conn.close()

        print()
        print("新闻统计:")
        for _, row in stats.iterrows():
            print(f"  {row['ticker']}: {row['count']} 条")

        print()
        print("现在可以运行: streamlit run demo_streamlit.py")
        print("=" * 60)
    else:
        print()
        print("❌ 新闻获取失败")
        print("请检查网络连接和API Key")

except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print()
    print("请运行: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
