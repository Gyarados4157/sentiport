#!/usr/bin/env python3
"""
新闻数据刷新脚本
使用 Alpha Vantage API 获取最新的股票新闻并存储到数据库
"""

import sys
from core_alpha_system import DatabaseManager, DataCollector
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数：刷新新闻数据"""

    print("=" * 60)
    print("📰 SentiPort 新闻数据刷新工具")
    print("=" * 60)
    print()

    # 初始化数据库和收集器
    logger.info("初始化数据库...")
    db = DatabaseManager()
    collector = DataCollector(db)

    # 获取股票列表
    try:
        import sqlite3
        import pandas as pd

        conn = sqlite3.connect(db.db_path)
        tickers_df = pd.read_sql("SELECT DISTINCT ticker FROM stock_prices", conn)
        conn.close()

        if tickers_df.empty:
            logger.warning("⚠️ 数据库中没有股票数据")
            logger.info("正在使用默认股票列表...")
            tickers = collector.get_sp500_tickers(limit=5)
        else:
            tickers = tickers_df['ticker'].tolist()
            logger.info(f"✅ 从数据库获取到 {len(tickers)} 只股票")
    except Exception as e:
        logger.error(f"读取股票列表失败: {e}")
        logger.info("使用默认股票列表...")
        tickers = collector.get_sp500_tickers(limit=5)

    # 显示将要获取的股票
    print(f"\n将获取以下股票的新闻数据:")
    print(f"  {', '.join(tickers[:10])}" + (" ..." if len(tickers) > 10 else ""))
    print()

    # 询问是否强制刷新
    force_refresh = False
    try:
        user_input = input("是否强制刷新（忽略缓存）? [y/N]: ").strip().lower()
        force_refresh = user_input in ['y', 'yes']
    except KeyboardInterrupt:
        print("\n\n❌ 用户取消操作")
        sys.exit(0)

    print()
    logger.info("🚀 开始获取新闻数据...")
    print()

    # 收集新闻
    success = collector.batch_collect_news(tickers, force_refresh=force_refresh)

    print()
    if success:
        logger.info("✅ 新闻数据刷新完成！")
        logger.info(f"数据库路径: {db.db_path}")
        logger.info("现在可以启动 Streamlit 界面查看新闻了")
        print()
        print("启动命令:")
        print("  streamlit run demo_streamlit.py")
    else:
        logger.error("❌ 新闻数据获取失败")
        logger.info("请检查网络连接和 API Key 是否正确")

    print()
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ 用户取消操作")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序运行出错: {e}", exc_info=True)
        sys.exit(1)
