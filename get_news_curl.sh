#!/bin/bash

# 使用 curl 获取新闻数据的脚本
API_KEY="CR9P7L6SGO9W1L8V"
DB_PATH="financial_data.db"

echo "============================================================================================"
echo "📰 获取真实新闻数据"
echo "============================================================================================"
echo ""

# 创建临时目录
TMP_DIR="/tmp/sentiport_news"
mkdir -p "$TMP_DIR"

# 股票列表
TICKERS=("AAPL" "MSFT" "GOOGL")

echo "📊 将获取以下股票的新闻: ${TICKERS[@]}"
echo ""

for i in "${!TICKERS[@]}"; do
    ticker="${TICKERS[$i]}"
    num=$((i + 1))
    total=${#TICKERS[@]}

    echo "[$num/$total] 正在获取 $ticker 的新闻..."

    # 使用 curl 获取数据
    response_file="$TMP_DIR/${ticker}_news.json"

    curl -s "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=$ticker&apikey=$API_KEY&limit=50" \
        -o "$response_file"

    if [ $? -eq 0 ] && [ -s "$response_file" ]; then
        echo "  ✅ 成功获取数据"

        # 使用 Python 处理 JSON 并插入数据库
        python3 << EOF
import json
import sqlite3
from datetime import datetime

try:
    # 读取 JSON
    with open('$response_file', 'r') as f:
        data = json.load(f)

    if 'feed' not in data:
        if 'Note' in data:
            print("  ⚠️  API 限制: " + data['Note'])
        else:
            print("  ⚠️  没有新闻数据")
    else:
        # 连接数据库
        conn = sqlite3.connect('$DB_PATH')
        cursor = conn.cursor()

        added = 0
        for item in data['feed'][:30]:
            # 解析日期
            time_pub = item.get('time_published', '')
            if len(time_pub) >= 8:
                date_str = f"{time_pub[:4]}-{time_pub[4:6]}-{time_pub[6:8]}"
            else:
                date_str = datetime.now().strftime('%Y-%m-%d')

            # 获取情感分数
            sentiment = 0.0
            for ts in item.get('ticker_sentiment', []):
                if ts.get('ticker') == '$ticker':
                    try:
                        sentiment = float(ts.get('ticker_sentiment_score', 0.0))
                    except:
                        pass
                    break

            title = item.get('title', '')[:500]
            summary = item.get('summary', '')[:1000]
            url = item.get('url', '')
            source = item.get('source', 'Unknown')

            # 检查是否已存在
            cursor.execute(
                "SELECT COUNT(*) FROM news_data WHERE ticker = ? AND title = ?",
                ('$ticker', title)
            )

            if cursor.fetchone()[0] == 0:
                cursor.execute(
                    "INSERT INTO news_data (date, ticker, title, summary, url, source, sentiment_score) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (date_str, '$ticker', title, summary, url, source, sentiment)
                )
                added += 1

        conn.commit()
        conn.close()
        print(f"  ✅ 添加了 {added} 条新闻")

except Exception as e:
    print(f"  ❌ 处理错误: {e}")
EOF

    else
        echo "  ❌ 获取失败"
    fi

    # API 限流
    if [ $num -lt $total ]; then
        echo "  ⏳ 等待 12 秒（API 限流）..."
        sleep 12
    fi

    echo ""
done

# 显示统计
echo "============================================================================================"
echo "📊 数据库新闻统计:"

sqlite3 "$DB_PATH" << EOF
SELECT ticker, COUNT(*) as count
FROM news_data
WHERE source != 'Simulated Dataset'
GROUP BY ticker;
EOF

echo ""
echo "✅ 新闻获取完成！"
echo ""
echo "现在可以运行: streamlit run demo_streamlit.py"
echo "============================================================================================"

# 清理临时文件
rm -rf "$TMP_DIR"
