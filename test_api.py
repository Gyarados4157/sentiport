#!/usr/bin/env python3
"""
测试 Alpha Vantage API 连接
"""

import requests
import json

API_KEY = "CR9P7L6SGO9W1L8V"
TICKER = "AAPL"

print("=" * 60)
print("🧪 测试 Alpha Vantage API 连接")
print("=" * 60)
print()

url = "https://www.alphavantage.co/query"
params = {
    'function': 'NEWS_SENTIMENT',
    'tickers': TICKER,
    'apikey': API_KEY,
    'limit': 5
}

print(f"📡 请求 API: {TICKER}")
print(f"🔑 API Key: {API_KEY}")
print()

try:
    response = requests.get(url, params=params, timeout=15)
    print(f"✅ HTTP 状态码: {response.status_code}")
    print()

    if response.status_code == 200:
        data = response.json()

        # 检查错误信息
        if 'Note' in data:
            print(f"⚠️ API 限制: {data['Note']}")
        elif 'Information' in data:
            print(f"⚠️ API 信息: {data['Information']}")
        elif 'Error Message' in data:
            print(f"❌ API 错误: {data['Error Message']}")
        elif 'feed' in data:
            print(f"✅ 成功获取 {len(data['feed'])} 条新闻")
            print()
            print("📰 最新新闻示例:")
            print("-" * 60)

            for i, item in enumerate(data['feed'][:3], 1):
                print(f"\n{i}. {item.get('title', 'N/A')}")
                print(f"   来源: {item.get('source', 'N/A')}")
                print(f"   时间: {item.get('time_published', 'N/A')}")
                print(f"   URL: {item.get('url', 'N/A')[:80]}...")

                # 情感分数
                sentiments = item.get('ticker_sentiment', [])
                for ts in sentiments:
                    if ts.get('ticker') == TICKER:
                        score = ts.get('ticker_sentiment_score', 0)
                        print(f"   情感评分: {score}")
                        break
        else:
            print("❌ 未知的响应格式")
            print(json.dumps(data, indent=2))
    else:
        print(f"❌ HTTP 错误: {response.status_code}")
        print(response.text[:500])

except requests.exceptions.Timeout:
    print("⏰ 请求超时")
except requests.exceptions.RequestException as e:
    print(f"❌ 网络错误: {e}")
except Exception as e:
    print(f"❌ 错误: {e}")

print()
print("=" * 60)
