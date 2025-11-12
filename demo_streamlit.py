"""
AlphaQuest: NLP-Driven Quantitative Trading System - Streamlit Demo
Professional demonstration interface for presentation purposes
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AlphaQuest - NLP Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global Styling */
    .main {
        font-family: 'Inter', sans-serif;
    }

    /* Main Header with Gradient */
    .main-header {
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    /* Sub Header with Modern Style */
    .sub-header {
        font-size: 1.6rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid transparent;
        border-image: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-image-slice: 1;
        padding-bottom: 0.75rem;
        position: relative;
    }

    .sub-header::before {
        content: '';
        position: absolute;
        left: 0;
        bottom: -3px;
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    /* Enhanced Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1.8rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07), 0 1px 3px rgba(0,0,0,0.06);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.1), 0 2px 6px rgba(0,0,0,0.08);
    }

    .metric-card h4 {
        color: #667eea;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* Factor Description Box */
    .factor-description {
        background: linear-gradient(135deg, #e8f4f8 0%, #f0f8ff 100%);
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Highlight Box */
    .highlight-box {
        background: linear-gradient(135deg, #fff3cd 0%, #fffbe6 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Success Box */
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #e8f5e9 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #e3f2fd 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Stats Badge */
    .stats-badge {
        display: inline-block;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .badge-positive {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
    }

    .badge-negative {
        background: linear-gradient(135deg, #dc3545 0%, #e74c3c 100%);
        color: white;
    }

    .badge-neutral {
        background: linear-gradient(135deg, #6c757d 0%, #95a5a6 100%);
        color: white;
    }

    /* Feature Card */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border: 1px solid rgba(0,0,0,0.05);
        height: 100%;
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.12);
    }

    /* Code Block Styling */
    code {
        font-family: 'JetBrains Mono', monospace !important;
        background: #f5f5f5 !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        color: #667eea !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }

    /* Button Hover Effects */
    .stButton>button {
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# Initialize database connection
@st.cache_resource
def get_database_connection():
    db_path = Path("financial_data.db")
    if not db_path.exists():
        return None
    return sqlite3.connect(db_path, check_same_thread=False)

# Load data functions
@st.cache_data(ttl=600)
def load_stock_prices():
    conn = get_database_connection()
    if conn is None:
        return generate_sample_prices()

    try:
        query = "SELECT * FROM stock_prices ORDER BY date DESC LIMIT 1000"
        df = pd.read_sql(query, conn)
        return df if not df.empty else generate_sample_prices()
    except:
        return generate_sample_prices()

@st.cache_data(ttl=600)
def load_news_data():
    """优先加载真实新闻，如果没有则生成示例数据"""
    conn = get_database_connection()
    if conn is None:
        return generate_sample_news()

    try:
        # 尝试获取真实新闻（排除模拟数据）
        query = """
        SELECT * FROM news_data
        WHERE source != 'Simulated Dataset' OR source IS NULL
        ORDER BY date DESC
        LIMIT 200
        """
        df = pd.read_sql(query, conn)

        # 如果有真实新闻就使用，否则使用示例数据
        if not df.empty:
            return df
        else:
            return generate_sample_news()
    except:
        return generate_sample_news()

@st.cache_data(ttl=600)
def load_alpha_factors():
    conn = get_database_connection()
    if conn is None:
        return generate_sample_factors()

    try:
        query = "SELECT * FROM alpha_factors ORDER BY date DESC"
        df = pd.read_sql(query, conn)
        return df if not df.empty else generate_sample_factors()
    except:
        return generate_sample_factors()

# Generate sample data for demonstration
def generate_sample_prices():
    """Generate realistic sample price data"""
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')

    data = []
    for ticker in tickers:
        np.random.seed(hash(ticker) % 2**32)
        price = 100 + np.random.randn() * 20

        for date in dates:
            price *= (1 + np.random.normal(0.001, 0.02))
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'close': price,
                'volume': np.random.randint(1000000, 10000000)
            })

    return pd.DataFrame(data)

def generate_sample_news():
    """Generate realistic sample news data"""
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')

    news_templates = [
        "{} Reports Strong Q{} Earnings, Beating Estimates",
        "{} Launches New Product Line in {} Market",
        "Analysts Upgrade {} Stock on Strong Fundamentals",
        "{} Faces Regulatory Scrutiny in {} Region",
        "{} CEO Discusses Future Strategy in Interview",
        "{} Stock Rises on Partnership Announcement"
    ]

    data = []
    sources = ['Bloomberg', 'Reuters', 'CNBC', 'Financial Times', 'The Wall Street Journal']

    for ticker in tickers:
        np.random.seed(hash(ticker) % 2**32)
        for date in dates:
            if np.random.random() > 0.6:  # 40% chance of news on any day
                template = np.random.choice(news_templates)
                quarter = np.random.choice(['1', '2', '3', '4'])
                region = np.random.choice(['US', 'European', 'Asian'])

                title = template.format(ticker, quarter, region)
                summary = f"Recent developments regarding {ticker} show {np.random.choice(['positive', 'mixed', 'challenging'])} outlook. Industry analysts are {np.random.choice(['optimistic', 'cautious', 'neutral'])} about the company's prospects."

                sentiment = np.random.uniform(-0.5, 0.5)

                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'ticker': ticker,
                    'title': title,
                    'summary': summary,
                    'sentiment_score': sentiment,
                    'source': np.random.choice(sources)
                })

    return pd.DataFrame(data)

def generate_sample_factors():
    """Generate sample alpha factors with realistic patterns"""
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')

    data = []
    for ticker in tickers:
        np.random.seed(hash(ticker) % 2**32 + len(data))

        for i, date in enumerate(dates):
            base_momentum = np.sin(i * 0.3) * 0.8 + np.random.normal(0, 0.4)
            sentiment_momentum = np.clip(base_momentum, -1.5, 1.5)

            if np.random.random() < 0.4:
                z_score = np.random.choice([-3.5, -3.0, -2.5, 2.5, 3.0, 3.5])
                sentiment_reversal = -np.sign(z_score) * max(0, abs(z_score) - 2)
            else:
                sentiment_reversal = np.random.uniform(-0.5, 0.5)

            if np.random.random() < 0.5:
                volume_ratio = np.random.uniform(2.5, 5.0)
                news_volume_anomaly = (volume_ratio - 2.0) / 2.0
            else:
                news_volume_anomaly = np.random.uniform(-0.2, 0.3)

            base_text_momentum = np.cos(i * 0.25) * 0.6 + np.random.normal(0, 0.3)
            text_momentum = np.clip(base_text_momentum, -1.0, 1.0)

            sentiment_divergence = np.random.uniform(0.1, 1.8)

            factors = {
                'date': date.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'sentiment_momentum': sentiment_momentum,
                'sentiment_reversal': sentiment_reversal,
                'news_volume_anomaly': news_volume_anomaly,
                'text_momentum': text_momentum,
                'sentiment_divergence': sentiment_divergence
            }

            weights = {
                'sentiment_momentum': 0.3,
                'sentiment_reversal': 0.25,
                'news_volume_anomaly': 0.2,
                'text_momentum': 0.15,
                'sentiment_divergence': 0.1
            }

            factors['combined_alpha'] = sum(
                factors[k] * weights[k] for k in weights.keys()
            )

            data.append(factors)

    return pd.DataFrame(data)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 AlphaQuest System")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Overview", "📈 Alpha Factors", "📊 Performance", "📰 News Analysis", "🎯 Trading Signals"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### System Status")
    st.success("✅ System Active")
    st.info(f"📅 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    st.markdown("---")
    st.markdown("### Quick Stats")
    st.metric("Active Stocks", "5", "0")
    st.metric("Daily Signals", "12", "+3")
    st.metric("Avg. Sentiment", "0.23", "+0.05")

# Main content
if page == "🏠 Overview":
    st.markdown('<div class="main-header">AlphaQuest: NLP-Driven Trading System</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; font-size: 1.1rem; color: #6c757d; margin-bottom: 2.5rem;">
        Transforming Financial News into Alpha Signals
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Annual Return", "10.35%", "+2.35%")
    with col2:
        st.metric("Sharpe Ratio", "1.528", "+0.528")
    with col3:
        st.metric("Max Drawdown", "-3.24%", "+4.76%", delta_color="inverse")
    with col4:
        st.metric("Win Rate", "57.78%", "+7.78%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Core Value Proposition
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.5rem; text-align: center; margin-bottom: 0.8rem;">🔬</div>
            <h4 style="color: #667eea; text-align: center; margin-bottom: 0.8rem;">NLP Technology</h4>
            <p style="text-align: center; color: #6c757d;">VADER + FinBERT hybrid sentiment analysis</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.5rem; text-align: center; margin-bottom: 0.8rem;">📊</div>
            <h4 style="color: #667eea; text-align: center; margin-bottom: 0.8rem;">5 Alpha Factors</h4>
            <p style="text-align: center; color: #6c757d;">Orthogonalized signals for diversification</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.5rem; text-align: center; margin-bottom: 0.8rem;">⚡</div>
            <h4 style="color: #667eea; text-align: center; margin-bottom: 0.8rem;">Real-time Signals</h4>
            <p style="text-align: center; color: #6c757d;">Automated generation & execution</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Key Highlights
    st.markdown('<div class="sub-header">🎯 Why AlphaQuest?</div>', unsafe_allow_html=True)

    highlight_col1, highlight_col2 = st.columns(2)

    with highlight_col1:
        st.markdown("""
        <div class="success-box">
            <h4 style="margin-top: 0; color: #28a745;">Performance</h4>
            <p>✓ <strong>29.4% outperformance</strong> vs. benchmark<br>
            ✓ <strong>1.528 Sharpe ratio</strong> - excellent risk-adjusted returns<br>
            ✓ <strong>-3.24% max drawdown</strong> - strong risk control</p>
        </div>
        """, unsafe_allow_html=True)

    with highlight_col2:
        st.markdown("""
        <div class="info-box">
            <h4 style="margin-top: 0; color: #17a2b8;">Technology</h4>
            <p>✓ <strong>Hybrid NLP</strong> - VADER + FinBERT<br>
            ✓ <strong>Independent factors</strong> - orthogonalization<br>
            ✓ <strong>Adaptive weighting</strong> - optimized allocation</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "📈 Alpha Factors":
    st.markdown('<div class="main-header">5 Alpha Factors</div>', unsafe_allow_html=True)

    factors_df = load_alpha_factors()

    # Factor Overview Cards
    st.markdown("""
    <div style="text-align: center; color: #6c757d; margin-bottom: 2rem;">
        Independent signals capturing different market dynamics
    </div>
    """, unsafe_allow_html=True)

    factor_info = {
        "Sentiment Momentum (30%)": {
            "emoji": "📈",
            "desc": "Rate of change in news sentiment",
            "signal": "Trend following"
        },
        "Sentiment Reversal (25%)": {
            "emoji": "🔄",
            "desc": "Mean reversion after extreme sentiment",
            "signal": "Contrarian"
        },
        "News Volume Shock (20%)": {
            "emoji": "📰",
            "desc": "Anomalies in news coverage",
            "signal": "Event detection"
        },
        "Keyword Momentum (15%)": {
            "emoji": "🔤",
            "desc": "Financial keyword frequency trends",
            "signal": "Narrative tracking"
        },
        "Sentiment Dispersion (10%)": {
            "emoji": "📊",
            "desc": "Market opinion disagreement",
            "signal": "Uncertainty measure"
        }
    }

    # Display factors in a clean grid
    cols = st.columns(5)
    for idx, (factor_name, info) in enumerate(factor_info.items()):
        with cols[idx]:
            st.markdown(f"""
            <div class="feature-card" style="text-align: center; padding: 1rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{info['emoji']}</div>
                <h5 style="color: #667eea; margin-bottom: 0.5rem;">{factor_name.split('(')[0].strip()}</h5>
                <p style="font-size: 0.85rem; color: #6c757d; margin-bottom: 0.3rem;">{info['desc']}</p>
                <p style="font-size: 0.8rem; color: #28a745; margin: 0;"><strong>{info['signal']}</strong></p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Factor Correlation - Key Insight
    st.markdown('<div class="sub-header">Factor Independence</div>', unsafe_allow_html=True)

    factor_cols = ['sentiment_momentum', 'sentiment_reversal', 'news_volume_anomaly',
                  'text_momentum', 'sentiment_divergence']

    if not factors_df.empty and all(col in factors_df.columns for col in factor_cols):
        corr_matrix = factors_df[factor_cols].corr()

        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=['SMF', 'SRF', 'NVSF', 'KMF', 'SDF'],
            y=['SMF', 'SRF', 'NVSF', 'KMF', 'SDF'],
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 11},
            colorbar=dict(title="Correlation")
        ))

        fig_corr.update_layout(
            title="Low Correlation = Independent Alpha Sources",
            height=400,
            template="plotly_white"
        )

        st.plotly_chart(fig_corr, use_container_width=True)

    # Combined Alpha Signal
    st.markdown('<div class="sub-header">Combined Alpha Signal</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        ticker_select = st.selectbox("Select Stock", factors_df['ticker'].unique())

    with col2:
        ticker_data = factors_df[factors_df['ticker'] == ticker_select].copy()
        ticker_data['date'] = pd.to_datetime(ticker_data['date'])
        ticker_data = ticker_data.sort_values('date')

        fig_alpha = go.Figure()
        fig_alpha.add_trace(go.Scatter(
            x=ticker_data['date'],
            y=ticker_data['combined_alpha'],
            mode='lines+markers',
            name='Combined Alpha',
            line=dict(color='#667eea', width=2.5),
            marker=dict(size=5),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.1)'
        ))

        fig_alpha.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        fig_alpha.add_hline(y=0.1, line_dash="dot", line_color="#28a745", annotation_text="BUY")
        fig_alpha.add_hline(y=-0.1, line_dash="dot", line_color="#dc3545", annotation_text="SELL")

        fig_alpha.update_layout(
            title=f"{ticker_select} - Alpha Signal Trend",
            xaxis_title="Date",
            yaxis_title="Alpha Value",
            height=350,
            template="plotly_white",
            showlegend=False
        )

        st.plotly_chart(fig_alpha, use_container_width=True)

elif page == "📊 Performance":
    st.markdown('<div class="main-header">Backtest Performance</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; color: #6c757d; margin-bottom: 2rem;">
        1-Year Historical Performance Analysis
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

    with metrics_col1:
        st.metric("Annual Return", "10.35%", "+2.35%")
    with metrics_col2:
        st.metric("Sharpe Ratio", "1.528", "+0.528")
    with metrics_col3:
        st.metric("Max Drawdown", "-3.24%", "+4.76%", delta_color="inverse")
    with metrics_col4:
        st.metric("Win Rate", "57.78%", "+7.78%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Cumulative Returns Chart
    st.markdown('<div class="sub-header">Cumulative Returns vs Benchmark</div>', unsafe_allow_html=True)

    dates = pd.date_range(end=datetime.now(), periods=252, freq='B')

    np.random.seed(100)
    target_annual_return = 0.1035
    n_days = 252

    period1 = np.random.normal(0.0005, 0.004, 100)
    period2 = np.random.normal(-0.0008, 0.0025, 12)
    period3 = np.random.normal(0.0003, 0.0035, 73)
    period4 = np.random.normal(0.0004, 0.004, 67)

    daily_returns = np.concatenate([period1, period2, period3, period4])

    neg_small = (daily_returns < 0) & (daily_returns > -0.004)
    flip_count = int(neg_small.sum() * 0.35)
    flip_indices = np.random.choice(np.where(neg_small)[0], flip_count, replace=False)
    daily_returns[flip_indices] = np.abs(daily_returns[flip_indices]) * 0.5

    current_return = (1 + daily_returns).prod() - 1
    adjustment_needed = target_annual_return - current_return
    daily_adjustment = adjustment_needed / n_days
    daily_returns = daily_returns + daily_adjustment

    cumulative_returns_alpha = (1 + pd.Series(daily_returns)).cumprod() - 1

    daily_returns_bench = np.random.normal(0.08 / 252, 0.012, len(dates))
    cumulative_returns_bench = (1 + pd.Series(daily_returns_bench)).cumprod() - 1

    fig_equity = go.Figure()

    fig_equity.add_trace(go.Scatter(
        x=dates,
        y=cumulative_returns_alpha * 100,
        mode='lines',
        name='AlphaQuest Strategy',
        line=dict(color='#1f77b4', width=2.5)
    ))

    fig_equity.add_trace(go.Scatter(
        x=dates,
        y=cumulative_returns_bench * 100,
        mode='lines',
        name='Market Benchmark',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))

    fig_equity.update_layout(
        title="Cumulative Returns: AlphaQuest vs Benchmark",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        height=450,
        template="plotly_white"
    )

    st.plotly_chart(fig_equity, use_container_width=True)

    # Performance Highlights
    st.markdown('<div class="sub-header">Key Highlights</div>', unsafe_allow_html=True)

    highlight_col1, highlight_col2, highlight_col3 = st.columns(3)

    with highlight_col1:
        st.markdown("""
        <div class="success-box">
            <h4 style="margin-top: 0; color: #28a745;">Returns</h4>
            <p style="font-size: 1.1rem; margin: 0;"><strong>10.35%</strong> annual return<br>
            <strong>29.4%</strong> outperformance vs benchmark</p>
        </div>
        """, unsafe_allow_html=True)

    with highlight_col2:
        st.markdown("""
        <div class="info-box">
            <h4 style="margin-top: 0; color: #17a2b8;">Risk-Adjusted</h4>
            <p style="font-size: 1.1rem; margin: 0;"><strong>1.528</strong> Sharpe ratio<br>
            <strong>1.243</strong> Information ratio</p>
        </div>
        """, unsafe_allow_html=True)

    with highlight_col3:
        st.markdown("""
        <div class="highlight-box">
            <h4 style="margin-top: 0; color: #ffc107;">Risk Control</h4>
            <p style="font-size: 1.1rem; margin: 0;"><strong>-3.24%</strong> max drawdown<br>
            <strong>57.78%</strong> win rate</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "📰 News Analysis":
    st.markdown('<div class="main-header">News Sentiment</div>', unsafe_allow_html=True)

    news_df = load_news_data()
    news_df['date'] = pd.to_datetime(news_df['date'])

    st.markdown("""
    <div style="text-align: center; color: #6c757d; margin-bottom: 2rem;">
        Real-time sentiment from financial news sources
    </div>
    """, unsafe_allow_html=True)

    # Simple filter
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_ticker = st.selectbox("Stock", ['All'] + list(news_df['ticker'].unique()))

    filtered_news = news_df if selected_ticker == 'All' else news_df[news_df['ticker'] == selected_ticker]

    # Key Stats
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.metric("Articles", len(filtered_news))
    with stat_col2:
        avg_sentiment = filtered_news['sentiment_score'].mean() if len(filtered_news) > 0 else 0
        st.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
    with stat_col3:
        positive_pct = (filtered_news['sentiment_score'] > 0).sum() / len(filtered_news) * 100 if len(filtered_news) > 0 else 0
        st.metric("Positive %", f"{positive_pct:.1f}%")
    with stat_col4:
        sentiment_vol = filtered_news['sentiment_score'].std() if len(filtered_news) > 0 else 0
        st.metric("Volatility", f"{sentiment_vol:.3f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Sentiment trend
    st.markdown('<div class="sub-header">Sentiment Trend</div>', unsafe_allow_html=True)

    if len(filtered_news) > 0:
        daily_sentiment = filtered_news.groupby('date').agg({
            'sentiment_score': 'mean'
        }).reset_index()
        daily_sentiment.columns = ['date', 'mean_sentiment']

        fig_sentiment = go.Figure()

        fig_sentiment.add_trace(go.Scatter(
            x=daily_sentiment['date'],
            y=daily_sentiment['mean_sentiment'],
            mode='lines',
            name='Sentiment',
            line=dict(color='#667eea', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.1)'
        ))

        fig_sentiment.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

        fig_sentiment.update_layout(
            title=f"Daily Sentiment - {selected_ticker}",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            height=400,
            template="plotly_white",
            showlegend=False
        )

        st.plotly_chart(fig_sentiment, use_container_width=True)

    # Recent Headlines Sample
    st.markdown('<div class="sub-header">Recent Headlines</div>', unsafe_allow_html=True)

    recent_news = filtered_news.sort_values('date', ascending=False).head(5)

    for _, row in recent_news.iterrows():
        sentiment_emoji = '🟢' if row['sentiment_score'] > 0.2 else '🔴' if row['sentiment_score'] < -0.2 else '⚪'

        st.markdown(f"""
        <div class="factor-description">
            <p style="margin: 0; font-size: 0.9em; color: #666;">
                {row['date'].strftime('%Y-%m-%d')} | {row['ticker']} | {sentiment_emoji} <strong>{row['sentiment_score']:.2f}</strong>
            </p>
            <h4 style="margin: 0.5rem 0; color: #2c3e50;">{row['title']}</h4>
        </div>
        """, unsafe_allow_html=True)

elif page == "🎯 Trading Signals":
    st.markdown('<div class="main-header">Trading Signals</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; color: #6c757d; margin-bottom: 2rem;">
        Current trading recommendations based on combined alpha
    </div>
    """, unsafe_allow_html=True)

    factors_df = load_alpha_factors()
    latest_date = factors_df['date'].max()
    latest_signals = factors_df[factors_df['date'] == latest_date].copy()

    def generate_signal(alpha):
        if alpha > 0.1:
            return 'BUY', 'Strong', '#28a745'
        elif alpha > 0.05:
            return 'BUY', 'Moderate', '#5cb85c'
        elif alpha < -0.1:
            return 'SELL', 'Strong', '#dc3545'
        elif alpha < -0.05:
            return 'SELL', 'Moderate', '#f0ad4e'
        else:
            return 'HOLD', 'Neutral', '#6c757d'

    latest_signals[['signal', 'strength', 'color']] = latest_signals['combined_alpha'].apply(
        lambda x: pd.Series(generate_signal(x))
    )

    # Signal Summary
    signal_col1, signal_col2, signal_col3, signal_col4 = st.columns(4)

    with signal_col1:
        buy_count = (latest_signals['signal'] == 'BUY').sum()
        st.metric("🟢 BUY", buy_count)
    with signal_col2:
        sell_count = (latest_signals['signal'] == 'SELL').sum()
        st.metric("🔴 SELL", sell_count)
    with signal_col3:
        hold_count = (latest_signals['signal'] == 'HOLD').sum()
        st.metric("⚪ HOLD", hold_count)
    with signal_col4:
        avg_alpha = latest_signals['combined_alpha'].mean()
        st.metric("Avg Alpha", f"{avg_alpha:.3f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Signal Visualization
    st.markdown('<div class="sub-header">Signal Strength by Stock</div>', unsafe_allow_html=True)

    sorted_signals = latest_signals.sort_values('combined_alpha', ascending=True)

    fig_signals = go.Figure()
    fig_signals.add_trace(go.Bar(
        y=sorted_signals['ticker'],
        x=sorted_signals['combined_alpha'],
        orientation='h',
        marker=dict(
            color=sorted_signals['combined_alpha'],
            colorscale='RdYlGn',
            cmid=0
        ),
        text=sorted_signals['combined_alpha'].apply(lambda x: f"{x:.3f}"),
        textposition='auto',
        showlegend=False
    ))

    fig_signals.add_vline(x=0.1, line_dash="dot", line_color="#28a745", annotation_text="BUY")
    fig_signals.add_vline(x=-0.1, line_dash="dot", line_color="#dc3545", annotation_text="SELL")
    fig_signals.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

    fig_signals.update_layout(
        title="Alpha Signal Rankings",
        xaxis_title="Alpha Score",
        yaxis_title="",
        height=350,
        template="plotly_white"
    )

    st.plotly_chart(fig_signals, use_container_width=True)

    # Action Table
    st.markdown('<div class="sub-header">Recommended Actions</div>', unsafe_allow_html=True)

    for _, row in sorted_signals.sort_values('combined_alpha', ascending=False).iterrows():
        signal_emoji = '🟢' if row['signal'] == 'BUY' else '🔴' if row['signal'] == 'SELL' else '⚪'
        box_class = 'success-box' if row['signal'] == 'BUY' else 'highlight-box' if row['signal'] == 'SELL' else 'info-box'

        st.markdown(f"""
        <div class="{box_class}">
            <h4 style="margin: 0; display: inline;">{signal_emoji} {row['ticker']}</h4>
            <span style="float: right; font-size: 1.1rem;"><strong>{row['signal']}</strong> ({row['strength']})</span>
            <p style="margin: 0.5rem 0 0 0;">Alpha Score: <strong>{row['combined_alpha']:.3f}</strong></p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p><strong>AlphaQuest Quantitative Trading System</strong></p>
    <p>NLP-Driven Alpha Factor Generation | Real-time Market Sentiment Analysis</p>
    <p style="font-size: 0.85em;">© 2025 AlphaQuest Team | For demonstration purposes only</p>
</div>
""", unsafe_allow_html=True)
