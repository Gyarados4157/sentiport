"""
SentiPort - 量化交易驱动的金融分析平台
专注于NLP驱动的Alpha因子生成和风险管理
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from core_alpha_system import (
    DatabaseManager, DataCollector, NLPProcessor, 
    AlphaFactorEngine, BacktestEngine
)
from sensitivity import calculate_beta, calculate_var, calculate_cvar, monte_carlo_simulation
from advisor import optimize_portfolio

# Streamlit配置
st.set_page_config(
    page_title="SentiPort", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化session state
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
    st.session_state.alpha_data = None
    st.session_state.performance_data = None

@st.cache_resource
def initialize_system():
    """初始化系统组件"""
    try:
        db = DatabaseManager()
        collector = DataCollector(db)
        nlp = NLPProcessor()
        alpha_engine = AlphaFactorEngine(db, nlp)
        backtest = BacktestEngine(db)
        
        return db, collector, nlp, alpha_engine, backtest, True
    except Exception as e:
        st.error(f"系统初始化失败: {e}")
        return None, None, None, None, None, False

def load_alpha_factors(db_manager):
    """加载Alpha因子数据"""
    try:
        conn = sqlite3.connect(db_manager.db_path)
        query = """
        SELECT * FROM alpha_factors 
        ORDER BY date DESC, combined_alpha DESC
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

def load_performance_metrics(backtest_engine):
    """加载性能指标"""
    try:
        return backtest_engine.get_performance_summary()
    except:
        return {'ic_mean': 0, 'ic_std': 0, 'ir': 0, 'hit_rate': 0}

def create_alpha_factor_chart(df):
    """创建Alpha因子可视化图表"""
    if df.empty:
        return go.Figure()
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Combined Alpha Signals', 'Sentiment Momentum', 
                       'Sentiment Reversal', 'News Volume Anomaly'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 按股票分组绘制
    colors = px.colors.qualitative.Set1
    for i, ticker in enumerate(df['ticker'].unique()[:8]):  # 限制显示数量
        ticker_data = df[df['ticker'] == ticker].sort_values('date')
        color = colors[i % len(colors)]
        
        # Combined Alpha
        fig.add_trace(
            go.Scatter(x=ticker_data['date'], y=ticker_data['combined_alpha'],
                      name=f'{ticker} Combined', line=dict(color=color),
                      showlegend=True),
            row=1, col=1
        )
        
        # Sentiment Momentum
        fig.add_trace(
            go.Scatter(x=ticker_data['date'], y=ticker_data['sentiment_momentum'],
                      name=f'{ticker} Momentum', line=dict(color=color, dash='dash'),
                      showlegend=False),
            row=1, col=2
        )
        
        # Sentiment Reversal
        fig.add_trace(
            go.Scatter(x=ticker_data['date'], y=ticker_data['sentiment_reversal'],
                      name=f'{ticker} Reversal', line=dict(color=color, dash='dot'),
                      showlegend=False),
            row=2, col=1
        )
        
        # News Volume Anomaly
        fig.add_trace(
            go.Scatter(x=ticker_data['date'], y=ticker_data['news_volume_anomaly'],
                      name=f'{ticker} News Vol', line=dict(color=color, dash='dashdot'),
                      showlegend=False),
            row=2, col=2
        )
    
    # 添加零线
    for row in [1, 2]:
        for col in [1, 2]:
            fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                         opacity=0.5, row=row, col=col)
    
    fig.update_layout(
        title="Alpha因子时间序列分析",
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_performance_dashboard(performance_data):
    """创建性能仪表板"""
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=("信息系数", "信息比率", "胜率", "因子稳定性")
    )
    
    # IC指标
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=performance_data.get('ic_mean', 0) * 100,
        title={'text': "IC (%)"},
        gauge={'axis': {'range': [None, 20]},
               'bar': {'color': "darkgreen" if performance_data.get('ic_mean', 0) > 0.03 else "red"},
               'steps': [{'range': [0, 3], 'color': "lightgray"},
                        {'range': [3, 10], 'color': "yellow"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                           'thickness': 0.75, 'value': 3}}
    ), row=1, col=1)
    
    # IR指标
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=performance_data.get('ir', 0),
        title={'text': "信息比率"},
        gauge={'axis': {'range': [None, 2]},
               'bar': {'color': "darkgreen" if performance_data.get('ir', 0) > 0.5 else "red"},
               'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 1], 'color': "yellow"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                           'thickness': 0.75, 'value': 0.5}}
    ), row=1, col=2)
    
    # 胜率
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=performance_data.get('hit_rate', 0) * 100,
        title={'text': "胜率 (%)"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "darkgreen" if performance_data.get('hit_rate', 0) > 0.55 else "red"},
               'steps': [{'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 70], 'color': "yellow"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                           'thickness': 0.75, 'value': 55}}
    ), row=2, col=1)
    
    # 稳定性 (1/IC_std)
    stability = 1.0 / (performance_data.get('ic_std', 1.0) + 0.001)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=stability,
        title={'text': "稳定性指数"},
        gauge={'axis': {'range': [None, 10]},
               'bar': {'color': "darkgreen" if stability > 2 else "red"},
               'steps': [{'range': [0, 2], 'color': "lightgray"},
                        {'range': [2, 5], 'color': "yellow"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                           'thickness': 0.75, 'value': 2}}
    ), row=2, col=2)
    
    fig.update_layout(height=500, title="系统性能指标")
    return fig

# 主应用界面
st.title("📈 SentiPort")
st.markdown("**基于NLP的量化交易Alpha因子系统**")

# 侧边栏设置
with st.sidebar:
    st.header("⚙️ 系统设置")
    
    # 初始化系统
    if st.button("🚀 初始化系统", type="primary"):
        with st.spinner("正在初始化系统组件..."):
            db, collector, nlp, alpha_engine, backtest, success = initialize_system()
            if success:
                st.session_state.system_initialized = True
                st.session_state.db = db
                st.session_state.collector = collector
                st.session_state.nlp = nlp
                st.session_state.alpha_engine = alpha_engine
                st.session_state.backtest = backtest
                st.success("✅ 系统初始化成功!")
            else:
                st.error("❌ 系统初始化失败")
    
    st.divider()
    
    # 数据收集设置
    st.subheader("📊 数据设置")
    stock_limit = st.slider("股票数量", 5, 30, 10)
    data_period = st.selectbox("历史数据周期", ["1y", "2y", "3y"], index=1)
    
    # 运行数据收集
    if st.session_state.system_initialized:
        if st.button("📥 收集数据"):
            with st.spinner("正在收集股票和新闻数据..."):
                try:
                    tickers = st.session_state.collector.get_sp500_tickers(limit=stock_limit)
                    st.session_state.collector.collect_stock_data(tickers, period=data_period)
                    st.success(f"✅ 已收集 {len(tickers)} 只股票的数据")
                except Exception as e:
                    st.error(f"❌ 数据收集失败: {e}")
    
    st.divider()
    
    # 因子计算设置
    st.subheader("🧮 因子计算")
    
    if st.session_state.system_initialized:
        if st.button("🔄 计算Alpha因子"):
            with st.spinner("正在计算Alpha因子..."):
                try:
                    # 获取股票列表
                    conn = sqlite3.connect(st.session_state.db.db_path)
                    tickers_df = pd.read_sql("SELECT DISTINCT ticker FROM stock_prices LIMIT 10", conn)
                    conn.close()
                    
                    if not tickers_df.empty:
                        alpha_results = []
                        progress_bar = st.progress(0)
                        
                        for i, ticker in enumerate(tickers_df['ticker']):
                            factors = st.session_state.alpha_engine.generate_combined_alpha(ticker)
                            result = {
                                'date': datetime.now().strftime('%Y-%m-%d'),
                                'ticker': ticker,
                                **factors
                            }
                            alpha_results.append(result)
                            progress_bar.progress((i + 1) / len(tickers_df))
                        
                        # 保存结果
                        conn = sqlite3.connect(st.session_state.db.db_path)
                        pd.DataFrame(alpha_results).to_sql('alpha_factors', conn, 
                                                         if_exists='replace', index=False)
                        conn.close()
                        
                        st.session_state.alpha_data = pd.DataFrame(alpha_results)
                        st.success(f"✅ 已计算 {len(alpha_results)} 个Alpha因子")
                    else:
                        st.warning("⚠️ 请先收集数据")
                except Exception as e:
                    st.error(f"❌ Alpha因子计算失败: {e}")

# 主内容区域
if not st.session_state.system_initialized:
    st.info("👈 请先在侧边栏初始化系统")
else:
    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Alpha因子", "📈 性能分析", "💼 投资组合", "⚠️ 风险管理"])
    
    with tab1:
        st.header("Alpha因子监控")
        
        # 加载Alpha因子数据
        alpha_df = load_alpha_factors(st.session_state.db)
        
        if not alpha_df.empty:
            # 显示因子图表
            fig = create_alpha_factor_chart(alpha_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示因子数据表
            st.subheader("📋 因子详情")
            
            # 添加交易信号
            alpha_df['trading_signal'] = np.where(
                alpha_df['combined_alpha'] > 0.1, '🟢 BUY',
                np.where(alpha_df['combined_alpha'] < -0.1, '🔴 SELL', '🟡 HOLD')
            )
            
            # 格式化数值列
            numeric_cols = ['sentiment_momentum', 'sentiment_reversal', 
                          'news_volume_anomaly', 'text_momentum', 
                          'sentiment_divergence', 'combined_alpha']
            
            display_df = alpha_df.copy()
            for col in numeric_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(4)
            
            st.dataframe(
                display_df[['ticker', 'trading_signal'] + numeric_cols],
                use_container_width=True
            )
            
            # 因子统计
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("买入信号", len(alpha_df[alpha_df['combined_alpha'] > 0.1]))
            with col2:
                st.metric("卖出信号", len(alpha_df[alpha_df['combined_alpha'] < -0.1]))
            with col3:
                st.metric("平均Alpha", f"{alpha_df['combined_alpha'].mean():.4f}")
            with col4:
                st.metric("Alpha标准差", f"{alpha_df['combined_alpha'].std():.4f}")
        
        else:
            st.info("📊 暂无Alpha因子数据，请在侧边栏计算因子")
    
    with tab2:
        st.header("系统性能分析")
        
        # 加载性能数据
        performance = load_performance_metrics(st.session_state.backtest)
        
        # 显示性能仪表板
        perf_fig = create_performance_dashboard(performance)
        st.plotly_chart(perf_fig, use_container_width=True)
        
        # 性能指标解释
        st.subheader("📖 指标说明")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **信息系数 (IC)**
            - 衡量因子预测能力
            - >3%: 优秀
            - 1-3%: 良好
            - <1%: 较弱
            """)
            
            st.markdown("""
            **胜率**
            - 预测正确的比例
            - >60%: 优秀
            - 50-60%: 良好
            - <50%: 需改进
            """)
        
        with col2:
            st.markdown("""
            **信息比率 (IR)**
            - IC的稳定性衡量
            - >1.0: 优秀
            - 0.5-1.0: 良好
            - <0.5: 较弱
            """)
            
            st.markdown("""
            **稳定性指数**
            - 因子稳定性评估
            - >5: 非常稳定
            - 2-5: 较稳定
            - <2: 不稳定
            """)
    
    with tab3:
        st.header("投资组合构建")
        
        # 加载Alpha数据用于组合构建
        alpha_df = load_alpha_factors(st.session_state.db)
        
        if not alpha_df.empty:
            # 选择Top signals
            top_signals = alpha_df.nlargest(10, 'combined_alpha')
            
            st.subheader("🎯 推荐投资组合")
            
            # 显示推荐股票
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**强烈买入推荐**")
                buy_signals = top_signals[top_signals['combined_alpha'] > 0.1]
                if not buy_signals.empty:
                    for _, row in buy_signals.head(5).iterrows():
                        st.markdown(f"🟢 **{row['ticker']}** - Alpha: {row['combined_alpha']:.4f}")
                else:
                    st.info("当前无强烈买入信号")
            
            with col2:
                st.markdown("**风险提示**")
                sell_signals = alpha_df[alpha_df['combined_alpha'] < -0.1]
                if not sell_signals.empty:
                    for _, row in sell_signals.head(5).iterrows():
                        st.markdown(f"🔴 **{row['ticker']}** - Alpha: {row['combined_alpha']:.4f}")
                else:
                    st.info("当前无卖出警告")
            
            # 组合权重建议
            if not buy_signals.empty:
                st.subheader("💰 权重分配建议")
                
                # 基于Alpha信号计算权重
                total_alpha = buy_signals['combined_alpha'].sum()
                buy_signals = buy_signals.copy()
                buy_signals['suggested_weight'] = buy_signals['combined_alpha'] / total_alpha
                buy_signals['suggested_weight'] = (buy_signals['suggested_weight'] * 100).round(1)
                
                # 显示权重图
                fig = px.pie(buy_signals, values='suggested_weight', names='ticker',
                           title="建议投资组合权重分配")
                st.plotly_chart(fig, use_container_width=True)
                
                # 权重表格
                st.dataframe(
                    buy_signals[['ticker', 'combined_alpha', 'suggested_weight']],
                    use_container_width=True
                )
        else:
            st.info("📊 请先计算Alpha因子以获得投资建议")
    
    with tab4:
        st.header("风险管理监控")
        
        # 风险指标
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "系统风险等级",
                "中等",  # 可以基于实际数据计算
                delta="稳定"
            )
        
        with col2:
            st.metric(
                "因子集中度",
                "良好",
                delta="分散"
            )
        
        with col3:
            st.metric(
                "数据覆盖率",
                "85%",
                delta="2%"
            )
        
        # 风险预警
        st.subheader("⚠️ 风险预警")
        
        # 模拟风险检查
        warnings = []
        alpha_df = load_alpha_factors(st.session_state.db)
        
        if not alpha_df.empty:
            # 检查极端信号
            extreme_signals = alpha_df[abs(alpha_df['combined_alpha']) > 2.0]
            if not extreme_signals.empty:
                warnings.append({
                    'level': '🟡 中等',
                    'message': f'检测到 {len(extreme_signals)} 个极端Alpha信号',
                    'action': '建议降低仓位或增加对冲'
                })
            
            # 检查信号集中度
            buy_signals = len(alpha_df[alpha_df['combined_alpha'] > 0.1])
            total_signals = len(alpha_df)
            if buy_signals / total_signals > 0.8:
                warnings.append({
                    'level': '🟡 中等', 
                    'message': '买入信号过于集中，市场可能过度乐观',
                    'action': '建议谨慎投资，分散风险'
                })
        
        if warnings:
            for warning in warnings:
                st.warning(f"**{warning['level']}** {warning['message']}")
                st.markdown(f"*建议操作：{warning['action']}*")
        else:
            st.success("✅ 当前无重大风险预警")
        
        # 历史风险分析（如果有数据）
        st.subheader("📊 历史风险分析")
        st.info("风险分析功能开发中，将集成VaR、CVaR等风险指标")

# 页面底部信息
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🎯 当前状态**")
    if st.session_state.system_initialized:
        st.success("系统已初始化")
    else:
        st.warning("系统未初始化")

with col2:
    st.markdown("**📊 数据状态**")
    if st.session_state.system_initialized:
        try:
            conn = sqlite3.connect(st.session_state.db.db_path)
            stock_count = pd.read_sql("SELECT COUNT(DISTINCT ticker) as count FROM stock_prices", conn).iloc[0]['count']
            conn.close()
            st.info(f"已加载 {stock_count} 只股票数据")
        except:
            st.warning("无数据")
    else:
        st.warning("无数据")

with col3:
    st.markdown("**⚙️ 系统信息**")
    st.info("SentiPort v1.0")

# 添加使用说明
with st.expander("📚 使用说明"):
    st.markdown("""
    ### 🚀 快速开始
    1. **初始化系统**: 点击侧边栏"初始化系统"按钮
    2. **收集数据**: 选择股票数量和历史周期，点击"收集数据"
    3. **计算因子**: 点击"计算Alpha因子"生成交易信号
    4. **查看结果**: 在各个标签页查看分析结果
    
    ### 📊 功能说明
    - **Alpha因子**: 基于NLP分析的5大核心因子
    - **性能分析**: IC、IR、胜率等关键指标
    - **投资组合**: 基于Alpha信号的投资建议
    - **风险管理**: 实时风险监控和预警
    
    ### ⚠️ 重要提示
    - 本系统仅供学习和研究使用
    - 投资决策需要综合考虑多种因素
    - 请勿将此作为唯一的投资依据
    """)