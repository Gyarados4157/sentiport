import pandas as pd
import numpy as np

def optimize_portfolio(prices: pd.DataFrame, risk_method: str = 'MV', cvar_alpha: float = 0.95) -> dict:
    """
    简化的投资组合优化，使用等权重或基于收益率的权重分配
    risk_method: 'MV' 或 'equal'
    """
    if prices.empty:
        return {}
    
    # 计算收益率
    returns = prices.pct_change().dropna()
    
    if risk_method == 'equal':
        # 等权重
        n_assets = len(prices.columns)
        weights = {ticker: 1.0/n_assets for ticker in prices.columns}
    else:
        # 基于夏普比率的简单权重分配
        mean_returns = returns.mean()
        volatilities = returns.std()
        
        # 避免除零
        volatilities = volatilities.replace(0, 1e-6)
        
        # 简单的风险调整收益
        risk_adjusted_returns = mean_returns / volatilities
        risk_adjusted_returns = risk_adjusted_returns.fillna(0)
        
        # 标准化为权重
        if risk_adjusted_returns.sum() > 0:
            weights = (risk_adjusted_returns / risk_adjusted_returns.sum()).to_dict()
        else:
            # 如果都是负值或零，使用等权重
            n_assets = len(prices.columns)
            weights = {ticker: 1.0/n_assets for ticker in prices.columns}
    
    # 确保权重和为1
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v/total_weight for k, v in weights.items()}
    
    return weights