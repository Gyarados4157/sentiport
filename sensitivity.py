import numpy as np
import pandas as pd


def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    计算资产相对于市场的贝塔值（系统性风险）。
    Beta = Cov(R_asset, R_market) / Var(R_market)
    """
    cov = np.cov(asset_returns, market_returns)[0, 1]
    var = np.var(market_returns)
    return cov / var


def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    历史 VaR
    """
    return -np.percentile(returns, (1 - confidence) * 100)


def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    条件 VaR
    """
    var_level = np.percentile(returns, (1 - confidence) * 100)
    tail_losses = returns[returns <= var_level]
    return -tail_losses.mean()


def monte_carlo_simulation(returns: pd.Series, num_simulations: int = 1000, horizon: int = 252) -> np.ndarray:
    """
    蒙特卡洛模拟对数收益序列
    返回 size=(horizon, num_simulations) 的累计收益路径
    """
    mu = returns.mean()
    sigma = returns.std()
    sims = np.exp(
        np.random.normal(mu / horizon, sigma / np.sqrt(horizon), (horizon, num_simulations))
    ).cumprod(axis=0)
    return sims
