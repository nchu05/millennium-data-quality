from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Dict, Any

class BacktestEngine(ABC):
    """Interface for backtesting a trading strategy."""
    
    def __init__(self, initial_cash: float):
        self.initial_cash = initial_cash
    
    @abstractmethod
    def run_backtest(self, orders: List[Dict[str, Any]], data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest simulation given orders and historical data."""
        pass

class SimpleBacktestEngine(BacktestEngine):
    """Simple backtest engine implementation."""
    """
    # TODO: Add checks so holdings can't go negative or cash can't go negative
    # TODO: Add support for short selling
    """
    
    def run_backtest(self, orders: List[Dict[str, Any]], data: pd.DataFrame) -> Dict[str, Any]:
        portfolio_values = []
        cash = self.initial_cash
        holdings = 0
        
        for order in orders:
            date = order["date"]
            close_price = data.loc[date, "Close"]
            quantity = order["quantity"]
            
            if order["type"] == "BUY":
                cash -= close_price * quantity
                holdings += quantity
            elif order["type"] == "SELL":
                if holdings >= quantity:  # can't sell more than have in this portfolio (no short selling for now since strategies are simpler)
                    cash += close_price * quantity
                    holdings -= quantity
                else:
                    print(f"{date}: Cannot sell more than holdings. Skipping order.")
                    continue
            
            portfolio_value = cash + holdings * close_price
            print(f"{date}: Portfolio Value - {portfolio_value:.2f}")
            portfolio_values.append((date, portfolio_value))
        
        return {"portfolio_values": pd.DataFrame(portfolio_values, columns=["Date", "Portfolio Value"]).set_index("Date")}

    def calculate(self, portfolio_values: pd.Series, returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict[str, float]:
        metrics = {}
        
        metrics['Daily Return'] = returns.mean()
        metrics['Cumulative Return'] = (1 + returns).prod() - 1
        metrics['Log Return'] = np.log(1 + returns).mean()
        metrics['Volatility'] = returns.std() * np.sqrt(252)  # annualize volatility
        
        if benchmark_returns is not None:
            aligned_returns, aligned_benchmark_returns = returns.align(benchmark_returns, join='inner')
            metrics['Information Coefficient'] = aligned_returns.corr(aligned_benchmark_returns)
        else:
            metrics['Information Coefficient'] = None
        
        risk_free_rate = 0.0045  
        excess_returns = returns - (risk_free_rate / 252)
        metrics['Sharpe Ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        running_max = portfolio_values.cummax()
        drawdown = (portfolio_values / running_max) - 1
        metrics['Max Drawdown'] = drawdown.min()
        
        # val at risk (VaR) 1 day horizon
        metrics['VaR 5%'] = returns.quantile(0.05)
        
        if benchmark_returns is not None:
            cov_matrix = np.cov(aligned_returns, aligned_benchmark_returns)
            metrics['Beta'] = cov_matrix[0, 1] / cov_matrix[1, 1]
            metrics['Alpha'] = (aligned_returns.mean() * 252) - (metrics['Beta'] * aligned_benchmark_returns.mean() * 252)
        else:
            metrics['Beta'] = None
            metrics['Alpha'] = None
        
        return metrics
