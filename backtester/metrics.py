from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict

class Metrics(ABC):
    """Interface for calculating portfolio metrics."""
    
    @abstractmethod
    def calculate(self, portfolio_values: pd.Series, returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """Calculate performance metrics from portfolio values and returns."""
        pass


class ExtendedMetrics(Metrics):
    """Extended metrics calculator implementation."""
    
    def calculate(self, portfolio_values: pd.Series, returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict[str, float]:
        metrics = {}
        
        metrics['Daily Return'] = returns.mean()
        metrics['Cumulative Return'] = (1 + returns).prod() - 1
        metrics['Log Return'] = np.log(1 + returns).mean()

        metrics['Volatility'] = returns.std() * np.sqrt(252)  # annualize volatility

        if benchmark_returns is not None:
            metrics['Information Coefficient'] = returns.corr(benchmark_returns)
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

        # TODO: Implement beta and alpha calculation
        
        # if benchmark_returns is not None:
        #     cov_matrix = np.cov(returns, benchmark_returns)
        #     # beta in simple terms is the sensitivity of the stock to the benchmark
        #     # alpha is the excess return of the stock over the benchmark, so this is our "edge"
        #     metrics['Beta'] = cov_matrix[0, 1] / cov_matrix[1, 1] 
        #     metrics['Alpha'] = (returns.mean() * 252) - (metrics['Beta'] * benchmark_returns.mean() * 252)
        # else:
        #     metrics['Beta'] = None
        #     metrics['Alpha'] = None

        return metrics
