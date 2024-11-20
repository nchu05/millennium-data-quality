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

    def run_backtest(self, orders: List[Dict[str, Any]], data: pd.DataFrame) -> Dict[str, Any]:
        portfolio_values = []
        cash = self.initial_cash
        holdings = {}

        for order in orders:
            date = order["date"]
            ticker = order["ticker"]
            close_price = data.at[date, ticker]
            quantity = order["quantity"]

            if order["type"] == "BUY":
                cash_needed = close_price * quantity
                if cash >= cash_needed:
                    cash -= cash_needed
                    holdings[ticker] = holdings.get(ticker, 0) + quantity
                else:
                    print(f"{date}: Insufficient cash to buy {quantity} shares of {ticker}. Skipping order.")
            elif order["type"] == "SELL":
                if holdings.get(ticker, 0) >= quantity:
                    cash += close_price * quantity
                    holdings[ticker] -= quantity
                else:
                    print(f"{date}: Cannot sell more than holdings for {ticker}. Skipping order.")

            total_value = cash
            for h_ticker, h_quantity in holdings.items():
                total_value += data.at[date, h_ticker] * h_quantity

            print(f"{date}: Portfolio Value - {total_value:.2f}")
            portfolio_values.append((date, total_value))

        return {"portfolio_values": pd.DataFrame(portfolio_values, columns=["Date", "Portfolio Value"]).set_index("Date")}