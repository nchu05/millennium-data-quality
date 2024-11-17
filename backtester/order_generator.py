from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any

class OrderGenerator(ABC):
    """Interface for generating trade orders based on a strategy."""
    
    @abstractmethod
    def generate_orders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate orders given historical price data."""
        pass

class MeanReversionOrderGenerator(OrderGenerator):
    """Mean reversion strategy implementation of OrderGenerator."""
        
    def generate_orders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        orders = []
        tickers = data.columns
        
        for ticker in tickers:
            ticker_data = data[ticker].to_frame(name='Close')
            ticker_data['100_day_avg'] = ticker_data['Close'].rolling(window=100).mean()
            
            for date, row in ticker_data.iterrows():
                if row['Close'] < row['100_day_avg']:
                    orders.append({"date": date, "type": "BUY", "ticker": ticker, "quantity": 100})
                else:
                    orders.append({"date": date, "type": "SELL", "ticker": ticker, "quantity": 100})
            
        return orders