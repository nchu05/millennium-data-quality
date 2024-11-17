import pandas as pd
from data_source import YahooFinanceDataSource
from order_generator import MeanReversionOrderGenerator
from backtest_engine import SimpleBacktestEngine
from metrics import ExtendedMetrics

def main():
    data_source = YahooFinanceDataSource()
    order_generator = MeanReversionOrderGenerator()
    backtest_engine = SimpleBacktestEngine(initial_cash=100000)
    metrics_calculator = ExtendedMetrics()

    # run sim on 13 year history of just AAPL, 100 day mean reversion strategy
    data = data_source.get_historical_data("AAPL", "2011-01-01", "2024-01-01")
    orders = order_generator.generate_orders(data)

    backtest_results = backtest_engine.run_backtest(orders, data)
    portfolio_values = backtest_results["portfolio_values"]["Portfolio Value"]
    returns = portfolio_values.pct_change().dropna()

    benchmark_data = data_source.get_historical_data("SPY", "2011-01-01", "2024-01-01")
    benchmark_returns = benchmark_data['Close'].pct_change().dropna()
    
    metrics = metrics_calculator.calculate(portfolio_values, returns, benchmark_returns)
    print("Backtest Metrics:", metrics)

if __name__ == "__main__":
    main()
