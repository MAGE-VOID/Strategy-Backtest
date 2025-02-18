# visualization/plot.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def plot_equity_balance(result_backtest: dict):
    equity_data = pd.DataFrame(result_backtest["equity_over_time"])
    equity_data["date"] = pd.to_datetime(equity_data["date"])
    plt.figure(figsize=(10, 6))
    plt.plot(equity_data["date"], equity_data["equity"], label="Equity", color="green")
    plt.plot(equity_data["date"], equity_data["balance"], label="Balance", color="blue")
    plt.title("Equity and Balance Over Time")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_price_with_signals(result_backtest: dict, df: pd.DataFrame, symbol: str):
    trades = pd.DataFrame(result_backtest["trades"])
    price_data = df[df["Symbol"] == symbol].copy()
    price_data.index = pd.to_datetime(price_data.index)
    trades["open_time"] = pd.to_datetime(trades["open_time"])
    trades["close_time"] = pd.to_datetime(trades["close_time"])
    plt.figure(figsize=(12, 8))
    plt.plot(price_data.index, price_data["Close"], label=f"{symbol} Price", color="blue")
    for _, trade in trades.iterrows():
        if trade["symbol"] == symbol:
            if trade["status"] == "open" and trade["type"] == "long":
                plt.scatter(trade["open_time"], trade["entry"], color="green", marker="^", s=100,
                            label="Buy (Open)" if "Buy (Open)" not in plt.gca().get_legend_handles_labels()[1] else "")
            elif trade["status"] == "closed" and trade["type"] == "long":
                plt.scatter(trade["close_time"], trade["exit"], color="red", marker="v", s=100,
                            label="Buy (Closed)" if "Buy (Closed)" not in plt.gca().get_legend_handles_labels()[1] else "")
            elif trade["status"] == "open" and trade["type"] == "short":
                plt.scatter(trade["open_time"], trade["entry"], color="orange", marker="v", s=100,
                            label="Sell (Open)" if "Sell (Open)" not in plt.gca().get_legend_handles_labels()[1] else "")
            elif trade["status"] == "closed" and trade["type"] == "short":
                plt.scatter(trade["close_time"], trade["exit"], color="purple", marker="^", s=100,
                            label="Sell (Closed)" if "Sell (Closed)" not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.ylim(price_data["Close"].min() * 0.95, price_data["Close"].max() * 1.05)
    plt.title(f"{symbol} Price with Buy and Sell Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.show()
