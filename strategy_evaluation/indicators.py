import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import util


def author():
    """
      :return: The GT username of the student
      :rtype: str
      """
    return "cjimenez36"


def get_sma(prices, lookback):
    """Calculate Simple Moving Average.
    """
    return prices.rolling(window=lookback, center=False).mean()


def get_volatility(prices, lookback):
    """Calculate Volatility as the standard deviation over the lookback period.
    """
    return prices.rolling(window=lookback, center=False).std()


def get_price_sma(prices, lookback):
    """Calculate Price/Simple Moving Average.
    """
    return prices / get_sma(prices, lookback)


def get_golden_cross_ratio(prices, lookback):
    """Calculate Golden Cross values as the difference between the short-term and long-term smas.
    """
    return get_sma(prices, lookback) / get_sma(prices, lookback * 4)


def get_golden_crosses(prices, lookback):
    """Calculate Golden Cross values as the difference between the short-term and long-term smas.
    """
    return get_sma(prices, lookback), get_sma(prices, lookback * 4)


def get_bollinger_band_percentage(prices, lookback):
    """Calculate Bollinger Band Percentage.
    """
    ma_price = get_sma(prices=prices, lookback=lookback)
    # Returns sample stdev by default, keeping as is since Sharpe ratio assumes sample calc is used.
    std_price = get_volatility(prices=prices, lookback=lookback)
    upper_band = ma_price + (std_price * 2)
    lower_band = ma_price - (std_price * 2)
    return (prices - lower_band) / (upper_band - lower_band)


def get_bollinger_bands(prices, lookback):
    """Calculate Bollinger Band Percentage.
    """
    ma_price = get_sma(prices=prices, lookback=lookback)
    # Returns sample stdev by default, keeping as is since Sharpe ratio assumes sample calc is used.
    std_price = get_volatility(prices=prices, lookback=lookback)
    upper_band = ma_price + (std_price * 2)
    lower_band = ma_price - (std_price * 2)
    return upper_band, lower_band


def get_momentum(prices, lookback):
    """Calculate Momentum.
    """
    return (prices / prices.shift(lookback)) - 1


def generate_indicator_charts(symbol="JPM",
                              lookback=20,
                              sd=dt.datetime(2008, 1, 1),
                              ed=dt.datetime(2009, 12, 31)):
    dates = pd.date_range(sd, ed)
    prices = util.get_data([symbol], dates)
    # Remove SPY because it is not needed.
    prices.drop(columns=["SPY"], inplace=True)
    # Sort index and normalize prices.
    prices.sort_index(inplace=True)
    prices = prices / prices.iloc[0]
    # Get indicators.
    indicators = prices.copy()
    indicators.rename(columns={symbol: "Price"}, inplace=True)
    indicators["Simple Moving Average"] = get_sma(prices=prices, lookback=lookback)
    indicators["Price/SMA"] = get_price_sma(prices=prices, lookback=lookback)
    indicators["Bollinger Band Percentage"] = get_bollinger_band_percentage(prices=prices, lookback=lookback)
    indicators["BBand Upper"], indicators["BBand Lower"] = get_bollinger_bands(prices=prices, lookback=lookback)
    indicators["Golden Cross Ratio"] = get_golden_cross_ratio(prices=prices, lookback=lookback)
    indicators["SMA({})".format(lookback)], indicators["SMA({})".format(lookback * 4)] = get_golden_crosses(prices,
                                                                                                            lookback)
    indicators["Momentum"] = get_momentum(prices=prices, lookback=lookback)
    indicators["Volatility"] = get_volatility(prices=prices, lookback=lookback)
    indicators.dropna(inplace=True)
    # Plot indicators.
    # Price/SMA.
    indicators[["Price", "Simple Moving Average", "Price/SMA"]].plot()
    plt.title("Price/SMA")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.savefig('price_sma.png')
    plt.close()
    # Bollinger Band Percentage.
    fig, axes = plt.subplots(2)
    fig.suptitle("Bollinger Band Percentage")
    axes[0].plot(indicators[["Simple Moving Average"]], label="Simple Moving Average")
    axes[0].plot(indicators[["BBand Upper"]], label="BBand Upper")
    axes[0].plot(indicators[["BBand Lower"]], label="BBand Lower")
    axes[0].legend(loc="lower right")
    axes[1].plot(indicators[["Bollinger Band Percentage"]], label="Bollinger Band Percentage")
    axes[1].legend(loc="lower right")
    axes[0].set(ylabel="Value")
    axes[1].set(xlabel="Date", ylabel="Value")
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=30)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('bbp.png')
    plt.close()
    # Golden Crosses.
    fig, axes = plt.subplots(2)
    fig.suptitle("Golden Crosses")
    axes[0].plot(indicators[["Price"]], label="Price")
    axes[0].plot(indicators[["SMA({})".format(lookback)]], label="SMA({})".format(lookback))
    axes[0].plot(indicators[["SMA({})".format(lookback * 4)]], label="SMA({})".format(lookback * 4))
    axes[0].legend(loc="lower right")
    axes[1].plot(indicators[["Golden Cross Ratio"]], label="Golden Cross Ratio")
    axes[1].legend(loc="lower right")
    axes[0].set(ylabel="Value")
    axes[1].set(xlabel="Date", ylabel="Value")
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=30)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.axhline(y=1, label='Intersections', color='#e8eaed')
    plt.savefig('crosses.png')
    plt.close()
    # Volatility.
    indicators[["Price", "Volatility", "BBand Upper", "BBand Lower"]].plot()
    plt.title("Volatility")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.savefig('volatility.png')
    plt.close()
    # Momentum.
    indicators[["Price", "Simple Moving Average", "Momentum"]].plot()
    plt.title("Momentum")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.savefig('momentum.png')
    plt.close()


if __name__ == "__main__":
    pass
