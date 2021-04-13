import marketsim as ms
import indicators as ind
from util import get_data, plot_data

import datetime as dt
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt


def author():
    return 'nriojas3'


def testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    # dates = pd.date_range(start_date, end_date)
    # df = get_data([stock], dates, True, colname='Adj Close').drop(columns=['SPY'])
    # df_norm = df / df.iloc[0, :]
    #
    # # generate dataframes from technical indicators
    # std = calculate_std(df_norm, lookback)
    # sma = calculate_sma(df_norm, lookback)
    # price_per_sma = calculate_price_per_sma(df_norm, sma)
    # momentum = calculate_momentum(df_norm, lookback)
    # ema = calculate_ema(df_norm, lookback)
    # bbp, top_band, bottom_band = calculate_BB_data(df_norm, lookback, sma, std)
    # ------------------------------- Place constants here ---------------------------
    commission = 9.95
    impact = 0.005
    lookback = 10
    # ------------------------------- Create dataframe ---------------------------
    dates = pd.date_range(sd, ed)
    dfStockPrice = get_data([symbol], dates, True, colname='Adj Close').drop(columns=['SPY'])
    dfStockPrice.sort_index()
    # Is this needed?
    # dfStockPrice = dfStockPrice.ffill().bfill()
    dfStockPriceNorm = dfStockPrice / dfStockPrice.iloc[0, :]
    dates = dfStockPriceNorm.index
    # ------------------------------- Get indicators  ---------------------------
    std = ind.calculate_std(dfStockPriceNorm, lookback)
    sma = ind.calculate_sma(dfStockPriceNorm, lookback)
    momentum = ind.calculate_momentum(dfStockPriceNorm, lookback)
    ema = ind.calculate_ema(dfStockPriceNorm, lookback)
    bbp, top_band, bottom_band = ind.calculate_BB_data(dfStockPriceNorm, lookback, sma, std)
    for day in range(len(dates)):
        pass


    return


def normalize_df(df):
    return df / df.iloc[0, :]


def generate_plot(optimal, benchmark):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(normalize_df(optimal), color='red')
    ax.plot(normalize_df(benchmark), color='green')
    ax.tick_params(axis='x', rotation=20)
    ax.grid()
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend(['Optimal', "Benchmark"])
    plt.title("JPM Theoretically Optimal Trading Strategy vs Benchmark Strategy")
    fig.text(0.5, 0.5, 'Property of Nathan Riojas',
             fontsize=30, color='gray',
             ha='center', va='center', rotation='30', alpha=0.36)
    plt.savefig("optimal.png")


def get_benchmark(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000,
                  position=1000, commission=9.95, impact=0.005):
    # commission, impact = 0, 0
    dates = pd.date_range(sd, ed)
    dfPrice = get_data([symbol], dates, True, colname='Adj Close').drop(columns=['SPY'])
    dfPrice.sort_index()
    dfTrades = dfPrice.copy()
    dfTrades = ((dfTrades.shift(-1) - dfTrades) / abs(dfTrades.shift(-1) - dfTrades) * 1000).fillna(0)
    dfTradesPast = (dfTrades.shift(1)).fillna(0)
    optimized_orders = dfTrades - dfTradesPast
    benchmark_orders = optimized_orders * 0
    benchmark_orders.iloc[0][0] = position
    benchmark_portfolio = ms.compute_portvals(benchmark_orders, sd, ed, symbol, sv, commission, impact)
    generate_plot(benchmark_portfolio, benchmark_portfolio)

def run():
    get_benchmark()
if __name__ == "__main__":
    get_benchmark()



