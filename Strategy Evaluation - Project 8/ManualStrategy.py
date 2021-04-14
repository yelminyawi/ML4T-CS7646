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


def generate_ema_signal(df, ema):
    dfCopy = df.copy()
    dfCopy[ema > df] = 1
    dfCopy[ema < df] = -1
    dfCopy[ema == df] = 0
    return dfCopy


# dates in sample (2008, 1, 1) to (2009, 12, 31)
# dates out of sample (2010, 1, 1) to (2011, 12, 31)
# testing stock symbols AAPL, SINE_FAST_NOISE, ML4T-220, UNH, JPL
def testPolicy(symbol="AAPL", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
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
    max_position = 1000
    # ------------------------------- Create dataframe ---------------------------
    dates = pd.date_range(sd, ed)
    dfStockPrice = get_data([symbol], dates, True, colname='Adj Close').drop(columns=['SPY'])
    dfStockPrice.sort_index()
    # Is this needed?
    dfStockPrice = dfStockPrice.ffill().bfill()
    dfStockPriceNorm = dfStockPrice / dfStockPrice.iloc[0, :]
    dates = dfStockPriceNorm.index
    orders = pd.DataFrame(0, index=dates, columns=['order type', 'position', 'actual shares'])
    # ------------------------------- Get indicators  ---------------------------
    std = ind.calculate_std(dfStockPriceNorm, lookback)
    sma = ind.calculate_sma(dfStockPriceNorm, lookback)
    momentum = ind.calculate_momentum(dfStockPriceNorm, lookback)
    ema = ind.calculate_ema(dfStockPriceNorm, lookback)
    ema_ind = generate_ema_signal(dfStockPriceNorm, ema)
    bbp, top_band, bottom_band = ind.calculate_BB_data(dfStockPriceNorm, lookback, sma, std)
    current_holdings = 0
    for index, row in dfStockPriceNorm.iterrows():
        m = momentum.loc[index][0]
        bb = bbp.loc[index][0]
        e = ema_ind.loc[index][0]
        if e > 0 and bb < 0.2 and m < -0.05 and current_holdings < max_position:
            current_holdings = 1000
            orders.loc[index]['order type'] = 'buy'
            if current_holdings == 0:
                orders.loc[index]['position'] = 1000
                orders.loc[index]['actual shares'] = 1000
            else:
                orders.loc[index]['position'] = 2000
                orders.loc[index]['actual shares'] = 2000
        elif e < 0 and bb > 0.8 and m > 0.05 and current_holdings > -max_position:
            current_holdings = -1000
            orders.loc[index]['order type'] = 'sell'
            if current_holdings == 0:
                orders.loc[index]['position'] = 1000
                orders.loc[index]['actual shares'] = -1000
            else:
                orders.loc[index]['position'] = 2000
                orders.loc[index]['actual shares'] = -2000
    # for index, row in orders.iterrows():
    #     print(row)
    orders_df = orders.copy().drop(columns=['order type', 'position'])
    manual = ms.compute_portvals(orders_df, sd, ed, symbol, sv, commission, impact)
    # generate_plot(manual, manual)
    return manual


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


def get_benchmark(symbol='AAPL', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000,
                  position=1000, commission=9.95, impact=0.005):
    # commission, impact = 0, 0
    dates = pd.date_range(sd, ed)
    dfPrice = get_data([symbol], dates, True, colname='Adj Close').drop(columns=['SPY'])
    dfPrice.sort_index()
    benchmark_orders = dfPrice * 0
    benchmark_orders.iloc[0][0] = position
    benchmark_portfolio = ms.compute_portvals(benchmark_orders, sd, ed, symbol, sv, commission, impact)
    # generate_plot(benchmark_portfolio, benchmark_portfolio)
    return benchmark_portfolio


def cumulative_return(df):
    last = df.iloc[-1][0]
    first = df.iloc[0][0]
    return (last / first) - 1


def daily_returns(df):
    return df.pct_change(1)


def analyze_portfolios(p1, p2):
    c1 = cumulative_return(normalize_df(p1))
    c2 = cumulative_return(normalize_df(p2))
    print('cumulative return')
    print("portfolio 1: ", c1, " portfolio 2:", c2)

    d1 = daily_returns(normalize_df(p1))
    d2 = daily_returns(normalize_df(p2))
    std1 = d1.std()
    std2 = d2.std()
    print('standard deviation')
    print("portfolio 1: ", std1[0], " portfolio 2:", std2[0])
    m1 = d1.mean()
    m2 = d2.mean()
    print('mean')
    print("portfolio 1: ", m1[0], " portfolio 2:", m2[0])


def run():
    get_benchmark()

# testing stock symbols AAPL, SINE_FAST_NOISE, ML4T-220, UNH, JPL
if __name__ == "__main__":
    symbol = "UNH"
    m = testPolicy(symbol=symbol)
    b = get_benchmark(symbol=symbol)
    generate_plot(m, b)
    analyze_portfolios(m, b)
