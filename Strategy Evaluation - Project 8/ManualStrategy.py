import marketsimcode as ms
import indicators as ind
from util import get_data
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

# to optimize everything at once
def author():
    return 'nriojas3'

def testPolicy(symbol="AAPL", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    # ------------------------------- Place constants here ---------------------------
    commission = 9.95  # specified on project page
    impact = 0.005  # specified on project page
    max_position = 1000
    look_back = 14
    s_opt = .95
    bb_opt = .35  # correlates to top BB greater than 0.85 and bottom less than 0.15
    m_opt = .15
    # ------------------------------- Create dataframe ---------------------------
    dates = pd.date_range(sd, ed)
    dfStockPrice = get_data([symbol], dates, True, colname='Adj Close').drop(columns=['SPY'])
    dfStockPrice.sort_index()
    dfStockPrice = dfStockPrice.ffill().bfill()
    dfStockPriceNorm = dfStockPrice / dfStockPrice.iloc[0, :]
    dates = dfStockPriceNorm.index
    orders = pd.DataFrame(0, index=dates, columns=['order type', 'position', symbol])
    # ------------------------------- Get indicators  ---------------------------
    std = ind.calculate_std(dfStockPriceNorm, look_back)
    sma = ind.calculate_sma(dfStockPriceNorm, look_back)
    pp_sma = ind.calculate_price_per_sma(dfStockPriceNorm, sma)
    momentum = ind.calculate_momentum(dfStockPriceNorm, look_back)
    bbp, top_band, bottom_band = ind.calculate_BB_data(dfStockPriceNorm, look_back, sma, std)
    current_holdings = 0
    i = 1
    # ------------------------------- Implement strategy -------------------------
    for index, row in dfStockPriceNorm.iterrows():
        # get indicators at specific day
        m = momentum.loc[index][0]
        bb = bbp.loc[index][0]
        s = pp_sma.loc[index][0]
        # skip first iteration due to lack of data
        if i > look_back:
            # ------------------------------- Buy signal  ---------------------------
            if s < s_opt and bb < 0.5-bb_opt and m < -m_opt and current_holdings < max_position:
                orders.loc[index]['order type'] = 'buy'
                # actual position
                if current_holdings == 0:
                    orders.loc[index]['position'] = 1000
                    orders.loc[index][symbol] = 1000
                # order
                else:
                    orders.loc[index]['position'] = 2000
                    orders.loc[index][symbol] = 2000
                current_holdings = 1000
            # ------------------------------- Sell signal  ---------------------------
            elif s > s_opt and bb > 0.5 + bb_opt and m < m_opt and current_holdings > -max_position:
                orders.loc[index]['order type'] = 'sell'
                if current_holdings == 0:
                    orders.loc[index]['position'] = 1000
                    orders.loc[index][symbol] = -1000
                else:
                    orders.loc[index]['position'] = 2000
                    orders.loc[index][symbol] = -2000
                current_holdings = -1000
        i += 1
    orders_df = orders.copy().drop(columns=['order type', 'position'])
    return orders_df


def normalize_df(df):
    return df / df.iloc[0, :]


def generate_plot(manual, benchmark, trades, symbol, in_sample=True):
    alpha = 0.7
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(normalize_df(manual), color='red')
    ax.plot(normalize_df(benchmark), color='green')
    ax.tick_params(axis='x', rotation=20)
    ax.grid()
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend(['Manual', "Benchmark"])
    plt.title("JPM Theoretically Optimal Trading Strategy vs Benchmark Strategy")
    fig.text(0.5, 0.5, 'Property of Nathan Riojas',
             fontsize=30, color='gray',
             ha='center', va='center', rotation='30', alpha=0.36)
    for index, row in trades.iterrows():
        if trades.loc[index][symbol] > 0:
            ax.axvline(x=index, color='b', linestyle='-', alpha=alpha)
        elif trades.loc[index][symbol] < 0:
            ax.axvline(x=index, color='k', linestyle='-', alpha=alpha)
    plt.legend(['Manual', "Benchmark", "Short", "Long"])
    if in_sample:
        plt.title("JPM In Sample Manual Strategy vs Benchmark Strategy")
        plot_name = "JPMinSample.png"
    else:
        plt.title("JPM Out of Sample Manual Strategy vs Benchmark Strategy")
        plot_name = "JPMoutSample.png"
    plt.savefig(plot_name)


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
    print("portfolio 1: ", c1, " portfolio 2:", c2, " difference ", c1 - c2)

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
    return c1 - c2


# Using this to generate plots for manual strategy vs benchmark
def generate_manual_plots(s='JPM', commission=9.95, impact=0.005, sv=100000):
    samples = [True, False]
    # generate plots and data for in sample and out of sample
    for in_sample in samples:
        if in_sample:
            sd = dt.datetime(2008, 1, 1)
            ed = dt.datetime(2009, 12, 31)
        else:
            sd = dt.datetime(2010, 1, 1)
            ed = dt.datetime(2011, 12, 31)
        manual = testPolicy(symbol=s, sd=sd, ed=ed)
        m = ms.compute_portvals(manual, sd, ed, s, sv, commission, impact)
        m.columns = [s]
        b = get_benchmark(symbol=s, sd=sd, ed=ed)
        generate_plot(m, b, manual, s, in_sample)


# Using this to generate plots for manual strategy vs benchmark
if __name__ == "__main__":
    commission = 9.95  # specified on project page
    impact = 0.005  # specified on project page
    sv = 100000
    s = "JPM"
    samples = [True, False]
    # generate plots and data for in sample and out of sample
    for in_sample in samples:
        # manual.columns = [symbol]
        # print(manual)
        if in_sample:
            sd = dt.datetime(2008, 1, 1)
            ed = dt.datetime(2009, 12, 31)
        else:
            sd = dt.datetime(2010, 1, 1)
            ed = dt.datetime(2011, 12, 31)
        manual = testPolicy(symbol=s, sd=sd, ed=ed)
        m = ms.compute_portvals(manual, sd, ed, s, sv, commission, impact)
        m.columns = [s]
        b = get_benchmark(symbol=s, sd=sd, ed=ed)
        generate_plot(m, b, manual, s, in_sample)
        c = analyze_portfolios(m, b)



