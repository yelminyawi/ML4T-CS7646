import datetime as dt
import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from util import get_data, plot_data


def author():
	return 'nriojas3'


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
	print(c1, c2)

	d1 = daily_returns(normalize_df(p1))
	d2 = daily_returns(normalize_df(p2))
	std1 = d1.std()
	std2 = d2.std()
	print('standard deviation')
	print(std1, std2)
	m1 = d1.mean()
	m2 = d2.mean()
	print('mean')
	print(m1, m2)



def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
	commission, impact = 0, 0
	dates = pd.date_range(sd, ed)
	dfPrice = get_data([symbol], dates, True, colname='Adj Close').drop(columns=['SPY'])
	dfPrice.sort_index()
	dfTrades = dfPrice.copy()
	dfTrades = ((dfTrades.shift(-1) - dfTrades) / abs(dfTrades.shift(-1) - dfTrades) * 1000).fillna(0)
	dfTradesPast = (dfTrades.shift(1)).fillna(0)
	optimized_orders = dfTrades - dfTradesPast
	benchmark_orders = optimized_orders * 0
	benchmark_orders.iloc[0][0] = 1000
	optimal_portfolio = compute_portvals(optimized_orders, sd, ed, symbol, sv, commission, impact)
	benchmark_portfolio = compute_portvals(benchmark_orders, sd, ed, symbol, sv, commission, impact)
	generate_plot(optimal_portfolio, benchmark_portfolio)

	#analyze_portfolios(optimal_portfolio, benchmark_portfolio)
	return optimized_orders

#  Here is where I am pasting my marketsim.py code

def get_stock_value(stock, date) -> float:
    temp_dates_df = pd.date_range(date, date)
    adj_close = get_data([stock], temp_dates_df, False)
    return adj_close.loc[date][0]


def get_account_value(account_dict, date) -> float:
    value = 0
    for stocks in account_dict:
        if stocks == 'cash':
            value += account_dict[stocks]
        else:
            stock_price = get_stock_value(stocks, date)
            value += account_dict[stocks] * stock_price
    return value


def buy_stock(stock, num_shares, account_dict, date, impact, commission) -> dict:
    stock_price = get_stock_value(stock, date)
    cost = stock_price * num_shares * (1 + impact)
    # Implement commission stuff here!!!!!
    account_dict['cash'] -= cost + commission
    if stock in account_dict:
        account_dict[stock] += num_shares
    else:
        account_dict[stock] = num_shares
    return account_dict


def sell_stock(stock, num_shares, account_dict, date, impact, commission) -> dict:
    stock_price = get_stock_value(stock, date)
    cost = stock_price * num_shares * (1 - impact)
    # Implement commission stuff here!!!!!
    account_dict['cash'] += cost - commission
    if stock in account_dict:
        account_dict[stock] -= num_shares
    else:
        account_dict[stock] = -1 * num_shares
    return account_dict


def is_trading_day(date) -> bool:
    return not pd.isna(get_stock_value('$SPX', date))


def compute_portvals(orders_df, start_date, end_date, stock="JPM", start_val=1000000, commission=9.95, impact=0.005):

    # parse orders csv into data frame ----------------------------------------------------------------------
    #orders_df = pd.read_csv(orders_file, parse_dates=True, na_values=['nan'])
    #orders_df = orders_df.sort_values(by='Date')
    #already have this



    # generate the dataframe to output daily portfolio values to --------------------------------------------
    #orders_dates = orders_df['Date']
    #start_date, end_date = orders_dates[0], orders_dates[len(orders_dates) - 1]
    dates = pd.date_range(start_date, end_date)
    portfolio_values = pd.DataFrame(index=dates)
    empty_arr = np.zeros(len(portfolio_values))
    zeros = pd.DataFrame(empty_arr)
    portfolio_values = portfolio_values.join(zeros)

    # I've already taken care of this
    # # use SPY data to remove non trading days from portfolio values dataframe using Inner Join --------------
    dfSPY = get_data(['$SPX'], dates, False, colname='Adj Close')
    dfSPY = dfSPY.dropna()
    # portfolio_values = portfolio_values.join(zeros)
    portfolio_values = portfolio_values.join(dfSPY, how='inner')
    portfolio_values = portfolio_values.drop(columns=['$SPX'])

    # create a dictionary to iteratively update and maintain portfolio value ---------------------------------
    account = {'cash': start_val}

    # create dictionary of dictionaries for portfolio allocation snapshots once orders are placed ------------
    portfolio_stocks = {start_date: account}
    current_position = 0
    # iterate orders dataframe, populate portfolio stocks dictionary with portfolio at each date -------------
    for index, row in orders_df.iterrows():
        #date, stock, order_type, num_shares = row[0], row[1], row[2], row[3]
        date = index
        order = row[0]
        num_shares = abs(order)
        buy = order > 0
        sell = order < 0
        # ignore trading days on the chance that SPY didn't filter these --------------------------------------
        # if not is_trading_day(date):
        #     continue
        # process order type, update account dictionary, add the account dictionary to portfolio stocks -------
        if buy:
            account = buy_stock(stock, num_shares, account, date, impact, commission)
            #d = pd.Timestamp(date)
            value = get_account_value(account, date)
            portfolio_values.loc[date][0] = value
            portfolio_stocks[date] = copy.deepcopy(account)
        elif sell:
            account = sell_stock(stock, num_shares, account, date, impact, commission)
            #d = pd.Timestamp(date)
            value = get_account_value(account, date)
            portfolio_values.loc[date][0] = value
            portfolio_stocks[date] = copy.deepcopy(account)

    # initialize a dictionary that will take on the values within the portfolio stocks dictionary -------------
    current_portfolio = {'cash': start_val}
    for dates in portfolio_values.index:
        val = portfolio_values.loc[dates][0]
        # na values should use the current portfolio allocations to calculate current value -------------------
        if pd.isna(val) and is_trading_day(dates):
            v = get_account_value(current_portfolio, dates)
            portfolio_values.loc[dates][0] = v
        # keep the current portfolio allocations updated to facilitate na updates -----------------------------
        else:
            current_portfolio = portfolio_stocks[dates]

    return portfolio_values

if __name__ == "__main__":

	df_trades = testPolicy()

