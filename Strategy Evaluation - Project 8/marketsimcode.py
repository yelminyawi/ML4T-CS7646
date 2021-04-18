""""""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""MC2-P1: Market simulator.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Atlanta, Georgia 30332  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
All Rights Reserved  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Template code for CS 4646/7646  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
works, including solutions to the projects assigned in this course. Students  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
and other users of this template code are advised not to share it with others  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
or to make it available on publicly viewable websites including repositories  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
or edited.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
We do grant permission to share solutions privately with non-students such  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
as potential employers. However, sharing with other current or future  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT honor code violation.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
-----do not edit anything above this line---  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Student Name: Tucker Balch (replace with your name)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT User ID: tb34 (replace with your User ID)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT ID: 900897987 (replace with your GT ID)  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import datetime as dt  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import os  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import numpy as np  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import pandas as pd
import copy
from util import get_data, plot_data  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			

'''
format for portfolio dictionary
stocks are associated with number of stocks, cash reflects total available cash
{'cash':100000, 'IBM':1000, 'AAPL':20}

OTHER CONSIDERATIONS:

1. What if your cash is empty and you try to buy?
2. How do I implement shorting
3. Specifically, what if first order is to sell?
4. What if I sell shares and it exceed number of shares I have
    i.e. mixed selling and shorting
5. Is my portfolio values dataframe returning dates inclusive?
'''


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


def author():
    return 'nriojas3'

