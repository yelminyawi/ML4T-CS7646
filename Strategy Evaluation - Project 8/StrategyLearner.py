""""""
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			

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

Student Name: Nathan Riojas 	  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT User ID: nriojas3  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
GT ID: 903646605 		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""

import datetime as dt
import pandas as pd
import util as ut
import indicators as ind
import RTLearner as rt
import BagLearner as bl


class StrategyLearner(object):
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        # tune these parameters
        self.leaf_size = 8
        self.look_back = 5
        self.testing_window = self.look_back
        self.return_min = .03
        self.bags = 20
        # create the learner
        self.learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": self.leaf_size}, bags=self.bags)

    def author(self):
        return 'nriojas3'

    # creates features dataframe for x train and test
    def generate_features_df(self, price_df):
        # get indicators and rename columns to indicator name
        std = ind.calculate_std(price_df, self.look_back)
        sma = ind.calculate_sma(price_df, self.look_back)
        pp_sma = ind.calculate_price_per_sma(price_df, sma)
        pp_sma.columns = ['SMA']
        momentum = ind.calculate_momentum(price_df, self.look_back)
        momentum.columns = ['Momentum']
        bbp, top_band, bottom_band = ind.calculate_BB_data(price_df, self.look_back, sma, std)
        bbp.columns = ['BBP']
        # concatenate to create a single x dataframe filled with features
        features = pd.concat((pp_sma, momentum, bbp), axis=1)
        return features

    # creates a dataframe of stock prices over a specified start and end date
    def generate_prices_df(self, symbol, sd, ed):
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        return prices

    def add_evidence(self, symbol="IBM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), sv=10000):
        # ------------------ generate dataframe of prices for stock -----------------------------
        prices = self.generate_prices_df(symbol, sd, ed)

        # ------------------ generate a train x dataframe using the indicators as features -------
        train_x = self.generate_features_df(prices)
        train_x = train_x[:-self.testing_window]

        # ---------- create the train y dataframe based on the future window specified -----------
        train_y = prices.copy() * 0
        train_y = train_y[:-self.testing_window]

        for i in range(len(prices) - self.testing_window):
            # cumulative return for the window
            return_during_window = (prices.iloc[i + self.testing_window, :] / prices.iloc[i, :]) - 1
            # Long
            if return_during_window[0] > (self.return_min + self.impact):
                train_y.iloc[i, :] = 1
            # Short
            elif return_during_window[0] < (-self.return_min - self.impact):
                train_y.iloc[i, :] = -1
            # Cash
            else:
                train_y.iloc[i, :] = 0
        # ----------------------------- train the learner ----------------------------------------
        self.learner.add_evidence(train_x.values, train_y.values)

    def testPolicy(self, symbol="IBM", sd=dt.datetime(2009, 1, 1), ed=dt.datetime(2010, 1, 1), sv=10000):
        # --------------------- generate dataframe of prices for stock ---------------------------
        prices = self.generate_prices_df(symbol, sd, ed)

        # ------------- generate a train x dataframe using the indicators as features ------------
        test_x = self.generate_features_df(prices)
        test_x = test_x[:]

        # ----------------------- generate test y data by querying learner -----------------------
        test_y = self.learner.query(test_x.values)

        # ----------------------------- create trades dataframe ----------------------------------
        trades = prices.copy() * 0
        current_holdings = 0
        for i in range(len(trades)):
            # Long signal
            if test_y[i] > 0:
                if current_holdings == -1:
                    trades.iloc[i, :] = 2000
                elif current_holdings == 0:
                    trades.iloc[i, :] = 1000
                current_holdings = 1
            # Short signal
            elif test_y[i] < 0:
                if current_holdings == 1:
                    trades.iloc[i, :] = -2000
                elif current_holdings == 0:
                    trades.iloc[i, :] = -1000
                current_holdings = -1
            # Cash signal
            elif test_y[i] == 0:
                if current_holdings == -1:
                    trades.iloc[i, :] = 1000
                elif current_holdings == 1:
                    trades.iloc[i, :] = -1000
                current_holdings = 0

        return trades


if __name__ == "__main__":
    print("One does not simply think up a strategy")
