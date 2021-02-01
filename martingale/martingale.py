""""""
"""Assess a betting strategy.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def author():
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :return: The GT username of the student  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :rtype: str  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """
    return "nriojas3"  # replace tb34 with your Georgia Tech username.


def gtid():
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :return: The GT ID of the student  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :rtype: int  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """
    return 903646605  # replace with your GT ID number


def get_spin_result(win_prob):
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param win_prob: The probability of winning  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type win_prob: float  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :return: The result of the spin.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :rtype: bool  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """
    result = False
    # b = random.random()
    # a = np.random.random()
    # print(a, b)
    # if a <= win_prob:
    #     result = True
    if np.random.random() <= win_prob:
        result = True


    return result


def gambling_simulator(bet_amount=1, win_prob=.474, num_spins=1000, min_winnings=80):
    # initialize my array accounting for 0 being no spin
    arr_win = np.zeros(num_spins + 1)
    arr_win.fill(80)
    arr_win[0] = 0
    spin_count = 0
    episode_winnings = 0
    while episode_winnings < min_winnings and spin_count < num_spins:
        won = False
        while not won and spin_count < num_spins:
            spin_count += 1
            #np.random.seed(gtid())
            won = get_spin_result(win_prob)
            if won:
                episode_winnings += bet_amount
            else:
                episode_winnings -= bet_amount
                bet_amount *= 2
            arr_win[spin_count] = episode_winnings
    return arr_win

def realistic_gambling_simulator(bet_amount=1, win_prob=.474, num_spins=1000, min_winnings=80):
    # initialize my array accounting for 0 being no spin
    arr_win = np.zeros(num_spins + 1)
    arr_win.fill(80)
    arr_win[0] = 0
    spin_count = 0
    episode_winnings = 0
    while episode_winnings < min_winnings and spin_count < num_spins and episode_winnings > -256:
        won = False
        while not won and spin_count < num_spins:
            spin_count += 1
            won = get_spin_result(win_prob)
            available_bet_amount = 256 + episode_winnings
            if bet_amount >= available_bet_amount:
                bet_amount = available_bet_amount
            if won:
                episode_winnings += bet_amount
            else:
                episode_winnings -= bet_amount
                bet_amount *= 2
            arr_win[spin_count] = episode_winnings
    if episode_winnings <= -256 and spin_count < num_spins:
        arr_win[spin_count:] = episode_winnings
    return arr_win

def dataFrameGenerate(simulations, realistic=False):
    # initialize dataframe
    i = np.ones(1001)
    df = pd.DataFrame(i, columns=['i'])
    for j in range(simulations):
        if not realistic:
            arr = gambling_simulator()
        else:
            arr = realistic_gambling_simulator()
        temp_df = pd.DataFrame(arr, columns=[str(j+1)])
        df = df.join(temp_df)
    df = df.drop(['i'], axis=1)
    return df

def get_stats(df, mean = False, median = False):
    if mean:
        m = df.mean(axis=1)
        std = df.std(axis=1)
        df_mean = pd.DataFrame(m, columns=['mean'])
        df_std = pd.DataFrame(std, columns=['+std'])
        df_std_neg = pd.DataFrame(-1 * std, columns=['-std'])
        df2 = df.join(df_mean)
        df2 = df2.join(df_std)
        df2 = df2.join(df_std_neg)
    if median:
        m = df.median(axis=1)
        std = df.std(axis=1)
        df_med = pd.DataFrame(m, columns=['median'])
        df_std = pd.DataFrame(std, columns=['+std'])
        df_std_neg = pd.DataFrame(-1 * std, columns=['-std'])
        df2 = df.join(df_med)
        df2 = df2.join(df_std)
        df2 = df2.join(df_std_neg)

    return df2

def plot_dataframe(df, columns =[], title="Plot", file_name="Figure2.png",all_cols=True):
    x_bounds = [0, 300]
    y_bounds = [-256, 100]
    if all_cols:
        columns = df.columns
    ax = df[columns].plot(title=title)
    ax.set_xlabel("Spin")
    ax.set_ylabel("Winnings")
    ax.set_xlim([x_bounds[0], x_bounds[1]])
    ax.set_ylim([y_bounds[0], y_bounds[1]])
    plt.savefig(file_name)
    #plt.show()

def test_code():
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    Method to test your code  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """
    np.random.seed(gtid())  # do this only once
    # Exp 1 Figure 1 ---------------------------------------------------------------
    df1 = dataFrameGenerate(10)
    cols =[]
    plot_dataframe(df1, cols, "10 Runs of Simulator", "Exp1Fig1.png", all_cols=True)

    # # Exp 1 Figure 2 ---------------------------------------------------------------
    df2 = dataFrameGenerate(1000)
    df2_mean = get_stats(df2, mean=True)
    cols = ['mean', '+std', '-std']
    plot_dataframe(df2_mean, cols, "Mean of 1000 Simulations", "Exp1Fig2.png", all_cols=False)

    # # Exp 1 Figure 3 ---------------------------------------------------------------
    df2_median = get_stats(df2, median=True)
    cols = ['median', '+std', '-std']
    plot_dataframe(df2_median, cols, "Median of 1000 Simulations", "Exp1Fig3.png", all_cols=False)

    # # Exp 1 Figure 4 ---------------------------------------------------------------
    df3 = dataFrameGenerate(1000, realistic=True)
    df3_mean = get_stats(df3, mean=True)
    cols = ['mean', '+std', '-std']
    plot_dataframe(df3_mean, cols, "Mean of 1000 Realistic Simulations", "Exp2Fig4.png", all_cols=False)

    # # Exp 1 Figure 5 ---------------------------------------------------------------
    df3_median = get_stats(df3, median=True)
    cols = ['median', '+std', '-std']
    plot_dataframe(df3_median, cols, "Median of 1000 Realistic Simulations", "Exp2Fig5.png", all_cols=False)


if __name__ == "__main__":
    test_code()


