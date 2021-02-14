""""""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
Test a learner.  (c) 2015 Tucker Balch  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
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
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import math  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import LinRegLearner as lrl
import DTLearner as dtl
import BagLearner as bgl
import RTLearner as rtl

def process_data(data):

    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows
    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]
    return train_x, train_y, test_x, test_y

def experiment_1(data, max_leaves):
    # separate out training and testing data
    train_x, train_y, test_x, test_y = process_data(data)
    # two rows detailing the rsme at leaves from 1 to max leaves
    rsme_arr = np.ones((2, max_leaves))
    # create array of learners with various leaf specifications
    for i in range(max_leaves):
        # create a learner and train it
        learner = dtl.DTLearner(leaf_size=i+1, verbose=True)  # create a LinRegLearner
        learner.add_evidence(train_x, train_y)  # train it
        # evaluate in sample
        pred_y_in = learner.query(train_x)  # get the predictions
        rmse_in = math.sqrt(((train_y - pred_y_in) ** 2).sum() / train_y.shape[0])
        rsme_arr[0][i] = rmse_in
        # evaluate out of sample
        pred_y_out = learner.query(test_x)  # get the predictions
        rmse_out = math.sqrt(((test_y - pred_y_out) ** 2).sum() / test_y.shape[0])
        rsme_arr[1][i] = rmse_out
    # convert array to a dataframe
    df = pd.DataFrame(rsme_arr.T)
    # plot the dataframe and add various labels
    fig, ax = plt.subplots()
    ax.plot(df[0], 'C10')
    ax.plot(df[1], 'C16')
    ax.grid()
    plt.xlabel("Number of Leaves")
    plt.ylabel("RMSE")
    plt.legend(["In Sample", "Out of Sample"])
    plt.title("DT: RMSE vs Number Of Leaves")
    # add watermark
    fig.text(0.5, 0.5, 'Property of Nathan Riojas',
             fontsize=30, color='gray',
             ha='center', va='center', rotation='30', alpha=0.36)
    plt.savefig("Exp1.png")


def experiment_2(data, max_leaves, bags):
    # separate out training and testing data
    train_x, train_y, test_x, test_y = process_data(data)
    # two rows detailing the rsme at leaves from 1 to max leaves
    rsme_arr = np.ones((2, max_leaves))
    # create array of learners with various leaf specifications
    for i in range(max_leaves):
        # create a learner and train it
        learner = bgl.BagLearner(
                    learner=dtl.DTLearner,
                    kwargs={"leaf_size": i+1},
                    bags=bags,
                    boost=False,
                    verbose=False,
                )
        learner.add_evidence(train_x, train_y)  # train it
        # evaluate in sample
        pred_y_in = learner.query(train_x)  # get the predictions
        rmse_in = math.sqrt(((train_y - pred_y_in) ** 2).sum() / train_y.shape[0])
        rsme_arr[0][i] = rmse_in
        # evaluate out of sample
        pred_y_out = learner.query(test_x)  # get the predictions
        rmse_out = math.sqrt(((test_y - pred_y_out) ** 2).sum() / test_y.shape[0])
        rsme_arr[1][i] = rmse_out
    # convert array to a dataframe
    df = pd.DataFrame(rsme_arr.T)
    # plot the dataframe and add various labels
    fig, ax = plt.subplots()
    ax.plot(df[0], 'C39')
    ax.plot(df[1], 'C1')
    ax.grid()
    plt.xlabel("Number of Leaves")
    plt.ylabel("RMSE")
    plt.legend(["In Sample", "Out of Sample"])
    plt.title("Bag Learner: RMSE vs Number Of Leaves (20 Bags)")
    # add watermark
    fig.text(0.5, 0.5, 'Property of Nathan Riojas',
             fontsize=30, color='gray',
             ha='center', va='center', rotation='30', alpha=0.36)
    plt.savefig("Exp2.png")


# here we use a different accuracy metric (Mean Absolute Error) to compare DTs and Rts
# in sample and out of sample are plotted together for both learners
def experiment_3_together(data, max_leaves):
    # separate out training and testing data
    train_x, train_y, test_x, test_y = process_data(data)
    # two rows detailing the rsme at leaves from 1 to max leaves
    mae_arr = np.ones((4, max_leaves))
    # create array of learners with various leaf specifications
    for i in range(max_leaves):
        # create a Decision Tree learner and train it
        learner1 = dtl.DTLearner(leaf_size=i+1, verbose=True)
        learner1.add_evidence(train_x, train_y)  # train it
        # create a Random Tree learner and train it
        learner2 = rtl.RTLearner(leaf_size=i + 1, verbose=True)
        learner2.add_evidence(train_x, train_y)  # train it
        # evaluate DT in sample
        pred_y_in1 = learner1.query(train_x)  # get the predictions
        mae_in1 = abs(train_y - pred_y_in1).sum() / train_y.shape[0]
        mae_arr[0][i] = mae_in1
        # evaluate RT in sample
        pred_y_in2 = learner2.query(train_x)  # get the predictions
        mae_in2 = abs(train_y - pred_y_in2).sum() / train_y.shape[0]
        mae_arr[1][i] = mae_in2
        # evaluate DT out of sample
        pred_y_out1 = learner1.query(test_x)  # get the predictions
        mae_out1 = abs(test_y - pred_y_out1).sum() / test_y.shape[0]
        mae_arr[2][i] = mae_out1
        # evaluate RT out of sample
        pred_y_out2 = learner2.query(test_x)  # get the predictions
        mae_out2 = abs(test_y - pred_y_out2).sum() / test_y.shape[0]
        mae_arr[3][i] = mae_out2
    # convert array to a dataframe
    df = pd.DataFrame(mae_arr.T)
    # plot the dataframe and add various labels
    fig, ax = plt.subplots()
    ax.plot(df[0], 'C3')
    ax.plot(df[1], 'C4')
    ax.plot(df[2], 'C1')
    ax.plot(df[3], 'C9')
    ax.grid()
    plt.xlabel("Number of Leaves")
    plt.ylabel("MAE")
    plt.legend(["DT In Sample", "RT In Sample", "DT Out of Sample", "RT Out of Sample"])
    plt.title("DT & RT: MAE vs Number Of Leaves")
    # add watermark
    fig.text(0.5, 0.5, 'Property of Nathan Riojas',
             fontsize=30, color='gray',
             ha='center', va='center', rotation='30', alpha=0.36)
    plt.savefig("Exp3-Together.png")

# here we use a different accuracy metric (Mean Absolute Error) to compare DTs and Rts
# in sample and out of sample are plotted together for both learners
def experiment_3_separate(data, max_leaves):
    # separate out training and testing data
    train_x, train_y, test_x, test_y = process_data(data)
    # two rows detailing the rsme at leaves from 1 to max leaves
    mae_arr_in = np.ones((2, max_leaves))
    mae_arr_out = np.ones((2, max_leaves))
    # create array of learners with various leaf specifications
    for i in range(max_leaves):
        # create a Decision Tree learner and train it
        learner1 = dtl.DTLearner(leaf_size=i + 1, verbose=True)
        learner1.add_evidence(train_x, train_y)  # train it
        # create a Random Tree learner and train it
        learner2 = rtl.RTLearner(leaf_size=i + 1, verbose=True)
        learner2.add_evidence(train_x, train_y)  # train it
        # evaluate DT in sample
        pred_y_in1 = learner1.query(train_x)  # get the predictions
        mae_in1 = abs(train_y - pred_y_in1).sum() / train_y.shape[0]
        mae_arr_in[0][i] = mae_in1
        # evaluate RT in sample
        pred_y_in2 = learner2.query(train_x)  # get the predictions
        mae_in2 = abs(train_y - pred_y_in2).sum() / train_y.shape[0]
        mae_arr_in[1][i] = mae_in2
        # evaluate DT out of sample
        pred_y_out1 = learner1.query(test_x)  # get the predictions
        mae_out1 = abs(test_y - pred_y_out1).sum() / test_y.shape[0]
        mae_arr_out[0][i] = mae_out1
        # evaluate RT out of sample
        pred_y_out2 = learner2.query(test_x)  # get the predictions
        mae_out2 = abs(test_y - pred_y_out2).sum() / test_y.shape[0]
        mae_arr_out[1][i] = mae_out2
    # convert array to a dataframe
    df1 = pd.DataFrame(mae_arr_in.T)
    # plot the dataframe and add various labels
    fig1, ax1 = plt.subplots()
    ax1.plot(df1[0], 'C30')
    ax1.plot(df1[1], 'C28')
    ax1.grid()
    plt.xlabel("Number of Leaves")
    plt.ylabel("MAE")
    plt.legend(["DT In Sample", "RT In Sample"])
    plt.title("MAE Decision Tree vs Random Tree In Sample")
    # add watermark
    fig1.text(0.5, 0.5, 'Property of Nathan Riojas',
              fontsize=30, color='gray',
              ha='center', va='center', rotation='30', alpha=0.36)

    df2 = pd.DataFrame(mae_arr_out.T)
    # plot the dataframe and add various labels
    fig2, ax2 = plt.subplots()
    ax2.plot(df2[0], 'C4')
    ax2.plot(df2[1], 'C21')
    ax2.grid()
    plt.xlabel("Number of Leaves")
    plt.ylabel("MAE")
    plt.legend(["DT Out of Sample", "RT Out of Sample"])
    plt.title("MAE Decision Tree vs Random Tree Out of Sample")
    # add watermark
    fig2.text(0.5, 0.5, 'Property of Nathan Riojas',
              fontsize=30, color='gray',
              ha='center', va='center', rotation='30', alpha=0.36)
    fig1.savefig("Exp3-Separate-InSample.png")
    fig2.savefig("Exp3-Separate-OutSample.png")


# here we are comparing the amount of time to train DT and RT learners
def experiment_3_time(data, max_leaves):
    # separate out training and testing data
    train_x, train_y, test_x, test_y = process_data(data)
    # two rows detailing the time DT's and RT's take to add evidence i.e. build their tree
    time_arr = np.ones((2, max_leaves))
    # create array of learners with various leaf specifications
    for i in range(max_leaves):
        # create a Decision Tree learner and time how long to train it
        learner1 = dtl.DTLearner(leaf_size=i+1, verbose=True)
        start1 = time.time()
        learner1.add_evidence(train_x, train_y)  # train it
        end1 = time.time()
        # create a Random Tree learner and time how long to train it
        learner2 = rtl.RTLearner(leaf_size=i + 1, verbose=True)
        start2 = time.time()
        learner2.add_evidence(train_x, train_y)  # train it
        end2 = time.time()
        time_arr[0][i] = end1 - start1
        time_arr[1][i] = end2 - start2

    # convert array to a dataframe
    df = pd.DataFrame(time_arr.T)
    # plot the dataframe and add various labels
    fig, ax = plt.subplots()
    ax.plot(df[0], 'C3')
    ax.plot(df[1], 'C10')
    ax.grid()
    plt.xlabel("Number of Leaves")
    plt.ylabel("Training Time (seconds)")
    plt.legend(["DT In Sample", "RT In Sample"])
    plt.title("Decision Tree vs Random Tree: Time To Train")
    # add watermark
    fig.text(0.5, 0.5, 'Property of Nathan Riojas',
             fontsize=30, color='gray',
             ha='center', va='center', rotation='30', alpha=0.36)
    plt.savefig("Exp3-Time.png")
    #plt.show()


if __name__ == "__main__":  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    if len(sys.argv) != 2:  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        print("Usage: python testlearner.py <filename>")  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        sys.exit(1)

    inf = open(sys.argv[1])
    # process instanbul.csv's first row strings and first column dates
    if sys.argv[1] == "Data/Istanbul.csv":
        data = np.array(
            [list(map(float, s.strip().split(",")[1:])) for s in inf.readlines()[1:]]
        )
    else:
        data = np.array(
            [list(map(float, s.strip().split(","))) for s in inf.readlines()]
        )
    experiment_1(data, 100)
    experiment_2(data, 100, 20)
    experiment_3_together(data, 100)
    experiment_3_separate(data, 100)
    experiment_3_time(data, 100)

