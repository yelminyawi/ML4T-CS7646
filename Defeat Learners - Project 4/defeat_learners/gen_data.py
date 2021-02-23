""""""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
template for generating data to fool learners (c) 2016 Tucker Balch  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
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
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import math  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
import numpy as np  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
# this function should return a dataset (X and Y) that will work  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
# better for linear regression than decision trees  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
def best_4_lin_reg(seed=1489683273):  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param seed: The random seed for your data generation.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type seed: int  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :return: Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :rtype: numpy.ndarray  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    np.random.seed(seed)
    # data must contain between 10 to 1000 (inclusive) entries
    rows = np.random.randint(10, 1001)
    # number of features must be between 2 and 10 (inclusive)
    cols = np.random.randint(2, 11)
    m = 1
    b = 0
    # generate random x data
    x = np.random.rand(rows, cols)
    # give the linear regression learner a linear function with similar correlation across columns
    y = m * x.mean(1) + b

    return x, y  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
def best_4_dt(seed=1489683273):  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :param seed: The random seed for your data generation.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :type seed: int  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :return: Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :rtype: numpy.ndarray  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    np.random.seed(seed)
    # data must contain between 10 to 1000 (inclusive) entries
    rows = np.random.randint(10, 1001)
    # number of features must be between 2 and 10 (inclusive)
    cols = np.random.randint(2, 11)
    # generate random x data
    x = np.random.rand(rows, cols)
    # defeat linear regression learner with a non linear (exponential) function y = x0^2 + x1^3
    y = x[:, 0]**2 + x[:, 1]**3

    return x, y  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
def author():  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :return: The GT username of the student  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    :rtype: str  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    return "nriojas3"  # Change this to your user ID
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
if __name__ == "__main__":  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    print("they call me Nathan.")
