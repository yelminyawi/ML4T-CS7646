import pandas as pd
import numpy as np


class DTLearner:
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        # this is based on the tabular view layout of node, factor, splitval, left, right
        self.tree = np.empty((0, 4))
        pass

    # data is of the form [label1, label2,.. labelN, y]
    def build_tree(self, d):
        data = np.atleast_2d(d)
        if data.shape[0] == 1:
            return [np.NAN, data[-1], np.NAN, np.NAN]
        if self.same_y(data):
            return [np.NAN, data[:, -1][0], np.NAN, np.NAN]
        else:
            i = self.best_col_corr(data)  # get the index of the highest correlated column
            SplitVal = data[:, i].median()
            lefttree = self.build_tree(data[data[:, i] <= SplitVal])
            righttree = self.build_tree(data[data[:, i] > SplitVal])
            root = np.array([i, SplitVal, lefttree.shape[0] + 1])
            return np.vstack((root, lefttree, righttree))

    def same_y(self, data):
        y = data[:, -1]
        return np.max(y) == np.min(y)

    def best_col_corr(self, data):
        best_correlation = -10000
        best_idx = -1
        transpose = data.T
        for col in range(transpose.shape[0] - 1):
            correlation = abs(np.corrcoef(transpose[col], y=data[:, -1]))
            if correlation > best_correlation:
                best_correlation = correlation
                best_idx = col
        return best_idx

    def train(self, x, y):
        # converting this to the form used in lecture to build the tree
        data = np.append(x, y, axis=1)
        self.tree = self.build_tree(data)

    def query(self, x):
        pass

