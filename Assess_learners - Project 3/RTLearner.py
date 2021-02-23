import pandas as pd
import numpy as np


class RTLearner:
    def __init__(self, leaf_size=1, verbose=True):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.wow = ''
        # this is based on the tabular view layout of [node, factor, splitval, left, right]
        self.tree = np.empty((0, 4), dtype=float)

    def author(self) -> str:
        return 'nriojas3'

    # build tree model, d form is label1, label2,...y
    # helper functions - same_y, no_split
    def build_tree(self, d) -> np.ndarray:
        # standardize data so that shape[0] will always work
        data = np.atleast_2d(d)
        # if data.shape[0] == 1:
        #     leaf = np.array([[np.NAN, data[0][-1], np.NAN, np.NAN]])
        #     return leaf
        if self.same_y(data):
            leaf = np.array([[np.NAN, data[0][-1], np.NAN, np.NAN]])
            return leaf
        # implement pruning here using leaf size attribute
        if data.shape[0] <= self.leaf_size:
            y = np.mean(data[:, -1])
            leaf = np.array([[np.NAN, y, np.NAN, np.NAN]])
            return leaf
        else:
            # find idx and split val for best correlated column
            i, SplitVal = self.find_best_split(data)
            # create a leaf when no split is possible
            if self.no_split(data, i, SplitVal):
                y = np.mean(data[:, -1])
                leaf = np.array([[np.NAN, y, np.NAN, np.NAN]])
                return leaf
            # recursively build tree
            lefttree = self.build_tree(data[data[:, i] <= SplitVal])
            righttree = self.build_tree(data[data[:, i] > SplitVal])
            # each node is technically a tree so return the stacked arrays
            root = np.array([[i, SplitVal, 1, lefttree.shape[0] + 1]])
            return np.vstack((root, lefttree, righttree))

    # randomly select a column to make a split on, viability of the split check is not done
    # like it was for DT learner
    # helper functions - rand_col, no_split
    def find_best_split(self, d) -> tuple:
        # make a deep copy of the input array
        data = np.empty_like(d)
        data[:] = d
        # find the initial best index
        i = np.random.randint(data.shape[1] - 2)
        SplitVal = np.median(data[:, i])
        return i, SplitVal

    def rand_col(self, data, omit_cols=[]) ->int:
        # the last column is not an option because it is my y column
        max_num = data.shape[1] - 2
        min_num = 0
        i = np.random.randint(low=min_num, high=max_num)
        while i in omit_cols:
            i = np.random.randint(low=min_num, high=max_num)
        return i

    # check if the median of data is able to split data adequately
    def no_split(self, data, i, SplitVal) -> bool:
        ltd = data[data[:, i] <= SplitVal].shape[0]
        rtd = data[data[:, i] > SplitVal].shape[0]
        return ltd == 0 or rtd == 0

    # data is all the same if max value is the same as the min value
    def same_y(self, data) -> bool:
        y = data[:, -1]
        return np.max(y) == np.min(y)

    # check if the branch is a leaf
    def is_leaf(self, arr) -> bool:
        return np.isnan(arr[0])

    # find value of x at the feature column in question
    def query_value(self, branch_arr, x) -> float:
        factor = int(branch_arr[0])
        return x[factor]

    # check if x belongs in the right tree
    # helper functions - query_value
    def next_is_right(self, branch_arr, x) -> bool:
        val = self.query_value(branch_arr, x)
        return val > branch_arr[1]

    # find the relative index of the right tree
    def relative_right_idx(self, branch_arr) -> int:
        return int(branch_arr[-1])

    # predicts output y for the instance of x labels
    # helper functions - is_leaf, next_is_right, relative_right_idx
    def search_tree(self, x_row_arr) -> float:
        # begin search at the root node
        max_row = self.tree.shape[0]
        idx = 0
        current_node = self.tree[idx]
        # search through tree until a leaf is found
        while not self.is_leaf(current_node):
            # determine the index of the next node
            if self.next_is_right(current_node, x_row_arr):
                idx += self.relative_right_idx(current_node)
            else:
                idx += 1
            if idx >= max_row:
                break
            current_node = self.tree[idx]
        return current_node[1]

    # shapes x and y data and calls build tree
    # helper functions - build_tree
    def add_evidence(self, x, y) -> None:
        # converting this to the form used in lecture to build the tree
        data = np.column_stack((x, y))
        self.tree = self.build_tree(data)

    # find y values associated with instances of x feature data
    def query(self, x) -> np.ndarray:
        y = np.empty(0)
        for rows in x:
            result = self.search_tree(rows)
            y = np.append(y, result)
        return y

