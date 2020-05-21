import numpy as np
from scipy import stats
from collections import Counter
from sklearn.metrics import r2_score
from lolviz import *
from collections import defaultdict

def bootstrap(X, y, size):  # X: n*p 2-d array, y: 1*b 1-d array
    index = np.arange(size)
    index_after = np.random.choice(index, size, replace=True)  # array
    X_after = X[index_after]
    y_after = y[index_after]
    bag_index = np.unique(index_after)  # array
    return X_after, y_after, bag_index

class DecisionNode:
    def __init__(self, col, split, lchild, rchild):  # l/rchild is tree structure
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # "Make decision based upon x_test[col] and split
        if x_test[self.col] <= self.split:
            return self.lchild.predict(x_test) 
        else:
            return self.rchild.predict(x_test)

class LeafNode: 
    def __init__(self, y, prediction):
        # Create leaf node from y values and prediction; prediction is mean(y) or mode(y)
        self.n = len(y)  #  num of observation in the leaf
        self.prediction = prediction

    def predict(self, x_test):  
        return self.n, self.prediction  # return a number and append to array

class RandomForest621:
    def __init__(self, min_samples_leaf, n_estimators, max_features, oob_score, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.n_estimators = n_estimators  # number of trees
        self.max_features = max_features  # a proportion
        self.oob_score = oob_score  # whether to use out-of-bag samples to estimate the R^2 on unseen data.
        self.loss = loss
        self.oob_score_ = np.nan

    def fit(self, X, y):
        tree = list()
        score_dict = defaultdict(list)
        for i in range(0, self.n_estimators):
            X1, y1, index = bootstrap(X, y, size=X.shape[0])
            tree.append(self.func(X1, y1)) # the value is tree-index pair
            oob_index = np.setdiff1d(np.arange(X.shape[0]), index) 
            if self.oob_score:
                for element in oob_index:
                    num_leaf, count = tree[i].predict(X[element])
                    score_dict[element].extend([count] * num_leaf)  # unsorted
        self.score_dict = score_dict
        if self.oob_score:
            score_list = list(dict(sorted(score_dict.items(), key=lambda x: x[0])).values())
            self.oob_score_ = self.oob_calculate(score_list, y[sorted(score_dict.keys())])
        self.root = tree  # list of tree

    def func(self, X, y):
        self.origin_loss = self.loss(y)
        col, split = self.fit_(X,y)
        if col == -1:  # no better split
            return self.create_leaf(y)
        lchild = self.func(X[X[:,col] <= split], y[X[:,col] <= split])  # reduce the row
        rchild = self.func(X[X[:,col] > split], y[X[:,col] > split])
        decision_node = DecisionNode(col,split,lchild,rchild)
        return decision_node
        
    def fit_(self, X, y):  # inherit the origin loss 
        origin_loss = self.origin_loss
        best_loss = {'col':-1, 'split':-1}
        col = X.shape[1]
        row = X.shape[0]
        iterate_col = np.random.choice(list(range(col)), min(round(self.max_features*X.shape[1]), col), replace=False)  # a list of col index
        for i in iterate_col:
            if row >= 11:
                k = np.random.choice(X[:,i], 11, replace=False) # 1_d row array
            else:
                k = X[:,i]  # 1d row array
            for element in k:
                y1 = y[X[:,i] <= element]
                y2 = y[X[:,i] > element]
                if len(y1) < self.min_samples_leaf or len(y2) < self.min_samples_leaf:
                # if len(y1) == 0 or len(y2) == 0:
                    continue
                after_loss = (len(y1)*self.loss(y1) + len(y2)*self.loss(y2))/(len(y1)+len(y2))
                if after_loss < origin_loss:
                    best_loss['col'] = i
                    best_loss['split'] = element
                    origin_loss = after_loss
        return best_loss['col'], best_loss['split']




