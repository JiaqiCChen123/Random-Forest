import numpy as np
from scipy import stats
from collections import Counter
from sklearn.metrics import r2_score

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
        self.n = len(y)  #  num of data in the leaf
        self.prediction = prediction

    def predict(self, x_test):  
        return self.prediction  # return a number and append to array
    
class DecisionTree621:
    def __init__(self, min_sample_leaf=1,loss=None):
        self.min_sample_leaf = min_sample_leaf
        self.loss = loss  # loss function; either np.std or gini

    def fit(self, X, y):
        self.root = self.func(X,y)

    def func(self, X, y):
        """ 
        Create a decision tree fit to (X,y) and save as self.root,
        the root of our decision tree, for either a classifier or regressor. 
        Leaf nodes for classifiers predict the most common class (the mode)
        and regressors predict the average y for samples in that leaf.
        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.origin_loss = self.loss(y)
        col, split = self.fit_(X,y)
        if col == -1:  # no better split
            return self.create_leaf(y)
        lchild = self.func(X[X[:,col] <= split], y[X[:,col] <= split])  # reduce the row
        rchild = self.func(X[X[:,col] > split], y[X[:,col] > split])
        decision_node = DecisionNode(col,split,lchild,rchild)
        return decision_node
        
    def fit_(self, X, y):  # inherit the origin loss
        """ 
        Recursively create and return a decision tree fit to (X,y) for either a classifier
        or regressor. This function should call self.create_leaf(X,y) to create the appropriate 
        leaf node, which will invoke either RegressionTree621.create_leaf() or ClassifierTree621. 
        create_leaf() depending on the type of self. This function is not part of the class 
        "interface" and is for internal use, but it embodies the decision tree fitting algorithm.
        (Make sure to call fit_() not fit() recursively.) 
        """ 
        origin_loss = self.origin_loss
        best_loss = {'col':-1, 'split':-1}
        col = X.shape[1]
        row = X.shape[0]
        for i in range(0,col):
            if row >= 11:
                k = np.random.choice(X[:,i], 11, replace=False) # 1_d row array
            else:
                k = X[:,i]  # 1d row array
            for element in k:
                y1 = y[X[:,i] <= element]
                y2 = y[X[:,i] > element]
                # if len(y1) < self.min_sample_leaf or len(y2) < self.min_sample_leaf:
                if len(y1) == 0 or len(y2) == 0:
                    continue
                after_loss = (len(y1)*self.loss(y1) + len(y2)*self.loss(y2))/(len(y1)+len(y2))
                if after_loss < origin_loss:
                    best_loss['col'] = i
                    best_loss['split'] = element
                    origin_loss = after_loss
        return best_loss['col'], best_loss['split']

    def predict(self, X_test):
        """ 
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and 
        works for both without modification!
        """
        y_predict = list()
        for x in X_test:  
            y_predict.append(self.root.predict(x))
        return  np.array(y_predict)

class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=np.std) 

    def score(self, X_test, y_test):
        """Return the R^2 of y_test vs predictions for each record in X_test""" 
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def create_leaf(self, y):
        """Return a new LeafNode for regression, passing y and mean(y) to the LeafNode constructor"""
        return LeafNode(y, prediction=sum(y)/len(y))

class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=gini) 

    def score(self, X_test, y_test):
        """Return the accuracy_score() of y_test vs predictions for each record in X_test"""
        y_predict = self.predict(X_test)
        return np.count_nonzero((y_predict-y_test) == 0)/len(y_test) # accuracy

    def create_leaf(self, y):
        """Return a new LeafNode for classification, passing y and mode(y) to the LeafNode constructor."""
        return LeafNode(y, prediction=stats.mode(y).mode)

def gini(y): 
    """Return the gini impurity score for values in y"""
    #  assume the input y is an (1,) array
    list_counter = np.array(list(Counter(list(y.reshape(-1))).values()))
    gini = 1 - sum((list_counter/sum(list_counter))**2)
    return gini