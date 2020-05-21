from tree import *


class RandomForestRegressor621(RandomForest621):
    def __init__(self, min_samples_leaf, n_estimators, max_features, oob_score):
        super().__init__(min_samples_leaf, n_estimators, max_features, oob_score,loss=np.std)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def predict(self, X_test):
        y_predict = list()   
        for x in X_test:   
            count = 0
            values = 0 
            for tree in self.root:
                num_leaf, value = tree.predict(x)
                count += num_leaf
                values += value * num_leaf
            y_predict.append(values/count)
        return np.array(y_predict)    

    def oob_calculate(self, score_list, y):
        list_after = [sum(i)/len(i) for i in score_list]
        return r2_score(list_after, y)

    def create_leaf(self, y):
        return LeafNode(y,prediction=sum(y)/len(y))


class RandomForestClassifier621(RandomForest621):
    def __init__(self, min_samples_leaf, n_estimators, max_features, oob_score):
        super().__init__(min_samples_leaf, n_estimators, max_features, oob_score,loss=gini)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return np.count_nonzero((y_predict-y_test) == 0)/len(y_test) 

    def predict(self, X_test):
        y_predict = list()   
        for x in X_test:   
            temp = list()
            for tree in self.root:
                num_leaf, value = tree.predict(x)
                temp.extend([value]*num_leaf)
            y_predict.append(float(stats.mode(temp).mode))
        return np.array(y_predict) 

    def oob_calculate(self, score_list, y):
        list_after = [float(stats.mode(i).mode) for i in score_list]
        return np.count_nonzero((list_after - y) == 0)/len(y)

    def create_leaf(self, y):
        return LeafNode(y,prediction=stats.mode(y).mode)

def gini(y):  # the smaller the better
    list_counter = np.array(list(Counter(list(y.reshape(-1))).values()))
    gini = 1 - sum((list_counter/sum(list_counter))**2)
    return gini


