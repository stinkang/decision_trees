# You may want to install "gprof2dot"
import io
from collections import Counter

import numpy as np
import scipy.io
import sklearn.model_selection
import sklearn.tree
from numpy import genfromtxt
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
import math
import save_csv

from sklearn.model_selection import KFold, ShuffleSplit


import pydot
import matplotlib.pyplot as plt

eps = 1e-5  # a small number
delt = 1e10

class DecisionTree:
    def __init__(self, max_depth=10, feature_labels=None, m=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.max_split_idx, self.max_split = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes
        self.m = m

    @staticmethod
    def H(y):
        terms = []
        counter = Counter()
        for label in y:
            counter[label] += 1
        for value in counter.most_common():
            terms.append(-value[-1]/len(y) * math.log(value[-1]/len(y), 2))
        return sum(terms)
    
    @staticmethod
    def H_after(X, y):
        # TODO implement gini_impurity function
        
        zipped = zip(X, y)
        in_order = sorted(zipped, key=lambda x: x[0])
        sorted_X, sorted_y = zip(*in_order)
        max_x, min_x = sorted_X[-1], sorted_X[0]
        total = sum(y)
        
        C = 0
        c = total
        D = 0
        d = len(y) - total
        previous = sorted_X[0]
        
        weighted_entropies = []
        splits = []
        changed = False
        for i in np.arange(len(sorted_X)):
            # maybe change this so that we're not splitting on exact min/max values
            if sorted_X[i] > previous:
                changed = True
                previous = sorted_X[i]
                left1 = -C * math.log(C/(C + D), 2) if C != 0 else 0
                left2 = -D * math.log(D/(C + D), 2) if D != 0 else 0
                right1 = -c * math.log(c/(c + d), 2) if c != 0 else 0
                right2 = -d * math.log(d/(c + d), 2) if d != 0 else 0
                if (sorted_X[i] == max_x):
                    splits.append(previous - eps)
                elif (sorted_X[i] == min_x):
                    splits.append(previous + eps)
                else:
                    splits.append(previous)
                weighted_entropies.append((1/(len(y)) * (left1 + left2 + right1 + right2)))
            value = sorted_y[i]
            C += value
            c -= value
            D += (value - 1) * -1
            d -= (value - 1) * -1
                
        if not changed:
            return 1, .5
        #print(min(weighted_entropies))
        
        return min(weighted_entropies), splits[np.argmin(weighted_entropies)]
    
    @staticmethod
    def max_information_gain_for_x(X, y):
        # TODO implement information gain function
        gain, split = DecisionTree.H_after(X, y)
        return DecisionTree.H(y) - gain, split

    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        if self.max_depth > 0:
            # compute entropy gain for all single-dimension splits,
            # thresholding with a linear interpolation of 10 values
            gains = []
            splits = []
            # The following logic prevents thresholding on exactly the minimum
            # or maximum values, which may not lead to any meaningful node
            # splits
            if self.m != None:
                features = np.random.choice(range(len(self.features)), m, replace=False)
            else:
                features = range(len(X[0]))
            for i in features:
                gain, split = self.max_information_gain_for_x(X[:, i], y)                        
                gains.append(gain)
                splits.append(split)

            gains = np.nan_to_num(np.array(gains))                                                             
            self.max_split_idx = np.argmax(gains)
            self.max_split = splits[self.max_split_idx]   
                                              
            X0, y0, X1, y1 = self.split(X, y, idx=self.max_split_idx, thresh=self.max_split)
            if X0.size > 0 and X1.size > 0:
                self.left = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features, m=self.m)
                self.left.fit(X0, y0)
                self.right = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features, m=self.m)
                self.right.fit(X1, y1)
            else:
                self.max_depth = 0
                self.data, self.features = X, y
                self.pred = stats.mode(y).mode[0]
        else:
            self.data, self.features = X, y
            self.pred = stats.mode(y).mode[0]
        return self

    def predict(self, X):
        if self.max_depth == 0:
            print("pred")
            print(self.pred)
            return self.pred * np.ones(X.shape[0])
        else:
            print("split")
            print(self.max_split)
            print(self.features[self.max_split_idx])
            X0, idx0, X1, idx1 = self.split_test(X, idx=self.max_split_idx, thresh=self.max_split)
            yhat = np.zeros(X.shape[0])
            print("left")
            yhat[idx0] = self.left.predict(X0)
            print("right")
            yhat[idx1] = self.right.predict(X1)
            return yhat


class BaggedTrees(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None, n=200, m=None):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.m = m
        self.decision_trees = [
            DecisionTree(10, self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        # TODO implement function
        for tree in self.decision_trees:
            bag = np.random.choice(range(len(X)), self.n)
            tree.fit(X[bag], y[bag])

    def predict(self, X):
        # TODO implement function
        predictions = []
        total = np.zeros(len(X))
        for tree in self.decision_trees:
            predictions.append(tree.predict(X))
        for prediction in predictions:
            total = np.add(total, prediction)
        for i in np.arange(len(total)):
            if total[i] / len(predictions) < .5:
                total[i] = 0
            else:
                total[i] = 1
        return total


class RandomForest(BaggedTrees):
    def __init__(self, params=None, n=200, m=4):
        BaggedTrees.__init__(self, params, n, m)
        pass


class BoostedRandomForest(RandomForest):
    def fit(self, X, y):
        self.w = np.ones(X.shape[0]) / X.shape[0]  # Weights on data
        self.a = np.zeros(self.n)  # Weights on decision trees
        # TODO implement function
        return self

    def predict(self, X):
        # TODO implement function
        pass


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    #fill_mode = False

    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            print(term)
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(np.float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack([np.array(data, dtype=np.float), np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        print("here")
        for i in range(data.shape[-1]):
            mode = stats.mode(data[((data[:, i] < -1 - eps) +
                                    (data[:, i] > -1 + eps))][:, i]).mode[0]
            print("mode:")
            print(mode)
            data[(data[:, i] > -1 - eps) * (data[:, i] < -1 + eps)][:, i] = mode
            for j in range(len(data[:, i])):
                if data[j][i] == -1:
                    data[j][i] = mode
            #print (data[(data[:, i] > -1 - eps) * (data[:, i] < -1 + eps)][:, i])

    #if k_nearest_neighbors_mode:
        
        
    return data, onehot_features


def evaluate(clf):
    print("Cross validation", sklearn.model_selection.cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [(features[term[0]], term[1]) for term in counter.most_common()]
        print("First splits", first_splits)


if __name__ == "__main__":
    dataset = "spam"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        print("data:")
        print(data[0])
        print(data[1:, 1:])
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=np.int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription", "creative",
            "height", "featured", "differ", "width", "other", "energy", "business", "message",
            "volumes", "revision", "path", "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis", "square_bracket",
            "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam_data/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        print(X)
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features:", features)
    print("Train/test size:", X.shape, Z.shape)

    print("\n\nPart 0: constant classifier")
    print("Accuracy", 1 - np.sum(y) / y.size)

    # Basic decision tree
    print("\n\nPart (a-b): simplified decision tree")
    dt = DecisionTree(max_depth=3, feature_labels=features)
    dt.fit(X, y)
    print("Predictions", dt.predict(Z))
    save_csv.results_to_csv(dt.predict(Z))
    
    print("kfold")
    
    """
    ss = ShuffleSplit(n_splits=5)
    
    totals = []
    for train_index, test_index in ss.split(X):
        dt = DecisionTree(max_depth=3, feature_labels=features)
        dt.fit(X[train_index], y[train_index])
        predictions = dt.predict(X[test_index])
        total = 0
        for i in range(len(predictions)):
            total += predictions[i] == y[test_index][i]
        totals.append(total/len(predictions))
    print(np.mean(totals))
    
    #bagged
    bt = BaggedTrees(features, 200)
    bt.fit(X, y)
    print("BaggedTrees Predictions", bt.predict(Z))
    
    print("kfold")
    totals1 = []
    for train_index, test_index in ss.split(X):
        bt = BaggedTrees(features, 200)
        bt.fit(X[train_index], y[train_index])
        predictions = bt.predict(X[test_index])
        total = 0
        for i in range(len(predictions)):
            total += predictions[i] == y[test_index][i]
        totals1.append(total/len(predictions))
    print(np.mean(totals1))
    
    #random forest
    rf = RandomForest(features, 400, m = 4)
    rf.fit(X, y)
    print("RandomForest Predictions", rf.predict(Z))
    save_csv.results_to_csv(rf.predict(Z))

    print("kfold")
    totals2 = []
    for train_index, test_index in ss.split(X):
        rf = RandomForest(features, 200, m=4)
        rf.fit(X[train_index], y[train_index])
        predictions = rf.predict(X[test_index])
        total = 0
        for i in range(len(predictions)):
            total += predictions[i] == y[test_index][i]
        totals2.append(total/len(predictions))
    print(np.mean(totals2))
    
    print("\n\nPart (c): sklearn's decision tree")
    clf = sklearn.tree.DecisionTreeClassifier(random_state=0, **params)
    clf.fit(X, y)
    evaluate(clf)
    out = io.StringIO()
    sklearn.tree.export_graphviz(
        clf, out_file=out, feature_names=features, class_names=class_names)
    graph = pydot.graph_from_dot_data(out.getvalue())
    pydot.graph_from_dot_data(out.getvalue())[0].write_pdf("%s-tree.pdf" % dataset)

    # TODO implement and evaluate parts c-h
    """
    # 3.5
    
    validation_accuracies = []
    ss = ShuffleSplit(n_splits=5)
    for i in range(1, 41):
        totals = []
        for train_index, test_index in ss.split(X):
            dt = DecisionTree(max_depth=i, feature_labels=features)
            dt.fit(X[train_index], y[train_index])
            predictions = dt.predict(X[test_index])
            total = 0
            for j in range(len(predictions)):
                total += predictions[j] == y[test_index][j]
            totals.append(total/len(predictions))
        validation_accuracies.append(np.mean(totals))
    plt.plot(range(1, 41), validation_accuracies)
    plt.title("Tree Depth vs Validation Accuracies")
    plt.xlabel("Tree Depth")
    plt.ylabel("Validation Accuracies")
    plt.show()
        
