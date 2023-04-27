import copy
import numpy as np
import random


class Node:
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.feature_idx = None
        self.feature_value = None
        self.node_prediction = None

    def gini_best_score(self, y, possible_splits):
        best_gain = -np.inf
        best_idx = 0

        # find position of best data split 3 pkt
        for split in possible_splits:
            left_positive = (y[:split+1] == 1).sum()
            left_negative = (y[:split+1] == 0).sum()
            left = left_positive + left_negative

            right_positive = (y[split+1:] == 1).sum()
            right_negative = (y[split+1:] == 0).sum()
            right = right_positive + right_negative

            gini_left = 1 - (left_positive/(left_positive+left_negative))**2 - \
                            (left_negative/(left_positive+left_negative))**2

            gini_right = 1 - (right_positive/(right_positive+right_negative))**2 - \
                             (right_negative/(right_positive+right_negative))**2

            gini_gain = 1 - (left/(left+right))*gini_left - (right/(left+right))*gini_right
            best_gain = max(best_gain, gini_gain)

        return best_idx, best_gain

    def split_data(self, X, y, idx, val):
        left_mask = X[:, idx] < val
        return (X[left_mask], y[left_mask]), (X[~left_mask], y[~left_mask])

    def find_possible_splits(self, data):
        possible_split_points = []
        for idx in range(data.shape[0] - 1):
            if data[idx] != data[idx + 1]:
                possible_split_points.append(idx)
        return possible_split_points

    def find_best_split(self, X, y, feature_subset):
        best_gain = -np.inf
        best_split = None

        # implement feature selection 5 pkt

        features = None

        if feature_subset:
            features = random.sample(range(X.shape[1]), k=feature_subset)
            #features = range(random.randint(0, feature_subset))
            #random.sample(range(100), 10)
        else:
            features = range(X.shape[1])


        for d in features:
            order = np.argsort(X[:, d])
            y_sorted = y[order]
            possible_splits = self.find_possible_splits(X[order, d])
            idx, value = self.gini_best_score(y_sorted, possible_splits)
            if value > best_gain:
                best_gain = value
                best_split = (d, [idx, idx + 1])

        if best_split is None:
            return None, None

        best_value = np.mean(X[best_split[1], best_split[0]])

        return best_split[0], best_value

    def predict(self, x):
        if self.feature_idx is None:
            return self.node_prediction
        if x[self.feature_idx] < self.feature_value:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)

    def train(self, X, y, params):

        self.node_prediction = np.mean(y)
        if X.shape[0] == 1 or self.node_prediction == 0 or self.node_prediction == 1:
            return True

        self.feature_idx, self.feature_value = self.find_best_split(X, y, params["feature_subset"])
        if self.feature_idx is None:
            return True

        (X_left, y_left), (X_right, y_right) = self.split_data(X, y, self.feature_idx, self.feature_value)

        if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            self.feature_idx = None
            return True

        # max tree depth
        if params["depth"] is not None:
            params["depth"] -= 1
        if params["depth"] == 0:
            self.feature_idx = None
            return True

        # create new nodes
        self.left_child, self.right_child = Node(), Node()
        self.left_child.train(X_left, y_left, copy.deepcopy(params))
        self.right_child.train(X_right, y_right, copy.deepcopy(params))
