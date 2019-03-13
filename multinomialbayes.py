from scipy import cluster
from bayes import Bayes
import pandas as pd
import numpy as np


NO_DISCRETIZATION = 0
EQUAL_BINS = 1
K_MEANS = 2
EQUAL_SIZE_BINS = 3


class MultinomialBayes(Bayes):
    def __init__(self, discretization_method=NO_DISCRETIZATION, num_of_bins=9):
        self.elements_numerosity = {}
        self.class_numerosity = {}
        self.class_probabilities = {}
        self.discretization_method = discretization_method
        self.bins = {}
        self.num_of_bins = num_of_bins

    def fit(self, X, y, sample_weight=None):
        # Fit the model according to the given training data.

        # discretize X
        if self.discretization_method == EQUAL_BINS:
            X = self._discretize_equal_bins(X)

        if self.discretization_method == K_MEANS:
            X = self._discretize_k_means(X)

        if self.discretization_method == EQUAL_SIZE_BINS:
            X = self._discretize_equal_size_bins(X)

        # count all elements of certain value in X (for each class and column and value)
        self._count_elements_by_class_column(X, y)
        self._count_class_probabilities(X, y)

        # print(self.elements_numerosity)
        return self

    def predict(self, X):  # test
        # Perform classification on samples in X (test)
        y_pred = []
        prob_partial = {}
        prob = {}

        for i in X.index:
            for class_name in self.class_numerosity:
                prob_partial[class_name] = {}
                prob[class_name] = self.class_probabilities[class_name]

                for column in self.elements_numerosity[class_name]:
                    sample = X.loc[i, [column]]
                    value = sample[column]
                    value = self._get_bin(value, column)

                    if value not in self.elements_numerosity[class_name][column] \
                            or self.elements_numerosity[class_name][column][value] == 0:
                        self.elements_numerosity[class_name][column][value] = 1
                        # print("###wygladzanie!!")

                    prob_partial[class_name][column] = (self.elements_numerosity[class_name][column][value]) \
                                                       / (self.class_numerosity[class_name])

                    # TODO: może inne wygładzanie niż po prostu +1, jeśli jest 0?
                    # smoothing: https://towardsdatascience.com/multinomial-naive-bayes-classifier-for-text-analysis-python-8dd6825ece67
                    prob[class_name] *= prob_partial[class_name][column]

            maximum = max(prob.values())
            pred_class = [name for name, value in prob.items() if value == maximum]  # all elements with value == max
            y_pred.append(pred_class[0])  # we simply take the first one

        return y_pred  # class labels for samples in X

    def get_params(self, deep=None):
        # Get parameters for this estimator.
        # print('getting params')
        params = {'discretization_method': self.discretization_method, 'num_of_bins': self.num_of_bins}
        return params

    def set_params(self,**params):
        # Set the parameters of this estimator.
        # print('setting params')
        return self

    def _discretize_equal_bins(self, X):
        for column in X.columns:

            self.bins[column] = np.linspace(start=min(X[column]), stop=max(X[column]), num=self.num_of_bins)
            #print(self.bins)
            X.loc[:, column] = np.digitize(X[column], self.bins[column], right=True)
        #print(X)
        return X

    def _discretize_k_means(self, X):
        for column in X.columns:
            self.bins[column], distortion = cluster.vq.kmeans(X[column].astype(float), self.num_of_bins)
            self.bins[column].tolist()
            self.bins[column].sort()
            X.loc[:, column] = np.digitize(X[column], self.bins[column], right=True)
        #print(X)
        return X

    def _discretize_equal_size_bins(self, X):
        for column in X.columns:
            column_values = X[column].values.copy()
            column_values.sort()
            splitted_values = np.array_split(column_values, self.num_of_bins)
            bins = []
            for array in splitted_values:
                bins.append(min(array))
            self.bins[column] = bins
            X.loc[:, column] = np.digitize(X[column], self.bins[column], right=True)
        return X

    def _count_elements_by_class_column(self, X, y):
        for column in X.columns:
            for value, i in zip(X[column], X[column].index.values):
                if y[i] not in self.elements_numerosity:
                    self.elements_numerosity[y[i]] = {}

                if column not in self.elements_numerosity[y[i]]:
                    self.elements_numerosity[y[i]][column] = {}

                if value not in self.elements_numerosity[y[i]][column]:
                    self.elements_numerosity[y[i]][column][value] = 0

                self.elements_numerosity[y[i]][column][value] += 1

    def _count_class_probabilities(self, X, y):
        for value, i in zip(X[X.columns[0]], X[X.columns[0]].index.values):
            if y[i] not in self.class_numerosity:
                self.class_numerosity[y[i]] = 0
            self.class_numerosity[y[i]] += 1

        # count probability of each class
        for class_name in set(y):
            if class_name in self.class_numerosity:
                self.class_probabilities[class_name] = self.class_numerosity[class_name]/len(y)

    def _get_bin(self, value, column):
        if self.discretization_method == NO_DISCRETIZATION:
            return value

        if self.discretization_method == EQUAL_BINS \
                or self.discretization_method == K_MEANS \
                or self.discretization_method == EQUAL_SIZE_BINS:
            for i, item in enumerate(self.bins[column]):
                if value <= item:
                    return i
