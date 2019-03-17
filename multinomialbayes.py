from scipy import cluster
from bayes import Bayes
import pandas as pd
import numpy as np
from discretization import discretize_equal_width, discretize_k_means, discretize_equal_frequency, \
    NO_DISCRETIZATION, EQUAL_FREQUENCY, K_MEANS, EQUAL_WIDTH


class MultinomialBayes(Bayes):
    def __init__(self):
        self.values_numerosity = {}
        self.class_numerosity = {}
        self.class_probabilities = {}
        self.bins = {}

    def fit(self, X, y, sample_weight=None):
        # Fit the model according to the given training data.
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

                for column in self.values_numerosity[class_name]:
                    sample = X.loc[i, [column]]
                    value = sample[column]

                    # smoothing by just adding 1
                    if value not in self.values_numerosity[class_name][column] \
                            or self.values_numerosity[class_name][column][value] == 0:
                        self.values_numerosity[class_name][column][value] = 0.001

                    prob_partial[class_name][column] = (self.values_numerosity[class_name][column][value]) \
                                                      / (self.class_numerosity[class_name])
                    prob[class_name] *= prob_partial[class_name][column]

            maximum = max(prob.values())
            pred_class = [name for name, value in prob.items() if value == maximum]  # all elements with value == max
            y_pred.append(pred_class[0])  # we simply take the first one

        return y_pred  # class labels for samples in X

    def get_params(self, deep=None):
        # Get parameters for this estimator.
        # print('getting params')
        params = {}
        return params

    def set_params(self, **params):
        # Set the parameters of this estimator.
        # print('setting params')
        return self

    def _count_elements_by_class_column(self, X, y):
        for column in X.columns:
            for value, i in zip(X[column], X[column].index.values):
                if y[i] not in self.values_numerosity:
                    self.values_numerosity[y[i]] = {}

                if column not in self.values_numerosity[y[i]]:
                    self.values_numerosity[y[i]][column] = {}

                if value not in self.values_numerosity[y[i]][column]:
                    self.values_numerosity[y[i]][column][value] = 0

                self.values_numerosity[y[i]][column][value] += 1

    def _count_class_probabilities(self, X, y):
        for value, i in zip(X[X.columns[0]], X[X.columns[0]].index.values):
            if y[i] not in self.class_numerosity:
                self.class_numerosity[y[i]] = 0
            self.class_numerosity[y[i]] += 1

        # count probability of each class
        for class_name in set(y):
            if class_name in self.class_numerosity:
                self.class_probabilities[class_name] = self.class_numerosity[class_name] / len(y)

