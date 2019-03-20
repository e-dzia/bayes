import sys
from bayes import Bayes
import numpy as np


class GaussianBayes(Bayes):
    def __init__(self):
        self.class_probabilities = {}
        self.means = {}
        self.variances = {}
        self.elements = {}
        self.class_numerosity = {}
        pass

    def fit(self, X, y, sample_weight=None):
        # Fit the model according to the given training data.
        # print('fitting')
        self._split_elements_by_class_and_column(X, y)
        self._count_mean_variances(X, y)
        self._count_class_probabilities(X, y)
        return self

    def predict(self, X):  # test
        # Perform classification on samples in X (test)
        # print('predicting')
        y_pred = []
        prob_partial = {}
        prob = {}

        for i in X.index:
            for class_name in self.class_probabilities:
                prob_partial[class_name] = {}
                prob[class_name] = self.class_probabilities[class_name]

                for column in self.means[class_name]:
                    sample = X.loc[i, [column]]
                    prob_partial[class_name][column] = 1 / np.sqrt(2 * np.pi * (self.variances[class_name][column])) \
                                                       * np.e ** ((-(sample[column] - self.means[class_name][column]) ** 2) /
                                                                  (2 * (self.variances[class_name][column])))
                    # if prob_partial[class_name][column] == 0:  # a case when the values are smaller than min float value
                    #     prob_partial[class_name][column] = 0.00000000000000000001
                    prob[class_name] *= prob_partial[class_name][column]

            maximum = max(prob.values())
            pred_class = [name for name, value in prob.items() if value == maximum]   # TODO
            y_pred.append(pred_class[0])
        return y_pred  # class labels for samples in X

    def get_params(self, deep=None):
        # Get parameters for this estimator.
        # print('getting params')
        params = {}
        return params  # no params

    def set_params(self,**params):
        # Set the parameters of this estimator.
        # print('setting params')
        return self

    def _split_elements_by_class_and_column(self, X, y):
        # get values by class and column
        for column in X.columns:
            for value, i in zip(X[column], X[column].index.values):
                if y[i] not in self.elements:
                    self.elements[y[i]] = {}

                if column not in self.elements[y[i]]:
                    self.elements[y[i]][column] = []

                self.elements[y[i]][column].append(value)

    def _count_mean_variances(self, X, y):
        # count mean and variance of each attribute for every class (Gaussian)
        for class_name in set(y):
            for column in X.columns:
                if class_name not in self.means:
                    self.means[class_name] = {}
                    self.variances[class_name] = {}

                if class_name in self.means and column not in self.means[class_name]:
                    self.means[class_name][column] = []

                self.means[class_name][column] = np.mean(self.elements[class_name][column])
                self.variances[class_name][column] = np.var(self.elements[class_name][column])

    def _count_class_probabilities(self, X, y):
        for value, i in zip(X[X.columns[0]], X[X.columns[0]].index.values):
            if y[i] not in self.class_numerosity:
                self.class_numerosity[y[i]] = 0
            self.class_numerosity[y[i]] += 1

        # count probability of each class
        for class_name in set(y):
            if class_name in self.class_numerosity:
                self.class_probabilities[class_name] = self.class_numerosity[class_name]/len(y)
