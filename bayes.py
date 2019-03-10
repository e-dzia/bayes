import numpy as np


class Bayes:
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
        self._count_class_probabilities(y)
        return self

    def predict(self, X):  # test
        # Perform classification on samples in X (test)
        # TODO
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
                    if prob_partial[class_name][column] == 0:
                        print("################podmianka!")
                        prob_partial[class_name][column] = 0.01
                    prob[class_name] *= prob_partial[class_name][column]

            maximum = max(prob.values())
            pred_class = [name for name, value in prob.items() if value == maximum]   # TODO
            y_pred.append(pred_class[0])

        return y_pred  # class labels for samples in X

    def score(self, X, y, sample_weight=None):
        # Returns the mean accuracy on the given test data and labels.
        # TODO
        # print('scoring')
        score = 0
        return score

    def get_params(self, deep=None):
        # Get parameters for this estimator.
        # TODO
        # print('getting params')
        params = {}
        return params

    def set_params(self,**params):
        # Set the parameters of this estimator.
        # TODO
        # print('setting params')
        return self

    def _split_elements_by_class_and_column(self, X, y):
        # get values by class and column
        for column in X.columns:
            for value, i in zip(X[column], X[column].index.values):
                if y[i] not in self.class_numerosity:
                    self.class_numerosity[y[i]] = 0

                if y[i] not in self.elements:
                    self.elements[y[i]] = {}

                if column not in self.elements[y[i]]:
                    self.elements[y[i]][column] = []

                self.elements[y[i]][column].append(value)
                self.class_numerosity[y[i]] += 1

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

    def _count_class_probabilities(self, y):
        # count probability of each class
        for class_name in set(y):
            if class_name in self.class_numerosity:
                self.class_probabilities[class_name] = self.class_numerosity[class_name]/len(y)
