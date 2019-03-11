from bayes import Bayes
import numpy as np


class MultinomialBayes(Bayes):
    def __init__(self, disretize_method=0):
        self.elements_numerosity = {}
        self.class_numerosity = {}
        self.class_probabilities = {}
        self.discretize_method = disretize_method
        pass

    def fit(self, X, y, sample_weight=None):
        # Fit the model according to the given training data.
        # print('fitting')
        # TODO: discretize X
        if self.discretize_method == 0:
            X = self._discretize_0(X)

        # TODO: count all elements of certain value in X (for each class and column and value)
        for column in X.columns:
            for value, i in zip(X[column], X[column].index.values):
                if y[i] not in self.elements_numerosity:
                    self.elements_numerosity[y[i]] = {}

                if column not in self.elements_numerosity[y[i]]:
                    self.elements_numerosity[y[i]][column] = {}

                if value not in self.elements_numerosity[y[i]][column]:
                    self.elements_numerosity[y[i]][column][value] = 0

                self.elements_numerosity[y[i]][column][value] += 1

        self._count_class_probabilities(X, y)
        return self

    def predict(self, X):  # test
        # Perform classification on samples in X (test)
        # TODO
        # print('predicting')
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
                    if value not in self.elements_numerosity[class_name][column] or self.elements_numerosity[class_name][column][value] == 0:
                        self.elements_numerosity[class_name][column][value] = 1
                    prob_partial[class_name][column] = (self.elements_numerosity[class_name][column][value]) \
                                                       / (self.class_numerosity[class_name])
                    # TODO: może inne wygładzanie niż po prostu +1?
                    # smoothing: https://towardsdatascience.com/multinomial-naive-bayes-classifier-for-text-analysis-python-8dd6825ece67
                    if prob_partial[class_name][column] == 0:
                        print("################podmianka!")
                        prob_partial[class_name][column] = 0.01
                    prob[class_name] *= prob_partial[class_name][column]

            maximum = max(prob.values())
            pred_class = [name for name, value in prob.items() if value == maximum]  # all elements with value == max
            y_pred.append(pred_class[0])  # we just take the first one

        return y_pred  # class labels for samples in X

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

    def _discretize_0(self, X):
        # TODO: chyba trzeba gdzieś zapisać, jakie są przedziały, żeby w "fit" były takie same!
        return X

    def _count_class_probabilities(self, X, y):
        for value, i in zip(X[X.columns[0]], X[X.columns[0]].index.values):
            if y[i] not in self.class_numerosity:
                self.class_numerosity[y[i]] = 0
            self.class_numerosity[y[i]] += 1

        # count probability of each class
        for class_name in set(y):
            if class_name in self.class_numerosity:
                self.class_probabilities[class_name] = self.class_numerosity[class_name]/len(y)
