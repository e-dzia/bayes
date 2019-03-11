from abc import abstractmethod, ABC

import numpy as np


class Bayes(ABC):

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        # Fit the model according to the given training data.
        # print('fitting')
        pass

    @abstractmethod
    def predict(self, X):  # test
        # Perform classification on samples in X (test)
        # print('predicting')
        pass

    def score(self, X, y, sample_weight=None):
        # Returns the mean accuracy on the given test data and labels.
        # print('scoring')
        pass

    @abstractmethod
    def get_params(self, deep=None):
        # Get parameters for this estimator.
        # TODO
        # print('getting params')
        pass

    @abstractmethod
    def set_params(self,**params):
        # Set the parameters of this estimator.
        # TODO
        # print('setting params')
        pass


