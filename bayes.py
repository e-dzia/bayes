
class Bayes:
    def __init__(self):
        pass

    def fit(self, X, y, sample_weight=None):
        # Fit the model according to the given training data.
        # TODO
        return self

    def predict(self, X):  # test
        # Perform classification on samples in X (test)
        # TODO
        y_pred = []
        return y_pred  # class labels for samples in X

    def score(self, X, y, sample_weight=None):
        # Returns the mean accuracy on the given test data and labels.
        # TODO
        score = 0
        return score

    def get_params(self, deep=None):
        # Get parameters for this estimator.
        # TODO
        params = {}
        return params

    def set_params(self,**params):
        # Set the parameters of this estimator.
        # TODO
        return self