"""
    Random predictor. Always predicts amongst the possible number of classes with a uniform distribution
"""
from sklearn.dummy import DummyClassifier


class RandomClassifier:

    def __init__(self):
        self.model = DummyClassifier()

    def transform(self, X):
        pass

    def predict(self, X):
        return self.model.predict(X)

    def fit(self, X, Y):
        self.model.fit(X, Y)