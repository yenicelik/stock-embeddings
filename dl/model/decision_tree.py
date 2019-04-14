from sklearn.tree import DecisionTreeClassifier


class DecisionTree:

    def __init__(self):
        self.model = DecisionTreeClassifier(min_samples_leaf=5000)

    def transform(self, X):
        pass

    def predict(self, X):
        return self.model.predict(X)

    def fit(self, X, Y):
        self.model.fit(X, Y)