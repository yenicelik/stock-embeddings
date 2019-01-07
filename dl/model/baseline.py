

class BaselineModel:

    def __init__(self):
        pass

    def model_definition(self):
        pass

    def optimizer_definition(self):
        pass

    def single_pass(self, X_batch, Y_batch):
        pass

    def predict(self, X):
        pass

    def fit(self, X, Y):
        """

        :param X: Full dataset
        :param Y:
        :return:
        """
        pass

class DecisionTree:

    def __init__(self):
        pass

    def transform(self, X):
        pass

    def fit(self, X, Y):
        pass


for X, Y, in (X_train, Y_train):
    model = BaselineModel()
    model.fit()