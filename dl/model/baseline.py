

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
