"""
    Implements the tsfresh feature algorithm, as writte here
    https://tsfresh.readthedocs.io/en/latest/
"""
import numpy as np

class Earthquake:

    def __init__(self):
        pass

    def transform(self, X, Y):
        """

        :param X:
        :param Y:
        :return:
        """
        # TODO: implement the algorithm which transformation which takes in X and returns a modified (X -> X_hat)

        X_hat = X
        Y_hat = Y
        return X_hat, Y_hat

if __name__ == "__main__":
    print("Testing out tsfresh on a random datasample. (Testing if crashes, mostly) ")

    n_stocks = 1000
    n_dates = 10000
    n_features = 200

    X_train = np.random.random((n_stocks, n_dates, n_features))
    Y_train = np.random.random((n_stocks, n_dates, n_features))

    transformer = Earthquake()
    X_hat = transformer.transform(X_train, Y_train)