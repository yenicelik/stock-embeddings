"""
    Implements the batchloader, which can then be use to iteratively
    load in data for the sess.run function
    (one could also use
        https://www.tensorflow.org/api_guides/python/reading_data
    but this simple BatchLoader should do
    )
"""
import numpy as np

class BatchLoader:
    """
        Assume each batch contains all stocks, but only a subset of dates
    """

    def __init__(self, X, Y, batch_size):
        self.counter = 0
        self.epoch = 0
        self.batch_size = batch_size
        self.X = X
        self.Y = Y

    def next(self):
        X_out = self.X[:,
                self.counter * self.batch_size:
                (self.counter + 1) * self.batch_size,
                :]
        Y_out = self.Y[:,
                self.counter * self.batch_size:
                (self.counter + 1) * self.batch_size,
                :]

        self.counter += 1

        return X_out, Y_out


if __name__ == "__main__":
    print("Quick check if the batchload works")
    print("on random datasets which conform with the input type")

    n_stocks = 1000
    n_dates = 10000
    n_features = 200

    X_train = np.random.random((n_stocks, n_dates, n_features))
    Y_train = np.random.random((n_stocks, n_dates, n_features))

    batch_loader = BatchLoader()


    for X_batch, Y_bach in batch_loader.next():
        sess.run(X_batch, Y_bach)