import numpy as np

def proposed_loss_function(y_pred, y):
    """
        Implements the loss function as defined in our proposal
    :param y_pred:
    :param y:
    :return:
    """
    raise NotImplementedError


if __name__ == "__main__":
    print("Write a short test if the function works:")

    n_stocks = 1000
    n_dates = 10000
    n_features = 200

    # TODO: Modify the dimensions accordingly.
    Y_pred = np.random.random((n_stocks, n_dates, n_features))
    Y_train = np.random.random((n_stocks, n_dates, n_features))

    y_pred = np.random.random()
    proposed_loss_function(y_pred, Y_train)


