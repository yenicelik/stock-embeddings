def import_data(directory):
    """
    :param directory: A directory (string)
    :return: A numpy array of size (stocks, dates, features).
    """
    raise NotImplementedError


def _preprocess(X, Y):
    """
        Some obvious preprocessing (i.e. removing NAN items etc.)
    :param X:
    :param Y:
    :return:
    """
    X_hat = X
    Y_hat = Y
    return X_hat, Y_hat


def create_train_val_test_split(data):
    """
    :param data: A numpy array of size (stocks, dates, features).
    :return:

        X_train, Y_train, X_val, Y_val, X_test, Y_test,
        each item being a numpy array, again of size (stocks, dates, features)
        (with according subsets of stocks / dates)

    """
    raise NotImplementedError


if __name__ == "__main__":
    example_path = "/Users/david/deeplearning/data/amex-nyse-nasdaq-stock-histories/full_history/"

    full_dataset = import_data(example_path)

    create_train_val_test_split(full_dataset)
