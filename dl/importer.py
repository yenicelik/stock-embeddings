
def import_data(directory):
    """
    :param directory: A directory (string)
    :return: A numpy array of size (stocks, dates, features).
    """
    raise NotImplementedError

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
    example_path = "/Users/david/deeplearning/data/amex_dataset"


