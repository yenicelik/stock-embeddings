import time
import os
import numpy as np
import modin.pandas as pd

import multiprocessing

from dotenv import load_dotenv
pd.set_option('display.max_columns', 500)
load_dotenv()


def _get_single_dataframe(filename):
    """
        Reads in a single dataframe, and applies some common operations
    :param filename:
    :return:
    """
    df = pd.read_csv(filename, sep=',')

    label, _, _ = filename.split(sep=".")
    label = label.split(sep="/")[-1] if "/" in label else label.split(sep="\\")[-1]
    df['Label'] = label
    # df.insert(loc=0, column='Label', value=filename)
    df['Date'] = pd.to_datetime(df['Date'])

    # TODO: @Thomas what do these variables specify?
    df['ReturnOpenPrevious'] = (df['Open'] - df['Open'].shift(1)) / df['Open'].shift(1)
    df['ReturnOpenNext'] = (df['Open'].shift(-1) - df['Open']) / df['Open']
    df['ReturnOpenNext'] = df['ReturnOpenNext'].astype(np.float32)
    df['ReturnOpenPrevious'] = df['ReturnOpenPrevious'].astype(np.float32)

    # print("Until here takes: ", time.time() - start_time)
    # Change dtype to float32 for faster memory access
    df['Close'] = df['Close'].astype(np.float32)
    df['High'] = df['High'].astype(np.float32)
    df['Low'] = df['Low'].astype(np.float32)
    df['Open'] = df['Open'].astype(np.float32)
    df['OpenInt'] = df['OpenInt'].astype(np.float32)
    df['Volume'] = df['Volume'].astype(np.float32)

    return df

def preprocess_individual_csvs_to_one_big_csv(development=False):
    """

    Create a .env file, and input the path to your source data.
    DATAPATH="/Users/david/deeplearning/"

    The dataset can be downloaded from
    https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/version/3

    # UPDATE: returns a pandas dataframe

    :param directory: A directory (string)
    :return: A numpy array of size (stocks, dates, features).
    """
    import pandas as pd

    datapath = os.getenv("DATAPATH")

    datasave = os.getenv("DATAPATH_PROCESSED") if not development else os.getenv("DATAPATH_PROCESSED_DEV")

    # Read takes 1 minute for 1000 files
    filenames = [x for x in os.listdir(datapath) if ( x[-4:] == '.txt' ) and os.path.getsize(datapath + x) > 0]
    filenames = [datapath + x for x in filenames]
    filenames = list(set(filenames)) # remove any duplicates
    filenames = sorted(filenames) # sort so we can resume reading in individual files

    if development:
        filenames = filenames[:101]

    print("Total number of file: ", len(filenames))

    pool = multiprocessing.Pool()
    print("Starting map...")
    start_time = time.time()
    all_dfs = pool.map(_get_single_dataframe, filenames)
    all_dfs = list(all_dfs)
    print("Took so many seconds", time.time() - start_time)

    print("All dfs are: ", len(all_dfs))

    result = pd.concat(all_dfs, ignore_index=True)

    print(result.head(2))
    print(len(result))

    print("Saving...")
    result.to_csv(datasave)
    print("Saved..")

    import modin.pandas as pd

    return result

def import_data(development=False):
    """

    :param development:
    :return:
    """

    datasave = os.getenv("DATAPATH_PROCESSED") if not development else os.getenv("DATAPATH_PROCESSED_DEV")
    df = pd.read_csv(datasave)
    print("Using dataframe: ", df.head(2))

    stock_symbols = np.sort(np.unique(df['Label'].values))
    dates = np.sort((np.unique(df['Date'].values)))

    # print("Dates are: ", dates) # TODO: There are many non-existing entries here!

    # Turn this into the python dictionary
    encoder_label = {l: idx for idx, l in enumerate(stock_symbols)}
    df['Label'] = [encoder_label.get(i) for i in df['Label']]
    print(df.head(2))
    print("Encode labels: ", encoder_label)

    encoder_date = {l: idx for idx, l in enumerate(dates)}
    df['Date'] = [encoder_date.get(i) for i in df['Date']]
    print(df.head(2))

    no_features = len(['Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt', 'ReturnOpenNext', 'ReturnOpenPrevious'])

    # Creating the numpy array which we will output
    out = np.empty((len(encoder_label), len(encoder_date), no_features))
    out[:, :, :] = None
    print(out.shape)

    matr = df.as_matrix(columns=['Label', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt', 'ReturnOpenNext', 'ReturnOpenPrevious'])
    print("Matrix is: ", matr[:2, :])

    out[matr[:,0].astype(int), matr[:,1].astype(int), :] = matr[:, 2:] # TODO: Double check this! (testing out the first few indices to conform to the dataframe)
    print(out.shape)

    # i = 0
    # for row in df.itertuples():
    #     if i % 100:
    #         print("I is: ", i, len(df))
    #
    #     # print("Label: ", int(row.Label))
    #     # print("Date: ", int(row.Date))
    #     # print("Row is: ", row)
    #     out[int(row.Label), int(row.Date) :] = np.asarray([row.Open, row.High, row.Low, row.Close, row.Volume, row.OpenInt, row.ReturnOpenNext, row.ReturnOpenPrevious]) # row[3:]

    print(out[:2, :])

    return out, encoder_date, encoder_label

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

    # result = preprocess_individual_csvs_to_one_big_csv(development=False)
    # print(result.head(2))

    result = import_data(development=True)

    # create_train_val_test_split(full_dataset)
