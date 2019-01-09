import time
import os
import numpy as np
import pandas as pd

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

    offset = 0

    # Load existing dataframe if starting from later item
    if offset > 0:
        filenames = filenames[offset:]
        result = pd.read_csv(datasave)
    else:
        result = None # pd.DataFrame(columns=['Label', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt', 'ReturnOpenPrevious', 'ReturnOpenNext'])

    pool = multiprocessing.Pool()
    print("Starting map...")
    start_time = time.time()
    all_dfs = pool.map(_get_single_dataframe, filenames)
    all_dfs = list(all_dfs)
    print("Took so many seconds", time.time() - start_time)

    for x in all_dfs:
        print(x.head(2))

    print("All dfs are: ", len(all_dfs))

    result = pd.concat(all_dfs, ignore_index=True)

    print(result.head(2))
    print(len(result))

    print("Saving...")
    result.to_csv(datasave)
    print("Saved..")

    # for c, filename in enumerate(filenames):
    #
    #     start_time = time.time()
    #
    #     df = _get_single_dataframe(filename)
    #
    #
    #     if result is None:
    #         result = df
    #         result.to_csv(datasave)
    #     else:
    #         result = result.append(df, ignore_index=True)
    #
    #     if c % 10 == 0:
    #         print(offset + c, " items took ", time.time() - start_time)
    #         print(df.head(2))
    #         # Append to file
    #
    #         # Save the intermediate results to disk
    #
    #     if c % 100 == 0:
    #         print("Saving to disk: ", offset + c)
    #         result.to_csv(datasave)
    #         print("Saved...")

    return result

def import_data(development=False):
    """

    :param development:
    :return:
    """
    datasave = os.getenv("DATAPATH_PROCESSED") if not development else os.getenv("DATAPATH_PROCESSED_DEV")
    out = pd.read_csv(datasave)
    print(out.head(2))

    return out

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

    result = preprocess_individual_csvs_to_one_big_csv(development=False)
    print(result.head(2))

    # create_train_val_test_split(full_dataset)
