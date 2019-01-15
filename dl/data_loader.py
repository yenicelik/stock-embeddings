import time
import os
import pickle
import numpy as np
import pandas as pd

import multiprocessing



from dotenv import load_dotenv
pd.set_option('display.max_columns', 500)
np.set_printoptions(threshold=np.nan)
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
    df.insert(loc=0, column='Label', value=label)
    #df['Label'] = label
    df['Date'] = pd.to_datetime(df['Date'])


    df['ReturnOpenNext1'] = (df['Open'].shift(-1) - df['Open']) / df['Open']
    df['ReturnOpenPrevious1'] = (df['Open'] - df['Open'].shift(1)) / df['Open'].shift(1)
    df['ReturnOpenPrevious2']=(df.Open-df.Open.shift(2))/df.Open.shift(2)
    df['ReturnOpenPrevious5']=(df.Open-df.Open.shift(5))/df.Open.shift(5)

    # df['ReturnOpenNext1'] = df['ReturnOpenNext1'].astype(np.float64)
    # df['ReturnOpenPrevious1'] = df['ReturnOpenPrevious1'].astype(np.float64)
    # df['ReturnOpenPrevious2'] = df['ReturnOpenPrevious2'].astype(np.float64)
    # df['ReturnOpenPrevious5'] = df['ReturnOpenPrevious5'].astype(np.float64)

    # print("Until here takes: ", time.time() - start_time)
    # Change dtype to float32 for faster memory access
    df['Close'] = df['Close'].astype(np.float32)
    df['High'] = df['High'].astype(np.float32)
    df['Low'] = df['Low'].astype(np.float32)
    df['Open'] = df['Open'].astype(np.float32)
    df['OpenInt'] = df['OpenInt'].astype(np.float32)
    df['Volume'] = df['Volume'].astype(np.float32)

    return df

def preprocess_individual_csvs_to_one_big_csv(development=False, direct_return=False):
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
    df = pd.concat(all_dfs, ignore_index=True)

    #Encoding
    stock_symbols = np.sort(df['Label'].unique().astype(str))
    encoder_label = {l: idx for idx, l in enumerate(stock_symbols)}
    df['Label'] = [encoder_label.get(i) for i in df['Label']]
    print(df.head(2))
    print("Encode labels: ", encoder_label)

    dates = np.sort((df['Date'].unique()))
    encoder_date = {l: idx for idx, l in enumerate(dates)}
    df['Date'] = [encoder_date.get(i) for i in df['Date'].values]
    print(df.head(2))

    #Generate matrix out
    # columns = df.columns
    # no_features = len(columns) - 2

    # Creating the numpy array which we will output
    # out = np.empty((len(encoder_label), len(encoder_date), no_features))
    # out[:, :, :] = np.nan
    # print(out.shape)
    # matr = df.values
    # print("Matr is: ", matr[:2, :])
    # out[matr[:, 0].astype(np.int), matr[:, 1].astype(np.int)] = matr[:, 2:]  # np.asarray(matr[:, 2:] for i in range(6))


    # We return the objects immediately, as pickling this file is too big! (if not development!)
    if (not development) and direct_return:
        return df, encoder_date, encoder_label

    if development:
        with open(os.getenv("DATA_PICKLE_DEV"), "wb") as f:
            pickle.dump({
                "encoder_label": encoder_label,
                "encoder_date": encoder_date,
                # "matr": out,
                "df":df
            }, f, protocol=4)
    else:
        with open(os.getenv("DATA_PICKLE"), "wb") as f:
            pickle.dump({
                "encoder_label": encoder_label,
                "encoder_date": encoder_date,
                # "matr": out,
                "df": df
            }, f, protocol=4)


    return df

def import_data(development=False, dataframe_format=False):
    """

    :param development:
    :return df,encoder_date,encoder_label if dataframe_format is True else np.matrix,encoder_date,encoder_label:
    """

    # Check if loading is possible

    pckl_path = os.getenv("DATA_PICKLE_DEV") if development else os.getenv("DATA_PICKLE")

    try:
        with open(pckl_path, "rb") as f:
            obj = pickle.load(f)
        if not("encoder_label" in obj and "encoder_date" in obj and "df" in obj):
            print("Error, not correctly stored")
            return False

        if  dataframe_format:
            return obj["df"], obj["encoder_date"], obj["encoder_label"]
        if not dataframe_format:
            return obj["matr"], obj["encoder_date"], obj["encoder_label"]

    except Exception as e:
        print(e)
        print("No file found! Importaing data...")

    return False

def preprocess(X):
    """
        Some obvious preprocessing (i.e. removing NAN items etc.)
    :param X: df
    :return: df without null rows
    """
    X_hat = X
    # print("Preprocessing!", X.head())
    # X_hat = X.loc[~X.isnull()]
    #
    # print("Stage 1")
    # print(X_hat[np.isnan(X_hat)].head())
    #
    # X_hat = X_hat.dropna() # wasn't quite the case before!
    #
    # print("Stage 2")
    # print(X_hat[np.isnan(X_hat)].head())
    #
    # X_hat = X_hat[~X_hat.isin(['NaN', 'NaT'])]
    #
    # print("Stage 3")
    # print(X_hat[np.isnan(X_hat)].head())
    #
    # X_hat = X_hat.sort_values(['Date', 'Label'])
    #
    # print("Stage 4")
    # print(X_hat[np.isnan(X_hat)].head())
    #
    # X_hat = X_hat[~np.isnan(X_hat)]
    #
    # print("Stage 5")
    # print(X_hat[np.isnan(X_hat)].head())
    #
    #
    # print("Stage 6")
    # print(X_hat[np.isnan(X_hat)].head())
    #
    # print("Preprocessed!", X_hat.head())

    X_hat = X_hat[X_hat.notnull()]
    X_hat = X_hat[np.isfinite(X_hat)]
    X_hat = X_hat.dropna()

    X_hat.reset_index(inplace=True, drop=True)

    return X_hat


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

    df = preprocess_individual_csvs_to_one_big_csv(development=True)
    print(df.shape)

    df, encoder_date, encoder_label = import_data(development=True, dataframe_format=True)
    print(df.shape)


    # create_train_val_test_split(full_dataset)
