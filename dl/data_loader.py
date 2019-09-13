import time
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import multiprocessing

from dotenv import load_dotenv

from dl.training.params import params

pd.set_option('display.max_columns', 500)
np.set_printoptions(threshold=np.nan)
load_dotenv()


def _get_single_dataframe(filename):
    """
        Reads in a single dataframe, and applies some common operations

        This function must be defined outside the class, otherwise, no multiprocessing can be applied!
    :param filename:
    :return:
    """
    df = pd.read_csv(filename, sep=',')

    label, _, _ = filename.split(sep=".")
    label = label.split(sep="/")[-1] if "/" in label else label.split(sep="\\")[-1]
    df.insert(loc=0, column='Label', value=label)
    # df['Label'] = label
    df['Date'] = pd.to_datetime(df['Date'])

    df['ReturnOpenNext1'] = (df['Open'].shift(-1) - df['Open']) / df['Open']
    df['ReturnOpenPrevious1'] = (df['Open'] - df['Open'].shift(1)) / df['Open'].shift(1)
    df['ReturnOpenPrevious2'] = (df.Open - df.Open.shift(2)) / df.Open.shift(2)
    df['ReturnOpenPrevious5'] = (df.Open - df.Open.shift(5)) / df.Open.shift(5)

    # print("Until here takes: ", time.time() - start_time)
    # Change dtype to float32 for faster memory access
    df['Close'] = df['Close'].astype(np.float32)
    df['High'] = df['High'].astype(np.float32)
    df['Low'] = df['Low'].astype(np.float32)
    df['Open'] = df['Open'].astype(np.float32)
    df['OpenInt'] = df['OpenInt'].astype(np.float32)
    df['Volume'] = df['Volume'].astype(np.float32)

    return df


class DataLoader:
    """
        Defines all the information to
    """

    def preprocess_individual_csvs_to_one_big_csv(self, direct_return=False):
        """

        Create a .env file, and input the path to your source data.
        DATAPATH="/Users/david/deeplearning/"

        The dataset can be downloaded from
        https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/version/3

        # UPDATE: returns a pandas dataframe

        :param directory: A directory (string)
        :param development: A boolean
        :return: A numpy array of size (stocks, dates, features).
        """
        import pandas as pd

        datapath = os.getenv("DATAPATH_DIR")

        # Read takes 1 minute for 1000 files
        filenames = [x for x in os.listdir(datapath) if (x[-4:] == '.txt') and os.path.getsize(datapath + x) > 0]
        filenames = [datapath + x for x in filenames]
        filenames = list(set(filenames))  # remove any duplicates
        filenames = sorted(filenames)  # sort so we can resume reading in individual files

        if params.development:
            filenames = filenames[:20]

        print("Total number of file: ", len(filenames))

        pool = multiprocessing.Pool()
        print("Starting map...")
        start_time = time.time()
        all_dfs = pool.map(_get_single_dataframe, filenames)
        all_dfs = list(all_dfs)
        print("Took so many seconds", time.time() - start_time)

        print("All dfs are: ", len(all_dfs))
        df = pd.concat(all_dfs, ignore_index=True)

        # Encoding
        stock_symbols = np.sort(df['Label'].unique().astype(str))
        encoder_label = {l: idx for idx, l in enumerate(stock_symbols)}
        decoder_label = dict(map(reversed, encoder_label.items()))
        df['Label'] = [encoder_label.get(i) for i in df['Label']]

        dates = np.sort((df['Date'].unique()))
        encoder_date = {l: idx for idx, l in enumerate(dates)}
        decoder_date = dict(map(reversed, encoder_date.items()))
        df['Date'] = [encoder_date.get(i) for i in df['Date'].values]

        # We return the objects immediately, as pickling this file is too big! (if not development!)
        if direct_return:
            return df, encoder_date, encoder_label, decoder_date, decoder_label

        if params.development:
            with open(os.getenv("DATA_PICKLE_DEV"), "wb") as f:
                pickle.dump({
                    "encoder_label": encoder_label,
                    "decoder_label": decoder_label,
                    "encoder_date": encoder_date,
                    "decoder_date": decoder_date,
                    "df": df
                }, f, protocol=4)
        else:
            with open(os.getenv("DATA_PICKLE"), "wb") as f:
                pickle.dump({
                    "encoder_label": encoder_label,
                    "decoder_label": decoder_label,
                    "encoder_date": encoder_date,
                    "decoder_date": decoder_date,
                    "df": df
                }, f, protocol=4)

        if direct_return:
            return df, encoder_date, encoder_label, decoder_date, decoder_label
        pkl_dir = os.getenv("DATAPATH_PROCESSED_DIR")
        pkl_file = pkl_dir + "all_dev.pkl" if params.development else pkl_dir + "all.pkl"

        with open(pkl_file, "wb") as f:
            pickle.dump({
                "encoder_label": encoder_label,
                "decoder_label": decoder_label,
                "encoder_date": encoder_date,
                "decoder_date": decoder_date,
                "df": df}, f, protocol=4)
        return df

    def import_data(self):
        """

        :param development:
        :return df,encoder_date,encoder_label if dataframe_format is True else np.matrix,encoder_date,encoder_label:
        """

        # Check if loading is possible

        pkl_dir = os.getenv("DATAPATH_PROCESSED_DIR")
        pkl_file = pkl_dir + "pickle_dev.pkl" if params.development else pkl_dir + "pickle.pkl"

        with open(pkl_file, "rb") as f:
            obj = pickle.load(f)
        if not ("encoder_label" in obj and "encoder_date" in obj and "df" in obj):
            print("Error, not correctly stored")
            return False

        return obj["df"], obj["encoder_date"], obj["encoder_label"], obj["decoder_date"], obj["decoder_label"]

    def __init__(self):
        pass

    def preprocess(self, X):
        """
            Some obvious preprocessing (i.e. removing NAN items etc.)
        :param X: df
        :return: df without null rows
        """
        X_hat = X

        X_hat = X_hat[X_hat.notnull()]
        X_hat = X_hat[np.isfinite(X_hat)]
        X_hat = X_hat.dropna()

        X_hat = X_hat[~((X_hat.ReturnOpenPrevious5 > 5) | (X_hat.ReturnOpenPrevious5 < -0.75))]
        X_hat = X_hat[~((X_hat.ReturnOpenNext1 > 5) | (X_hat.ReturnOpenNext1 < -0.75))]
        X_hat = X_hat.sort_values(['Date', 'Label'])

        response_col = X_hat.columns.get_loc("ReturnOpenNext1")
        scaler = StandardScaler()
        numerical_feature_cols = list(X_hat.columns[response_col + 1:])
        X_hat[numerical_feature_cols] = scaler.fit_transform(X_hat[numerical_feature_cols])
        print("Done scalar fitting!")

        X_hat.reset_index(inplace=True, drop=True)

        return X_hat


dataloader = DataLoader()

if __name__ == "__main__":
    df = dataloader.preprocess_individual_csvs_to_one_big_csv()
    print(df.shape)

    df, encoder_date, encoder_label, decoder_date, decoder_label = dataloader.import_data()
    print(df.shape)

    # create_train_val_test_split(full_dataset)
