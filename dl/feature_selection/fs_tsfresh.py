"""
    Implements the tsfresh feature algorithm, as writte here
    https://tsfresh.readthedocs.io/en/latest/

    Look at this notebook specifically
    https://github.com/blue-yonder/tsfresh/blob/master/notebooks/timeseries_forecasting_google_stock.ipynb

"""
import numpy as np
import tsfresh
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

from tsfresh.utilities.dataframe_functions import roll_time_series

from dl.data_loader import import_data


class TSFresh:

    def __init__(self):
        self.fitted = False

    def fit(self, timeseries_X, timeseries_y):
        """
            This prepare function can precalculate certain features given a smaller datasample (i.e. if the entire dataset is too huge).
            It will then keep the datasamples as is.

        :param timeseries_X:
        :param timeseries_y:
        :return:
        """

        assert len(timeseries_X) == len(timeseries_y)

        # X_tsfresh containes the extracted tsfresh features
        self.X_tsfresh = extract_features(timeseries_X, column_id="Label", column_sort="Date", n_jobs=8) # TODO: Is the id column the label?
        impute(self.X_tsfresh)

        # which are now filtered to only contain relevant features
        #

        print(self.X_tsfresh)

        self.X_tsfresh_filtered = select_features(self.X_tsfresh, timeseries_y) # y --> Should be the price that should be predicted


        # we can easily construct the corresponding settings object
        self.kind_to_fc_parameters = tsfresh.feature_extraction.settings.from_columns(self.X_tsfresh_filtered.columns) # Previously not .columns!

        self.fitted = True

    def transform(self, timeseries_X, timeseries_y):
        """

        :param X:
        :param Y:
        :return:
        """
        assert self.fitted
        # Will simply

        # Extract the features
        # extracted_features = extract_features(timeseries, column_id="id", column_sort="time")

        # TODO: What should the output of this be? This depends on the input of the neural network
        # and possibly on the input on the input of other models as well
        timeseries = timeseries_X.dropna() # TODO: what to do with nans!
        extracted_features = extract_features(
            timeseries,
            kind_to_fc_parameters=self.kind_to_fc_parameters,
            column_id="Label",
            column_sort="Date",
            n_jobs=16
        )# feature_extraction_settings = MinimalFeatureExtractionSettings())

        # print("Extracted features are: ", extracted_features)
        print(extracted_features.columns)
        print(len(extracted_features.columns))

        # Filter some features
        impute(extracted_features)

        # features_filtered = select_features(extracted_features, timeseries)
        print("IMPUTED")

        # print("Extracted features are: ", extracted_features)
        print(extracted_features.columns)
        print(len(extracted_features.columns))


        return extracted_features



        # # TODO: implement the TSFresh transformation
        # X_hat = X
        # Y_hat = Y
        # return X_hat, Y_hat

if __name__ == "__main__":
    print("Testing out tsfresh on a random datasample. (Testing if crashes, mostly) ")

    n_stocks = 2 # 1000
    n_dates = 2 # 10000
    n_features = 5 # 200

    X_train = np.random.random((n_stocks, n_dates, n_features))
    Y_train = np.random.random((n_stocks, n_dates, n_features))

    # Use the development dataframe
    df = import_data(development=True, dataframe_format=True)

    df['id'] = df[]

    # TODO: Sort all the values from here on already as well

    print("Total length is: ")
    print(len(df))

    # Select a subset of relevant features

    df = df.dropna()

    df_sampled = df.head(6000)
    print("Uniques: ", df_sampled['Label'].unique())

    print(df_sampled.head(2))


    transformer = TSFresh()

    # Drop all na's
    df_sampled = df_sampled.dropna()

    X_df_sampled = df_sampled[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt', 'Label', 'ReturnOpenPrevious']]
    y_df_sampled = df_sampled['ReturnOpenNext'].values[-1] # TODO: y_df_sampled is the very last item

    transformer.fit(X_df_sampled, y_df_sampled)

    # Use the real dataset as well
    df = df.head(100000)
    X_df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt', 'Label', 'ReturnOpenPrevious']]
    y_df = df['ReturnOpenNext'].values


    df_sampled_hat = transformer.transform(X_df_sampled, y_df_sampled)
    # print("Out is: ")
    # print(df_hat.head(2))

