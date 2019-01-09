"""
    Implements the tsfresh feature algorithm, as writte here
    https://tsfresh.readthedocs.io/en/latest/
"""
import numpy as np
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

from dl.data_loader import import_data


class TSFresh:

    def __init__(self):
        pass

    def transform(self, timeseries):
        """

        :param X:
        :param Y:
        :return:
        """

        # Extract the features
        # extracted_features = extract_features(timeseries, column_id="id", column_sort="time")

        # TODO: What should the output of this be? This depends on the input of the neural network
        # and possibly on the input on the input of other models as well
        timeseries = timeseries.dropna() # TODO: what to do with nans!
        extracted_features = extract_features(timeseries, column_id="Label", column_sort="Date", n_jobs=16)# feature_extraction_settings = MinimalFeatureExtractionSettings())

        # print("Extracted features are: ", extracted_features)
        print(extracted_features.columns)
        print(len(extracted_features.columns))

        # Filter some features
        impute(extracted_features)
        features_filtered = select_features(extracted_features, y) # y --> Should be the price that should be predicted

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

    print("Total length is: ")
    print(len(df))

    # Select a subset of relevant features

    # TODO: Look at 1000 items only
    df = df.head(1000)


    print(df.head(2))


    transformer = TSFresh()
    df_hat = transformer.transform(df)
    # print("Out is: ")
    # print(df_hat.head(2))