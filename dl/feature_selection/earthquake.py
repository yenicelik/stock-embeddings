import time

from dl.data_loader import import_data, preprocess_individual_csvs_to_one_big_csv
from dl.data_loader import preprocess
import pandas as pd
import numpy as np
import sys

class Earthquake:

    def __init__(self):
        pass

    def earthquake_check(ResponseSeries, PredictorSeries, number_of_bins=4, number_of_shuffles=100, debug=False):

        bin_indexis = pd.qcut(PredictorSeries, number_of_bins, labels=False)
        result = list()

        for i0_bin in range(number_of_bins):
            start_time = time.time()

            if debug: print("i0_bin:{}".format(i0_bin))
            ResponseSeriesGivenIndex = ResponseSeries.loc[i0_bin == bin_indexis]
            p_given_i0 = np.sum(ResponseSeriesGivenIndex > 0) / len(ResponseSeriesGivenIndex)
            if debug: print("p_given_i0:{}".format(p_given_i0))
            if debug: print("ResponseSeriesGivenIndex.head():\n{}".format(ResponseSeriesGivenIndex.head()))

            ShuffledCopyResponseSeries = ResponseSeries.copy(deep=True)
            shuffled_p_given_i0_list = list()
            for i in range(number_of_shuffles):
                if debug:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                np.random.shuffle(ShuffledCopyResponseSeries.values)
                ShuffledCopyResponseSeriesGivenIndex = ShuffledCopyResponseSeries.loc[i0_bin == bin_indexis]
                shuffled_p_given_i0 = np.sum(ShuffledCopyResponseSeriesGivenIndex > 0) / len(
                    ShuffledCopyResponseSeriesGivenIndex)
                shuffled_p_given_i0_list.append(shuffled_p_given_i0)
            shuffled_p_given_i0_mean = np.mean(shuffled_p_given_i0_list)
            shuffled_p_given_i0_std = np.std(shuffled_p_given_i0_list, ddof=1)
            std_distance_given_i0 = (p_given_i0 - shuffled_p_given_i0_mean) / shuffled_p_given_i0_std
            result.append(std_distance_given_i0)
            if debug:
                print("shuffled_p_given_i0_mean:{}".format(shuffled_p_given_i0_mean))
                print("shuffled_p_given_i0_std:{}".format(shuffled_p_given_i0_std))
                print("std_distance_given_i0:{}".format(std_distance_given_i0))
            print("Time for one bin: ", time.time() - start_time)

        return (np.mean(np.absolute(result)))

    def transform(self, X, Y):
        """

        :param X: is a dataframe
        :param Y: is the dataframe of the response variable
        (must be aligned on the row-index to the dataframe of the input variables)
        :return:
        """
        # TODO: implement the algorithm which transformation which takes in X and returns a modified (X -> X_hat)



        X_hat = X
        Y_hat = Y
        return X_hat, Y_hat

def testcase_1():
    """
        The testcase that thomas had written
    :return:
    """

    number_of_shuffles = 10
    number_of_bins = 10

    development=True

    print("Testing out Earthquake.earthquake")
    # preprocess_individual_csvs_to_one_big_csv(development=development)
    market_df, encoder_date, encoder_label, decoder_date, decoder_label = import_data(development=development)
    market_df = preprocess(market_df)
    print("Starting serieses")

    # 1. experiment with abb
    abb_label = encoder_label.get("abb")
    abb_df = market_df[market_df.Label == abb_label].reset_index(drop=True)
    abb_df.head(10)
    ResponseSeries=abb_df.ReturnOpenNext1
    PredictorSeries=abb_df.ReturnOpenPrevious1
    result = Earthquake.earthquake_check(ResponseSeries, PredictorSeries, debug=False)
    print("result:{}".format(result))

    # 2. experiment: dompare with random numbers
    ResponseSeries = pd.Series(np.random.randn(len(ResponseSeries)))
    PredictorSeries = pd.Series(np.random.randn(len(ResponseSeries)))
    result=Earthquake.earthquake_check(ResponseSeries, PredictorSeries, debug=False)
    print("result:{}".format(result))

    # 3. experiment: compare with full correlation
    ResponseSeries = abb_df.ReturnOpenNext1
    result=Earthquake.earthquake_check(ResponseSeries, ResponseSeries, debug=False)
    print("result:{}".format(result))

    # 4. experiment:. Long input
    ResponseSeries = market_df.ReturnOpenNext1
    PredictorSeries = market_df.ReturnOpenPrevious1
    print(ResponseSeries.shape)
    result = Earthquake.earthquake_check(ResponseSeries, PredictorSeries, number_of_bins=number_of_bins,
                              number_of_shuffles=number_of_shuffles, debug=False)
    print("4. experiment:result:{}".format(result))

    # 5. experiment:. compared to long random input
    ResponseSeries = pd.Series(np.random.randn(len(ResponseSeries)))
    PredictorSeries = pd.Series(np.random.randn(len(ResponseSeries)))
    print(ResponseSeries.shape)
    result = Earthquake.earthquake_check(ResponseSeries, PredictorSeries, number_of_bins=number_of_bins,
                                         number_of_shuffles=number_of_shuffles, debug=False)
    print("5. experiment:result:{}".format(result))

def testcase_rolling_apply():
    """
        Applies the above function for rolling over the
        entire dataframe (like a 1d convolutional filter)
    :return:
    """

    number_of_shuffles = 10
    number_of_bins = 10
    window_size = 10000 # I guess this is an ok window size (approximately per stock)

    development = True

    print("Testing out Earthquake.earthquake")
    # preprocess_individual_csvs_to_one_big_csv(development=development)
    market_df, encoder_date, encoder_label, decoder_date, decoder_label = import_data(development=development)
    market_df = preprocess(market_df)
    print("Starting serieses")
    print("Market df is: ", market_df)

    # 1. experiment with abb
    abb_label = encoder_label.get("abb")
    abb_df = market_df[market_df.Label == abb_label].reset_index(drop=True)
    abb_df.head(10)
    ResponseSeries=abb_df.ReturnOpenNext1
    PredictorSeries=abb_df.ReturnOpenPrevious1

    # lambda functoin we apply over rolling feature
    def earthquake_rowwise(x):
        """
            x must be one individual row from the python row
        :param x:
        :return:
        """
        # TODO: Any other features that could be important here?
        result = Earthquake.earthquake_check(x.ReturnOpenPrevious1, x.ReturnOpenNext1, debug=False)
        return result

    market_df['significance'] = None

    print("Now applying rolling features")
    for i in range(0, len(market_df), window_size):
        result = earthquake_rowwise(market_df.iloc[i:i+window_size])
        market_df.iloc[i:i+window_size, -1] = result
        print("Result is: ", result)

    print("Tail is: ")
    print(market_df.tail(10))
    print("Head is:")
    print(market_df.head(10))



    # result = market_df.rolling(window=10).apply(earthquake_rowwise)
    # print("Results are: ", result)

    # market_df.rolling_apply()
    # result = Earthquake.earthquake_check(ResponseSeries, PredictorSeries, debug=False)
    # print("result:{}".format(result))



if __name__ == "__main__":

    # testcase_1()
    testcase_rolling_apply()
