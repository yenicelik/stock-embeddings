

from dl.data_loader import import_data
from dl.data_loader import preprocess
from dl.data_loader import preprocess_individual_csvs_to_one_big_csv

class BaselineModel:

    def __init__(self):
        pass

    def model_definition(self):
        pass

    def optimizer_definition(self):
        pass

    def single_pass(self, X_batch, Y_batch):
        pass

    def predict(self, X):
        pass

    def fit(self, X, Y):
        """

        :param X: Full dataset
        :param Y:
        :return:
        """
        pass

class DecisionTree:

    def __init__(self):
        pass

    def transform(self, X):
        pass

    def fit(self, X, Y):
        pass




if __name__ == "__main__":

    result = preprocess_individual_csvs_to_one_big_csv(development=True)

    df, encoder_date, encoder_label = import_data(development=True,dataframe_format=True)
    print(df.head())
    df=preprocess(df)
    print(df.head())

    # for X, Y, in (X_train, Y_train):
    #     model = BaselineModel()
    #     model.fit()

    # create_train_val_test_split(full_dataset)
