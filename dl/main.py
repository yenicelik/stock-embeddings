import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dl.data_loader import import_data, preprocess_individual_csvs_to_one_big_csv, preprocess
from dl.model.baseline import BaselineModel

def train_kaggle_baseline_model(development, load_model):


    # TODO: Run the following line once before you run the rest
    # df, encoder_date, encoder_label = preprocess_individual_csvs_to_one_big_csv(development=development, direct_return=True)
    # df = preprocess_individual_csvs_to_one_big_csv(development=development, direct_return=False)

    df, encoder_date, encoder_label = import_data(development=development)

    market_df = preprocess(df)

    response_col = market_df.columns.get_loc("ReturnOpenNext1")
    scaler = StandardScaler()
    num_feature_cols = list(market_df.columns[response_col + 1:])

    print(market_df.head())
    # TODO: That this is not empty seems to stress me out a bit!!!
    # print(market_df[np.isnan(market_df)].head())

    market_df[num_feature_cols] = scaler.fit_transform(market_df[num_feature_cols])

    print("Done scalar fitting!")

    model = BaselineModel(encoder_label, num_feature_cols=num_feature_cols, dev=development)
    model.optimizer_definition()
    model.keras_model.summary()


    def get_input(market_df, indices):
        X_num = market_df.loc[indices, num_feature_cols].values
        X = {'num_input': X_num}
        X['label_input'] = market_df.loc[indices, 'Label'].values
        y = (market_df.loc[indices, 'ReturnOpenNext1'] >= 0).values
        return X, y,


    market_train_indices, market_val_indices = train_test_split(market_df.index, test_size=0.25, random_state=23)

    X_train, y_train = get_input(market_df, market_train_indices)
    X_valid, y_valid = get_input(market_df, market_val_indices)

    model.fit(X_train, y_train.astype(int), load_model=load_model)

    predict_valid = model.predict(X_valid)[:, 0] * 2 - 1
    predict_train = model.predict(X_train)[:, 0] * 2 - 1

    print(accuracy_score(predict_train > 0, y_train > 0))
    print(accuracy_score(predict_valid > 0, y_valid > 0))

if __name__ == "__main__":
    print("Starting script!")

    from sys import platform


    # Make dev true if on linux machine!
    is_linux = (platform == "linux" or platform == "linux2")
    is_dev = not is_linux

    is_dev = False
    is_dev = True

    print("Running dev: ", is_dev)

    # load model if not linux
    train_kaggle_baseline_model(development=is_dev, load_model=False)
