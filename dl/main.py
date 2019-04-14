import argparse
from sys import platform


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dl.data_loader import import_data, preprocess_individual_csvs_to_one_big_csv, preprocess
from dl.feature_selection.earthquake import Earthquake
from dl.model.baseline import BaselineModel
from dl.model.baseline_noembedding import BaselineModelNoEmbedding
from dl.model.decision_tree import DecisionTree
from dl.model.random import RandomClassifier
from dl.model.xgboost_classifier import XGBoostClassifier

# TODO: @Thomas, I am a bit suspicious of the `train_test_split` function. Does it actually do a split according to the first 75% vs the last 25%? (or in this case, 90%/10%)

# def train_kaggle_baseline_model(development, is_leonhard):
#
#     if is_leonhard:
#         df, encoder_date, encoder_label, decoder_date, decoder_label = preprocess_individual_csvs_to_one_big_csv(development=development, direct_return=True)
#     else:
#         # df = preprocess_individual_csvs_to_one_big_csv(development=development, direct_return=False)
#         df, encoder_date, encoder_label, decoder_date, decoder_label = import_data(development=development)
#
# #from dl.model.xgboost_classifier import XGBoostClassifier
#


def train_kaggle_baseline_model(market_df, development):
    response_col = market_df.columns.get_loc("ReturnOpenNext1")
    numerical_feature_cols = list(market_df.columns[response_col + 1:])
    model = BaselineModel(
        encoder_label,
        number_of_numerical_inputs=len(numerical_feature_cols),
        development=development
    )
    model.keras_model.summary()

    def get_input(market_df, indices):
        X_num = market_df.loc[indices, numerical_feature_cols].values
        X = {'num_input': X_num}
        X['label_input'] = market_df.loc[indices, 'Label'].values
        y = (market_df.loc[indices, 'ReturnOpenNext1'] >= 0).values
        return X, y,


    market_indices, market_test_indices = train_test_split(market_df.index, shuffle=False, test_size=0.1, random_state=23)
    market_train_indices, market_val_indices = train_test_split(market_indices, shuffle=False, test_size=0.1, random_state=23)

    X_train, y_train = get_input(market_df, market_train_indices)
    X_valid, y_valid = get_input(market_df, market_val_indices)
    X_test, y_test = get_input(market_df, market_test_indices)

    model.fit(X_train, y_train.astype(int), X_val=X_valid, y_val=y_valid)


    predict_train = model.predict(X_train)[:, 0] * 2 - 1
    predict_valid = model.predict(X_valid)[:, 0] * 2 - 1
    predict_test = model.predict(X_test)[:, 0] * 2 - 1

    print("Train: ", accuracy_score(predict_train > 0, y_train > 0))
    print("Validation: ", accuracy_score(predict_valid > 0, y_valid > 0))
    print("Test: ", accuracy_score(predict_test > 0, y_test > 0))

def train_kaggle_baseline_noembedding_model(development, is_leonhard):

    if is_leonhard:
        df, encoder_date, encoder_label, decoder_date, decoder_label = preprocess_individual_csvs_to_one_big_csv(development=development, direct_return=True)
    else:
        # df = preprocess_individual_csvs_to_one_big_csv(development=development, direct_return=False)
        df, encoder_date, encoder_label, decoder_date, decoder_label = import_data(development=development)

    market_df = preprocess(df)

    response_col = market_df.columns.get_loc("ReturnOpenNext1")
    scaler = StandardScaler()
    num_feature_cols = list(market_df.columns[response_col + 1:])

    print(market_df.head())
    # TODO: That this is not empty seems to stress me out a bit!!!
    # print(market_df[np.isnan(market_df)].head())

    market_df[num_feature_cols] = scaler.fit_transform(market_df[num_feature_cols])

    print("Done scalar fitting!")

    model = BaselineModelNoEmbedding(encoder_label, num_feature_cols=num_feature_cols, dev=development)
    model.optimizer_definition()
    model.keras_model.summary()


    def get_input(market_df, indices):
        X_num = market_df.loc[indices, num_feature_cols].values
        X = {'num_input': X_num}
        X['label_input'] = market_df.loc[indices, 'Label'].values
        y = (market_df.loc[indices, 'ReturnOpenNext1'] >= 0).values
        return X, y,


    market_indices, market_test_indices = train_test_split(market_df.index, test_size=0.1, random_state=23)
    market_train_indices, market_val_indices = train_test_split(market_indices, test_size=0.1, random_state=23)

    X_train, y_train = get_input(market_df, market_train_indices)
    X_valid, y_valid = get_input(market_df, market_val_indices)
    X_test, y_test = get_input(market_df, market_test_indices)

    model.fit(X_train, y_train.astype(int), X_val=X_valid, y_val=y_valid)

    predict_train = model.predict(X_train)[:, 0] * 2 - 1
    predict_valid = model.predict(X_valid)[:, 0] * 2 - 1
    predict_test = model.predict(X_test)[:, 0] * 2 - 1

    print("No embedding baseline NN")
    print("Train: ", accuracy_score(predict_train > 0, y_train > 0))
    print("Validation: ", accuracy_score(predict_valid > 0, y_valid > 0))
    print("Test: ", accuracy_score(predict_test > 0, y_test > 0))

    model.save_model()

def train_kaggle_baseline_earthquake_model(development, is_leonhard):

    if is_leonhard:
        df, encoder_date, encoder_label, decoder_date, decoder_label = preprocess_individual_csvs_to_one_big_csv(development=development, direct_return=True)
    else:
        # df = preprocess_individual_csvs_to_one_big_csv(development=development, direct_return=False)
        df, encoder_date, encoder_label, decoder_date, decoder_label = import_data(development=development)

    market_df = preprocess(df)

    response_col = market_df.columns.get_loc("ReturnOpenNext1")
    scaler = StandardScaler()
    num_feature_cols = list(market_df.columns[response_col + 1:])

    print(market_df.head())
    # TODO: That this is not empty seems to stress me out a bit!!!
    # print(market_df[np.isnan(market_df)].head())

    market_df[num_feature_cols] = scaler.fit_transform(market_df[num_feature_cols])

    earthquake_model = Earthquake()
    market_df, _ = earthquake_model.transform(market_df)

    print("Done scalar fitting!")
    print("Dataframe now is: ", market_df)

    model = BaselineModel(encoder_label, num_feature_cols=num_feature_cols, dev=development)
    model.optimizer_definition()
    model.keras_model.summary()


    def get_input(market_df, indices):
        X_num = market_df.loc[indices, num_feature_cols].values
        X = {'num_input': X_num}
        X['label_input'] = market_df.loc[indices, 'Label'].values
        y = (market_df.loc[indices, 'ReturnOpenNext1'] >= 0).values
        return X, y,


    market_indices, market_test_indices = train_test_split(market_df.index, test_size=0.1, random_state=23)
    market_train_indices, market_val_indices = train_test_split(market_indices, test_size=0.1, random_state=23)

    X_train, y_train = get_input(market_df, market_train_indices)
    X_valid, y_valid = get_input(market_df, market_val_indices)
    X_test, y_test = get_input(market_df, market_test_indices)

    model.fit(X_train, y_train.astype(int), X_val=X_valid, y_val=y_valid)

    predict_train = model.predict(X_train)[:, 0] * 2 - 1
    predict_valid = model.predict(X_valid)[:, 0] * 2 - 1
    predict_test = model.predict(X_test)[:, 0] * 2 - 1

    print("No embedding baseline NN")
    print("Train: ", accuracy_score(predict_train > 0, y_train > 0))
    print("Validation: ", accuracy_score(predict_valid > 0, y_valid > 0))
    print("Test: ", accuracy_score(predict_test > 0, y_test > 0))

def train_kaggle_baseline_noembedding_earthquake_model(development, is_leonhard):

    if is_leonhard:
        df, encoder_date, encoder_label, decoder_date, decoder_label = preprocess_individual_csvs_to_one_big_csv(development=development, direct_return=True)
        df, encoder_date, encoder_label,decoder_date_decoder_label = preprocess_individual_csvs_to_one_big_csv(development=development, direct_return=True)
    else:
        # df = preprocess_individual_csvs_to_one_big_csv(development=development, direct_return=False)
        df, encoder_date, encoder_label, decoder_date, decoder_label = import_data(development=development)

    market_df = preprocess(df)

    response_col = market_df.columns.get_loc("ReturnOpenNext1")
    numerical_feature_cols = list(market_df.columns[response_col + 1:])

    earthquake_model = Earthquake()
    market_df, _ = earthquake_model.transform(market_df)

    print("Done scalar fitting!")
    print("Dataframe now is: ", market_df)

    model = BaselineModelNoEmbedding(encoder_label, num_feature_cols=numerical_feature_cols, dev=development)
    model.optimizer_definition()
    model.keras_model.summary()


    def get_input(market_df, indices):
        X_num = market_df.loc[indices, num_feature_cols].values
        X = {'num_input': X_num}
        X['label_input'] = market_df.loc[indices, 'Label'].values
        y = (market_df.loc[indices, 'ReturnOpenNext1'] >= 0).values
        return X, y,


    market_indices, market_test_indices = train_test_split(market_df.index, test_size=0.1, random_state=23)
    market_train_indices, market_val_indices = train_test_split(market_indices, test_size=0.1, random_state=23)

    X_train, y_train = get_input(market_df, market_train_indices)
    X_valid, y_valid = get_input(market_df, market_val_indices)
    X_test, y_test = get_input(market_df, market_test_indices)

    model.fit(X_train, y_train.astype(int), X_val=X_valid, y_val=y_valid)

    predict_train = model.predict(X_train)[:, 0] * 2 - 1
    predict_valid = model.predict(X_valid)[:, 0] * 2 - 1
    predict_test = model.predict(X_test)[:, 0] * 2 - 1

    print("No embedding baseline NN")
    print("Train: ", accuracy_score(predict_train > 0, y_train > 0))
    print("Validation: ", accuracy_score(predict_valid > 0, y_valid > 0))
    print("Test: ", accuracy_score(predict_test > 0, y_test > 0))

def train_xgboost_model(development, is_leonhard):

    if is_leonhard:
        df, encoder_date, encoder_label, decoder_date, decoder_label = preprocess_individual_csvs_to_one_big_csv(development=development, direct_return=True)
    else:
        # df = preprocess_individual_csvs_to_one_big_csv(development=development, direct_return=False)
        df, encoder_date, encoder_label, decoder_date, decoder_label = import_data(development=development)

    market_df = preprocess(df)

    response_col = market_df.columns.get_loc("ReturnOpenNext1")
    scaler = StandardScaler()
    num_feature_cols = list(market_df.columns[response_col + 1:])

    print(market_df.head())
    # TODO: That this is not empty seems to stress me out a bit!!!
    # print(market_df[np.isnan(market_df)].head())

    market_df[num_feature_cols] = scaler.fit_transform(market_df[num_feature_cols])

    print("Done scalar fitting!")

    model = XGBoostClassifier()

    def get_input(market_df, indices):
        X = market_df.loc[indices, num_feature_cols].values
        # X = {'num_input': X_num}
        # X['label_input'] = market_df.loc[indices, 'Label'].values
        y = (market_df.loc[indices, 'ReturnOpenNext1'] >= 0).values
        return X, y

    market_indices, market_test_indices = train_test_split(market_df.index, test_size=0.1, random_state=23)
    market_train_indices, market_val_indices = train_test_split(market_indices, test_size=0.1, random_state=23)

    X_train, y_train = get_input(market_df, market_train_indices)
    X_valid, y_valid = get_input(market_df, market_val_indices)
    X_test, y_test = get_input(market_df, market_test_indices)

    print("Fitting xgboost!")
    # print("Inputs are: ", type(X_train), type(y_train))
    # print(X_train)
    model.fit(X_train, y_train.astype(int))

    predict_train = model.predict(X_train) * 2 - 1
    predict_valid = model.predict(X_valid) * 2 - 1
    predict_test = model.predict(X_test) * 2 - 1

    print("Xgboost!")
    print("Train: ", accuracy_score(predict_train > 0, y_train > 0))
    print("Validation: ", accuracy_score(predict_valid > 0, y_valid > 0))
    print("Test: ", accuracy_score(predict_test > 0, y_test > 0))

def train_decisiontree_model(development, is_leonhard):

    if is_leonhard:
        df, encoder_date, encoder_label, decoder_date, decoder_label = preprocess_individual_csvs_to_one_big_csv(development=development, direct_return=True)
    else:
        # df = preprocess_individual_csvs_to_one_big_csv(development=development, direct_return=False)
        df, encoder_date, encoder_label, decoder_date, decoder_label = import_data(development=development)

    market_df = preprocess(df)

    response_col = market_df.columns.get_loc("ReturnOpenNext1")
    scaler = StandardScaler()
    num_feature_cols = list(market_df.columns[response_col + 1:])

    print(market_df.head())
    # TODO: That this is not empty seems to stress me out a bit!!!
    # print(market_df[np.isnan(market_df)].head())

    market_df[num_feature_cols] = scaler.fit_transform(market_df[num_feature_cols])

    print("Done scalar fitting!")

    model = DecisionTree()

    def get_input(market_df, indices):
        X = market_df.loc[indices, num_feature_cols].values
        # X = {'num_input': X_num}
        # X['label_input'] = market_df.loc[indices, 'Label'].values
        y = (market_df.loc[indices, 'ReturnOpenNext1'] >= 0).values
        return X, y

    market_indices, market_test_indices = train_test_split(market_df.index, test_size=0.1, random_state=23)
    market_train_indices, market_val_indices = train_test_split(market_indices, test_size=0.1, random_state=23)

    X_train, y_train = get_input(market_df, market_train_indices)
    X_valid, y_valid = get_input(market_df, market_val_indices)
    X_test, y_test = get_input(market_df, market_test_indices)

    print("Fitting decision tree!!")
    # print("Inputs are: ", type(X_train), type(y_train))
    # print(X_train)
    model.fit(X_train, y_train.astype(int))

    predict_train = model.predict(X_train) * 2 - 1
    predict_valid = model.predict(X_valid) * 2 - 1
    predict_test = model.predict(X_test) * 2 - 1

    print("Decision Tree!!")
    print("Train: ", accuracy_score(predict_train > 0, y_train > 0))
    print("Validation: ", accuracy_score(predict_valid > 0, y_valid > 0))
    print("Test: ", accuracy_score(predict_test > 0, y_test > 0))

def train_random_classifier_model(development, is_leonhard):

    if is_leonhard:
        df, encoder_date, encoder_label, decoder_date, decoder_label = preprocess_individual_csvs_to_one_big_csv(development=development, direct_return=True)
    else:
        # df = preprocess_individual_csvs_to_one_big_csv(development=development, direct_return=False)
        df, encoder_date, encoder_label, decoder_date, decoder_label = import_data(development=development)

    market_df = preprocess(df)

    response_col = market_df.columns.get_loc("ReturnOpenNext1")
    scaler = StandardScaler()
    num_feature_cols = list(market_df.columns[response_col + 1:])

    print(market_df.head())
    # TODO: That this is not empty seems to stress me out a bit!!!
    # print(market_df[np.isnan(market_df)].head())

    market_df[num_feature_cols] = scaler.fit_transform(market_df[num_feature_cols])

    print("Done scalar fitting!")

    model = RandomClassifier()

    def get_input(market_df, indices):
        X = market_df.loc[indices, num_feature_cols].values
        # X = {'num_input': X_num}
        # X['label_input'] = market_df.loc[indices, 'Label'].values
        y = (market_df.loc[indices, 'ReturnOpenNext1'] >= 0).values
        return X, y

    market_indices, market_test_indices = train_test_split(market_df.index, test_size=0.1, random_state=23)
    market_train_indices, market_val_indices = train_test_split(market_indices, test_size=0.1, random_state=23)

    X_train, y_train = get_input(market_df, market_train_indices)
    X_valid, y_valid = get_input(market_df, market_val_indices)
    X_test, y_test = get_input(market_df, market_test_indices)

    print("'Fitting' random classifier!!")
    # print("Inputs are: ", type(X_train), type(y_train))
    # print(X_train)
    model.fit(X_train, y_train.astype(int))

    predict_train = model.predict(X_train) * 2 - 1
    predict_valid = model.predict(X_valid) * 2 - 1
    predict_test = model.predict(X_test) * 2 - 1

    print("Random Classifier!!")
    print("Train: ", accuracy_score(predict_train > 0, y_train > 0))
    print("Validation: ", accuracy_score(predict_valid > 0, y_valid > 0))
    print("Test: ", accuracy_score(predict_test > 0, y_test > 0))

if __name__ == "__main__":

    print("Starting script!")
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--production', action='store_true', default=False,help='production')
    parser.add_argument('--create_big_csv', action='store_true', default=False,help='create_big_csv')
    args = parser.parse_args()
    print("Arguments: production:{}".format(args.production))
    print("Arguments: create_big_csv:{}".format(args.create_big_csv))


    # Make dev true if on linux machine!
    is_linux = (platform == "linux" or platform == "linux2")
    # is_dev = not is_linux

    is_dev = False
    # is_dev = True

    # is_dev = not args.production
    print("Running is_linux:{}, is_dev_{}: ".format(is_linux, is_dev))

    print("Running dev: ", is_dev)
    if args.create_big_csv:
        res= preprocess_individual_csvs_to_one_big_csv(development=is_dev)

    df, encoder_date, encoder_label, decoder_date, decoder_label = import_data(development=is_dev)
    market_df = preprocess(df)
    # load model if not linux
    # train_kaggle_baseline_model(development=is_dev, is_leonhard=is_linux)
    # train_xgboost_model(development=is_dev, is_leonhard=is_linux)
    # train_kaggle_baseline_noembedding_model(development=is_dev, is_leonhard=is_linux)
    # train_kaggle_baseline_earthquake_model(development=is_dev, is_leonhard=is_linux)
    # train_kaggle_baseline_noembedding_earthquake_model(development=is_dev, is_leonhard=is_linux)
    # train_decisiontree_model(development=is_dev, is_leonhard=is_linux)
    # train_random_classifier_model(development=is_dev, is_leonhard=is_linux)
    train_kaggle_baseline_model(market_df=market_df, development=is_dev)
    # train_xgboost_model(development=is_dev, is_leonhard=is_linux)
    #train_kaggle_baseline_noembedding_model(development=is_dev, is_leonhard=is_linux)
