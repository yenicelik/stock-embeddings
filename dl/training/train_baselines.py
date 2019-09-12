"""
    Train the baseline models. Specifically, this means
    - xgboost
    - decision tree
    - random selection
"""
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dl.data_loader import dataloader
from dl.model.baselines.decision_tree import DecisionTree
from dl.model.baselines.random import RandomClassifier
from dl.model.baselines.xgboost_classifier import XGBoostClassifier


def _print_accuracy_scores(
        name,
        predict_train,
        y_train,
        predict_valid,
        y_valid,
        predict_test,
        y_test
):
    print(name)
    print("Train: ", accuracy_score(predict_train > 0, y_train > 0))
    print("Validation: ", accuracy_score(predict_valid > 0, y_valid > 0))
    print("Test: ", accuracy_score(predict_test > 0, y_test > 0))


def _provide_data(development, is_leonhard):
    if is_leonhard:
        df, encoder_date, encoder_label, decoder_date, decoder_label = dataloader.preprocess_individual_csvs_to_one_big_csv(
            development=development, direct_return=True)
    else:
        # df = preprocess_individual_csvs_to_one_big_csv(development=development, direct_return=False)
        df, encoder_date, encoder_label, decoder_date, decoder_label = dataloader.import_data(development=development)

    market_df = dataloader.preprocess(df)

    response_col = market_df.columns.get_loc("ReturnOpenNext1")
    scaler = StandardScaler()
    num_feature_cols = list(market_df.columns[response_col + 1:])

    print(market_df.head())
    # TODO: That this is not empty seems to stress me out a bit!!!
    # print(market_df[np.isnan(market_df)].head())

    market_df[num_feature_cols] = scaler.fit_transform(market_df[num_feature_cols])

    print("Done scalar fitting!")

    # TODO: Return the individual items
    return market_df, num_feature_cols


def train_traditional_models(development, is_leonhard):
    models = [
        ("RandomClassifier", RandomClassifier),
        ("XGBoostClassifier", XGBoostClassifier),
        ("DecisionTreeClassifier", DecisionTree)
    ]

    def get_input(market_df, indices, num_feature_cols):
        X = market_df.loc[indices, num_feature_cols].values
        # X = {'num_input': X_num}
        # X['label_input'] = market_df.loc[indices, 'Label'].values
        y = (market_df.loc[indices, 'ReturnOpenNext1'] >= 0).values
        return X, y

    market_df, num_feature_cols = _provide_data(development=development, is_leonhard=is_leonhard)

    for name, model in models:
        model = model()
        market_indices, market_test_indices = train_test_split(market_df.index, test_size=0.1, random_state=23)
        market_train_indices, market_val_indices = train_test_split(market_indices, test_size=0.1, random_state=23)

        X_train, y_train = get_input(market_df, market_train_indices, num_feature_cols)
        X_valid, y_valid = get_input(market_df, market_val_indices, num_feature_cols)
        X_test, y_test = get_input(market_df, market_test_indices, num_feature_cols)

        print("Fitting xgboost!")
        # print("Inputs are: ", type(X_train), type(y_train))
        # print(X_train)
        model.fit(X=X_train, Y=y_train.astype(int))

        predict_train = model.predict(X_train) * 2 - 1
        predict_valid = model.predict(X_valid) * 2 - 1
        predict_test = model.predict(X_test) * 2 - 1

        print("Xgboost!")
        _print_accuracy_scores(
            name=name,
            predict_train=predict_train,
            y_train=y_train,
            predict_valid=predict_valid,
            y_valid=predict_valid,
            predict_test=predict_test,
            y_test=y_test
        )

if __name__ == "__main__":
    print("Starting to train all the traditional baselines")
    train_traditional_models(
        development=True,
        is_leonhard=False
    )