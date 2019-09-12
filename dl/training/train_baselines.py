"""
    Train the baseline models. Specifically, this means
    - xgboost
    - decision tree
    - random selection
"""
from sklearn.model_selection import train_test_split

from dl.model.baselines.decision_tree import DecisionTree
from dl.model.baselines.random import RandomClassifier
from dl.model.baselines.xgboost_classifier import XGBoostClassifier
from dl.model.nn.baseline import get_input
from dl.training.utils import _print_accuracy_scores, _provide_data


def train_traditional_models(development, is_leonhard):
    models = [
        ("RandomClassifier", RandomClassifier),
        ("XGBoostClassifier", XGBoostClassifier),
        ("DecisionTreeClassifier", DecisionTree)
    ]

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