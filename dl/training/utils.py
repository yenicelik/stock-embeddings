"""
    Common functions accross the different training scripts
"""
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from dl.data_loader import dataloader
from dl.training.params import params


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


def _provide_data():
    if params.is_leonhard:
        df, encoder_date, encoder_label, decoder_date, decoder_label = dataloader.preprocess_individual_csvs_to_one_big_csv(
            direct_return=True
        )
    else:
        # df = preprocess_individual_csvs_to_one_big_csv(development=development, direct_return=False)
        df, encoder_date, encoder_label, decoder_date, decoder_label = dataloader.import_data()

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


def get_input(market_df, indices, num_feature_cols, extended=False):
    if extended:
        X_num = market_df.loc[indices, num_feature_cols].values
        X = {'num_input': X_num}
        X['label_input'] = market_df.loc[indices, 'Label'].values
    else:
        X = market_df.loc[indices, num_feature_cols].values
    y = (market_df.loc[indices, 'ReturnOpenNext1'] >= 0).values
    return X, y