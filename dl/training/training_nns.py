"""
    Includes a class which trains the model
"""
import argparse
from sys import platform
import numpy as np

from sklearn.model_selection import train_test_split

from dl.data_loader import dataloader
from dl.model.nn.baseline import BaselineModel
from dl.model.nn.baseline_noembedding import BaselineModelNoEmbedding
from dl.training.utils import _provide_data, get_input, _print_accuracy_scores


class Trainer:
    """
        Handles all the logic to train a model, includes some hyperparameter definitions.
    """

    def __init__(self):
        # Choose one of:
        self.embedding = False
        print("Starting script!")
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--production', action='store_true', default=False, help='production')
        parser.add_argument('--create_big_csv', action='store_true', default=False, help='create_big_csv')
        args = parser.parse_args()
        print("Arguments: production:{}".format(args.production))
        print("Arguments: create_big_csv:{}".format(args.create_big_csv))

        # Make dev true if on linux machine!
        is_linux = (platform == "linux" or platform == "linux2")

        self.development = True
        is_leonhard = False

        df, encoder_date, encoder_label, decoder_date, decoder_label = dataloader.import_data(
            development=self.development
        )

        self.market_df, self.num_feature_cols = _provide_data(
            development=self.development,
            is_leonhard=is_leonhard
        )

        if self.embedding:
            self.current_model = BaselineModel(
                encoder_label,
                number_of_numerical_inputs=len(self.num_feature_cols),
                development=self.development
            )
        else:
            self.current_model = BaselineModelNoEmbedding(
                encoder_label,
                num_feature_cols=self.num_feature_cols,
                dev=self.development
            )
        self.current_model.keras_model.summary()

    def train_nn_models(self):

        if not self.embedding:
            self.current_model.optimizer_definition()

        self.current_model.keras_model.summary()

        market_indices, market_test_indices = train_test_split(
            self.market_df.index,
            shuffle=False,
            test_size=0.1,
            random_state=23
        )
        market_train_indices, market_val_indices = train_test_split(
            market_indices,
            shuffle=False,
            test_size=0.1,
            random_state=23
        )

        X_train, y_train = get_input(
            self.market_df,
            market_train_indices,
            self.num_feature_cols,
            extended=True
        )
        X_valid, y_valid = get_input(
            self.market_df,
            market_val_indices,
            self.num_feature_cols,
            extended=True
        )
        X_test, y_test = get_input(
            self.market_df,
            market_test_indices,
            self.num_feature_cols,
            extended=True
        )

        input("Press a button to continue..")
        print("Fitting model!")
        self.current_model.fit(X_train, y_train.astype(int), X_val=X_valid, y_val=y_valid)

        input("Press a button to continue.. (2)")
        predict_train = self.current_model.predict(X_train) * 2 - 1
        predict_valid = self.current_model.predict(X_valid) * 2 - 1
        predict_test = self.current_model.predict(X_test) * 2 - 1

        _print_accuracy_scores(
            name="embedding" if self.embedding else "no_embedding",
            predict_train=predict_train,
            y_train=y_train,
            predict_valid=predict_valid,
            y_valid=predict_valid,
            predict_test=predict_test,
            y_test=y_test
        )
        input("Done!")

        self.current_model.save_model()
        # if self.embedding:
        #     # Test Items
        #     np.save("/cluster/home/yedavid/embedding_test_predicted.npy", predict_test)
        #     np.save("/cluster/home/yedavid/embedding_test_real.npy", y_test)
        # else:
        #     # Test Items
        #     np.save("/cluster/home/yedavid/no_embedding_test_predicted.npy", predict_test)
        #     np.save("/cluster/home/yedavid/no_embedding_test_real.npy", y_test)


if __name__ == "__main__":
    print("Running training for NNs: ")

    trainer = Trainer()
    trainer.train_nn_models()
