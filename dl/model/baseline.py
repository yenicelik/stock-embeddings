import os
import pickle

from dl.data_loader import import_data
from dl.data_loader import preprocess
from dl.data_loader import preprocess_individual_csvs_to_one_big_csv

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization
from keras.losses import binary_crossentropy, mse
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class BaselineModel:
    embedding_dimension = 10

    @property
    def name(self):
        return "model_kaggle_basepath"

    def __init__(self, encoder_label, num_feature_cols, dev=False):
        self.savepath = os.getenv("MODEL_SAVEPATH_BASEPATH")
        self.savepath = self.savepath + self.name
        self.savepath = self.savepath + "dev.pkl" if dev else self.savepath + ".pkl"
        self.fitted = False

        label_input = Input(shape=[1], name="label_input")
        label_embedding = Embedding(len(encoder_label), self.embedding_dimension)(label_input)
        label_logits = Flatten()(label_embedding)
        label_logits = Dense(32, activation='relu')(label_logits)

        numerical_inputs = Input(shape=(len(num_feature_cols),), name='num_input')
        numerical_logits = numerical_inputs
        numerical_logits = BatchNormalization()(
            numerical_logits)  # I do not think this makes sense, since we scaler.fit_transform
        numerical_logits = Dense(128, activation='relu')(numerical_logits)
        numerical_logits = Dense(64, activation='relu')(numerical_logits)

        logits = Concatenate()([numerical_logits, label_logits])
        logits = Dense(64, activation='relu')(logits)
        # out = Dense(1, activation='sigmoid')(logits)
        out = Dense(1, )(logits)
        self.keras_model = Model(inputs=[label_input, numerical_inputs], outputs=out)

    def save_model(self):
        """
            Saves the model
        :return:
        """
        with open(self.savepath, "wb") as f:
            pickle.dump({
                "keras_model": self.keras_model,
            }, f)

    def load_model(self):
        """
            Loads the model
        :return:
        """
        # TODO: check if weights are saved with pickle
        with open(self.savepath, "rb") as f:
            obj = pickle.load(f)
            self.keras_model = obj["keras_model"]
        if not ("keras_model" in obj):
            print("Error, not correctly stored")
            assert False, ("Model could not be loaded!")
        self.fitted = True

    def optimizer_definition(self):
        self.keras_model.compile(optimizer='adam', loss=binary_crossentropy)

    def predict(self, X):
        return self.keras_model.predict(X)

    def fit(self, X, y, load_model=False):
        """
            NOTE! You can also load them model instead of training it!
        :param X: Full dataset
        :param Y:
        :return:
        """

        if load_model:
            self.load_model()
            print("Loaded model instead of fitting!")
            return True

        from keras.callbacks import EarlyStopping, ModelCheckpoint

        check_point = ModelCheckpoint('model.hdf5', verbose=True, save_best_only=True)
        early_stop = EarlyStopping(patience=5, verbose=True)
        self.keras_model.fit(X, y,
                             # validation_data=(X_valid, y_valid.astype(int)),
                             epochs=6,
                             verbose=False,
                             callbacks=[early_stop, check_point])

        self.save_model()

        self.fitted = True

