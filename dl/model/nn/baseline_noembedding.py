"""
    Fully connected NN taken from
        https://www.kaggle.com/christofhenkel/market-data-nn-baseline
"""
import os
import pickle

from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization
from keras.losses import binary_crossentropy


class BaselineModelNoEmbedding:
    embedding_dimension = 10

    @property
    def name(self):
        return "model_kaggle_noembedding_basepath"

    def __init__(self, encoder_label, num_feature_cols, dev=False, regression=False):
        self.regression = regression
        self.savepath = os.getenv("MODEL_SAVEPATH_BASEPATH")
        self.savepath = self.savepath + self.name
        self.savepath = self.savepath + "dev.pkl" if dev else self.savepath + ".pkl"
        self.fitted = False

        label_input = Input(shape=[1], name="label_input")
        # Simply ignore these input labels. For simplicity, keep the code here

        numerical_inputs = Input(shape=(len(num_feature_cols),), name='num_input')
        numerical_logits = numerical_inputs
        numerical_logits = BatchNormalization()(
            numerical_logits)  # I do not think this makes sense, since we scaler.fit_transform
        numerical_logits = Dense(128, activation='relu')(numerical_logits)
        numerical_logits = Dense(64, activation='relu')(numerical_logits)

        logits = Dense(64, activation='relu')(numerical_logits)
        if regression:
            out = Dense(1, )(logits)
        else:
            out = Dense(1, activation='sigmoid')(logits)
        self.keras_model = Model(inputs=[label_input, numerical_inputs], outputs=out)

    def save_model(self):
        """
            Saves the model
        :return:
        """
        with open(self.savepath, "w") as f:
            pickle.dump({
                "keras_model": self.keras_model,
            }, f)

    def load_model(self):
        """
            Loads the model
        :return:
        """
        # TODO: check if weights are saved with pickle
        with open(self.savepath, "r") as f:
            obj = pickle.load(f)
            self.keras_model = obj["keras_model"]
        if not ("keras_model" in obj):
            print("Error, not correctly stored")
            assert False, ("Model could not be loaded!")
        self.fitted = True

    def optimizer_definition(self):
        if self.regression:
            self.keras_model.compile(optimizer='adam', loss='mean_squared_error')
        else:
            self.keras_model.compile(optimizer='adam', loss=binary_crossentropy, metrics=['accuracy'])

    def predict(self, X):
        return self.keras_model.predict(X)

    def fit(self, X, y, X_val, y_val, load_model=False):
        """
            NOTE! You can also load them model instead of training it!
        :param X: Full dataset
        :param Y:
        :return:
        """

        # if load_model:
        #     self.load_model()
        #     print("Loaded model instead of fitting!")
        #     return True

        from keras.callbacks import EarlyStopping, ModelCheckpoint

        check_point = ModelCheckpoint('model.hdf5', verbose=True, save_best_only=True)
        early_stop = EarlyStopping(patience=5, verbose=True)
        self.keras_model.fit(X, y,
                             validation_data=(X_val, y_val),
                             epochs=20,
                             verbose=1,
                             callbacks=[early_stop, check_point])

        # self.save_model()
