"""
    Fully connected NN taken from
        https://www.kaggle.com/christofhenkel/market-data-nn-baseline
"""
import os
import pickle

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization
from keras.losses import binary_crossentropy

from dl.training.params import params

class BaselineModel:

    @property
    def name(self):
        return "BaselineModel"

    def __init__(self, encoder_label, number_of_numerical_inputs, regression=False):
        self.regression = regression
        self.savepath = os.getenv("MODELPATH_DIR") + self.name + "_emb{}".format(params.embedding_dimension)
        self.savepath = self.savepath + "_dev.pkl" if params.development else self.savepath + ".pkl"
        self.keras_modelcheckpoint_path = os.getenv("MODELPATH_DIR") + self.name
        self.keras_modelcheckpoint_path = self.keras_modelcheckpoint_path + "_keras_dev.hdf5" if params.development else self.keras_modelcheckpoint_path + "_keras.hdf5"
        self.fitted = False

        label_input = Input(shape=[1], name="label_input")
        label_embedding = Embedding(len(encoder_label), params.embedding_dimension)(label_input)
        label_logits = Flatten()(label_embedding)
        label_logits = Dense(32, activation='relu')(label_logits)

        print("Number of numerical inputs is: ", number_of_numerical_inputs)

        numerical_inputs = Input(shape=(number_of_numerical_inputs,), name='num_input')
        numerical_logits = numerical_inputs
        numerical_logits = BatchNormalization()(
            numerical_logits)  # I do not think this makes sense, since we scaler.fit_transform
        numerical_logits = Dense(128, activation='relu')(numerical_logits)
        numerical_logits = Dense(64, activation='relu')(numerical_logits)

        logits = Concatenate()([numerical_logits, label_logits])
        logits = Dense(64, activation='relu')(logits)
        if regression:
            out = Dense(1, )(logits)
        else:
            out = Dense(1, activation='sigmoid')(logits)
        self.keras_model = Model(inputs=[label_input, numerical_inputs], outputs=out)
        if self.regression:
            self.keras_model.compile(optimizer='adam', loss='mean_squared_error')
        else:
            self.keras_model.compile(optimizer='adam', loss=binary_crossentropy, metrics=['accuracy'])

    def save_model(self):
        """
            Saves the model
        :return:
        """
        with open(self.savepath, "wb") as f:
            pickle.dump({"keras_model": self.keras_model, }, f)

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

    def predict(self, X):
        return self.keras_model.predict(X)

    def fit(self, X, y, X_val, y_val, num_epochs=1, batch_size=64):
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

        # check_point = ModelCheckpoint('model.hdf5', verbose=True, save_best_only=True)
        check_point = ModelCheckpoint(self.keras_modelcheckpoint_path, verbose=True)
        early_stop = EarlyStopping(patience=5, verbose=True)
        self.keras_model.fit(X, y,
                             validation_data=(X_val, y_val),
                             epochs=20,
                             verbose=1,
                             callbacks=[early_stop, check_point])

        print("Embedding dimensions are: ", params.embedding_dimension)

        self.save_model()
