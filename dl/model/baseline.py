"""
    Fully connected NN taken from
        https://www.kaggle.com/christofhenkel/market-data-nn-baseline
"""
import os
import pickle

from keras.regularizers import l1

from dl.data_loader import import_data
from dl.data_loader import preprocess
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization
from keras.losses import binary_crossentropy, mse
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf


def get_input(market_df, indices):
    response_col = market_df.columns.get_loc("ReturnOpenNext1")
    numerical_feature_cols = list(market_df.columns[response_col + 1:])
    X_num = market_df.loc[indices, numerical_feature_cols].values
    X = {'num_input': X_num}
    X['label_input'] = market_df.loc[indices, 'Label'].values
    y = (market_df.loc[indices, 'ReturnOpenNext1'] >= 0).values
    return X, y,


class BaselineModel:
    embedding_dimension = 50

    @property
    def name(self):
        return "BaselineModel"

    def __init__(self, encoder_label, number_of_numerical_inputs, development=True,regression=False):
        self.regression=regression
        self.savepath = os.getenv("MODELPATH_DIR") + self.name
        self.savepath = self.savepath + "_dev.pkl" if development else self.savepath + ".pkl"
        self.keras_modelcheckpoint_path=os.getenv("MODELPATH_DIR") + self.name
        self.keras_modelcheckpoint_path=self.keras_modelcheckpoint_path+"_keras_dev.hdf5" if development else self.keras_modelcheckpoint_path + "_keras.hdf5"
        self.fitted = False

        label_input = Input(shape=[1], name="label_input")
        label_embedding = Embedding(len(encoder_label), self.embedding_dimension)(label_input)
        label_logits = Flatten()(label_embedding)
        label_logits = Dense(32, activation='relu')(label_logits)

        print("Number of numerical inputs is: ", number_of_numerical_inputs)

        numerical_inputs = Input(shape=(number_of_numerical_inputs,), name='num_input')
        numerical_logits = numerical_inputs
        numerical_logits = BatchNormalization()(numerical_logits)  # I do not think this makes sense, since we scaler.fit_transform
        numerical_logits = Dense(128, activation='relu')(numerical_logits)
        numerical_logits = Dense(64, activation='relu')(numerical_logits)

        logits = Concatenate()([numerical_logits, label_logits])
        logits = Dense(64, activation='relu')(logits)
        if regression:
            out = Dense(1,)(logits)
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
            pickle.dump({"keras_model": self.keras_model,}, f)

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

    def fit(self, X, y, X_val, y_val,num_epochs=1,batch_size=64):
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

        print("Embedding dimensions are: ", self.embedding_dimension)

        # self.save_model()

class BaselineModelTensorflow:
    embedding_dimension = 10
    def __init__(self,encoder_label,num_feature_cols):

        self.input_label = tf.placeholder(tf.int32, shape=[None], name="input_label")  #Shape batch_size
        self.embedding_matrix= tf.Variable(tf.random_uniform((len(encoder_label), self.embedding_dimension), -1, 1))
        label_layer_1= tf.nn.embedding_lookup(self.embedding_matrix, self.input_label)  #shape batch_size,embedding_dimension
        label_layer_2 = tf.layers.dense(label_layer_1, 32, activation=tf.nn.relu)#shape batch_size,32

        self.input_numericals = tf.placeholder(tf.float32, shape=[None, len(num_feature_cols)], name="input_numericals") #shape batch_size,len(num_feature_cols)
        numericals_layer_1 = tf.layers.dense(self.input_numericals, 128, activation=tf.nn.relu) #shape batch_size,128
        numericals_layer_2 = tf.layers.dense(numericals_layer_1, 64, activation=tf.nn.relu)#shape batch_size,64

        logits=tf.concat([label_layer_2, numericals_layer_2], 1)  #shape batch_size,92
        label_layer_2 = tf.layers.dense(logits, 64, activation=tf.nn.relu)  # shape batch_size,32
        label_layer_2 = tf.layers.dense(logits, 1, activation=tf.nn.relu)  # shape batch_size,32

        self.input_y = tf.placeholder(tf.int32, shape=[None, 2], name="input_y")



        logits = Dense(64, activation='relu')(logits)
        out = Dense(1, activation='sigmoid')(logits)
        #out = Dense(1, )(logits)
        #something like losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=out)
        #or losses=tf.losses.mean_squared_error(self.input_y,out)

    #def model_definition(self):

    def optimizer_definition(self):
        pass

    def single_pass(self, X_batch, Y_batch):
        pass

    def predict(self, X):
        return self.keras_model.predict(X)

    def fit(self, X, y):
        self.keras_model.fit(X, y,validation_data=(X_val, y_val),epochs=num_epochs,verbose=0,callbacks=[early_stop, check_point])
        self.fitted = True

if __name__ == "__main__":

    df, encoder_date, encoder_label, decoder_date, decoder_label = import_data(development=True)
    market_df=preprocess(df)


    market_indices, market_test_indices = train_test_split(market_df.index, test_size=0.1, random_state=23)
    market_train_indices, market_val_indices = train_test_split(market_indices, test_size=0.1, random_state=23)

    X_train, y_train = get_input(market_df, market_train_indices)
    X_valid, y_valid = get_input(market_df, market_val_indices)
    X_test, y_test = get_input(market_df, market_test_indices)

    number_of_numerical_inputs=X_train['num_input'].shape[1]
    model = BaselineModel(encoder_label, number_of_numerical_inputs, development=True)
    model.keras_model.summary()
    model.fit(X_train, y_train.astype(int), X_val=X_valid, y_val=y_valid,num_epochs=3)
    model.save_model()

    predict_train = model.predict(X_train)[:, 0] * 2 - 1
    predict_valid = model.predict(X_valid)[:, 0] * 2 - 1
    predict_test = model.predict(X_test)[:, 0] * 2 - 1
    print("Train: ", accuracy_score(predict_train > 0, y_train > 0))
    print("Validation: ", accuracy_score(predict_valid > 0, y_valid > 0))
    print("Test: ", accuracy_score(predict_test > 0, y_test > 0))

    # First way to load a model:
    # model.load_model()
    # predict_train = model.predict(X_train)[:, 0] * 2 - 1
    # predict_valid = model.predict(X_valid)[:, 0] * 2 - 1
    # predict_test = model.predict(X_test)[:, 0] * 2 - 1
    # print("Train: ", accuracy_score(predict_train > 0, y_train > 0))
    # print("Validation: ", accuracy_score(predict_valid > 0, y_valid > 0))
    # print("Test: ", accuracy_score(predict_test > 0, y_test > 0))
    # exit(0)

    # Second way to load a model:
    # model.keras_model.load_weights(model.keras_modelcheckpoint_path)
    # predict_train = model.predict(X_train)[:, 0] * 2 - 1
    # predict_valid = model.predict(X_valid)[:, 0] * 2 - 1
    # predict_test = model.predict(X_test)[:, 0] * 2 - 1
    # print("Train: ", accuracy_score(predict_train > 0, y_train > 0))
    # print("Validation: ", accuracy_score(predict_valid > 0, y_valid > 0))
    # print("Test: ", accuracy_score(predict_test > 0, y_test > 0))
    # exit(0)
    #

