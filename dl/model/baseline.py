

from dl.data_loader import import_data
from dl.data_loader import preprocess
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization
from keras.losses import binary_crossentropy, mse
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf


class BaselineModel:
    embedding_dimension = 10
    def __init__(self, encoder_label, num_feature_cols,regression=False):
        self.regression=regression
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
        if regression:
            out = Dense(1,)(logits)
        else:
            out = Dense(1, activation='sigmoid')(logits)
        self.keras_model = Model(inputs=[label_input, numerical_inputs], outputs=out)


    def optimizer_definition(self):
        if self.regression:
            self.keras_model.compile(optimizer='adam', loss='mean_squared_error')
        else:
            self.keras_model.compile(optimizer='adam', loss=binary_crossentropy,metrics=['accuracy'])

    def single_pass(self, X_batch, Y_batch):
        pass

    def predict(self, X):
        return self.keras_model.predict(X)

    def fit(self, X, y,validation_data=None):
        """
        :param X: Full dataset
        :param Y:
        :return:
        """
        from keras.callbacks import EarlyStopping, ModelCheckpoint

        check_point = ModelCheckpoint('model.hdf5', verbose=True, save_best_only=True)
        early_stop = EarlyStopping(patience=5, verbose=True)
        self.keras_model.fit(X, y,epochs=5,verbose=2,validation_data=validation_data)
                 #validation_data=(X_valid, y_valid.astype(int)),
                 # callbacks=[early_stop, metrics=['accuracy']])

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
       pass


class DecisionTree:

    def __init__(self):
        pass

    def transform(self, X):
        pass

    def fit(self, X, Y):
        pass


if __name__ == "__main__":

    df, encoder_date, encoder_label,decoder_date, decoder_label = import_data(development=True,dataframe_format=True)
    market_df=preprocess(df)

    response_col = market_df.columns.get_loc("ReturnOpenNext1")
    scaler = StandardScaler()
    num_feature_cols = list(market_df.columns[response_col + 1:])
    market_df[num_feature_cols] = scaler.fit_transform(market_df[num_feature_cols])

    model=BaselineModel(encoder_label,num_feature_cols)
    model.keras_model.summary()
    model.optimizer_definition()

    def get_input(market_df, indices):
        X_num = market_df.loc[indices, num_feature_cols].values
        X = {'num_input': X_num}
        X['label_input'] = market_df.loc[indices, 'Label'].values
        y = (market_df.loc[indices,'ReturnOpenNext1']>= 0).values
        return X, y,


    market_train_indices, market_val_indices = train_test_split(market_df.index, test_size=0.25, random_state=23)


    X_train, y_train = get_input(market_df, market_train_indices)
    X_valid, y_valid = get_input(market_df, market_val_indices)

    model.fit(X_train, y_train.astype(int),validation_data=(X_valid, y_valid.astype(int)))


    predict_valid = model.predict(X_valid)[:, 0] * 2 - 1
    predict_train = model.predict(X_train)[:, 0] * 2 - 1

    print(accuracy_score(predict_train > 0, y_train > 0))
    print(accuracy_score(predict_valid > 0, y_valid > 0))


