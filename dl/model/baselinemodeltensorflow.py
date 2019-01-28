"""
    Fully connected NN taken from
        https://www.kaggle.com/christofhenkel/market-data-nn-baseline
"""
import os
import pickle

from dl.data_loader import import_data
from dl.data_loader import preprocess
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf



class BaselineModelTensorflow:
    embedding_dimension = 10

    @property
    def name(self):
        return "BaselineModelTensorflow"


    def __init__(self,encoder_label,numerical_feature_cols, development=True,regression=False):

        self.input_y = tf.placeholder(tf.int32, shape=[None], name="input_y")

        self.input_label = tf.placeholder(tf.int32, shape=[None], name="input_label")  #Shape batch_size
        self.embedding_matrix= tf.Variable(tf.random_uniform((len(encoder_label), self.embedding_dimension), -1, 1))
        label_layer_1= tf.nn.embedding_lookup(self.embedding_matrix, self.input_label)  #shape batch_size,embedding_dimension
        label_layer_2 = tf.layers.dense(label_layer_1, 32, activation=tf.nn.relu)#shape batch_size,32

        self.input_numericals = tf.placeholder(tf.float32, shape=[None, len(numerical_feature_cols)], name="input_numericals") #shape batch_size,len(num_feature_cols)
        numericals_layer_1 = tf.layers.dense(self.input_numericals, 128, activation=tf.nn.relu) #shape batch_size,128
        numericals_layer_2 = tf.layers.dense(numericals_layer_1, 64, activation=tf.nn.relu)#shape batch_size,64

        concat_layer=tf.concat([label_layer_2, numericals_layer_2], 1)  #shape batch_size,92
        out_layer_1= tf.layers.dense(concat_layer, 64, activation=tf.nn.relu)  # shape batch_size,32
        if regression:
            out_layer_2 = tf.layers.dense(out_layer_1, 1)
        else:
         #   logits = tf.layers.dense(logits_layer, 1, activation=tf.nn.sigmoid)
            out_layer_2 = tf.layers.dense(out_layer_1, 1, activation=tf.nn.sigmoid)
        losses=tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y,logits=out_layer_2)
        self.loss = tf.reduce_mean(losses)

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

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True)
        sess = tf.Session(config=session_conf)
        sess.run(tf.global_variables_initializer())

        num_epochs=3
        batch_size=64
        l=X.size[0]
        numer_of_batches_per_epoch=l//batch_size
        for epoch in range(num_epochs):
            for batch_loop in range(numer_of_batches_per_epoch):
                X_batch=X[batch_loop*batch_size,(batch_loop+1)*batch_size,:]
                y_batch=y[batch_loop*batch_size,(batch_loop+1)*batch_size]
                feed_dict = {
                    self.input_y:y_batch,
                    self.input_label: X_batch
                }
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, self.loss],
                    feed_dict)
        self.fitted = True



if __name__ == "__main__":

    df, encoder_date, encoder_label, decoder_date, decoder_label = import_data(development=True)
    market_df=preprocess(df)

    response_col = market_df.columns.get_loc("ReturnOpenNext1")
    numerical_feature_cols = list(market_df.columns[response_col + 1:])
    model=BaselineModelTensorflow(encoder_label,numerical_feature_cols,development=True)

    def get_input(market_df, indices):
        X_num = market_df.loc[indices, numerical_feature_cols].values
        X = {'num_input': X_num}
        X['label_input'] = market_df.loc[indices, 'Label'].values
        y = (market_df.loc[indices,'ReturnOpenNext1']>= 0).values
        return X, y,


    market_indices, market_test_indices = train_test_split(market_df.index, test_size=0.1, random_state=23)
    market_train_indices, market_val_indices = train_test_split(market_indices, test_size=0.1, random_state=23)

    X_train, y_train = get_input(market_df, market_train_indices)
    X_valid, y_valid = get_input(market_df, market_val_indices)
    X_test, y_test = get_input(market_df, market_test_indices)

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
    model.keras_model.load_weights(model.keras_modelcheckpoint_path)
    predict_train = model.predict(X_train)[:, 0] * 2 - 1
    predict_valid = model.predict(X_valid)[:, 0] * 2 - 1
    predict_test = model.predict(X_test)[:, 0] * 2 - 1
    print("Train: ", accuracy_score(predict_train > 0, y_train > 0))
    print("Validation: ", accuracy_score(predict_valid > 0, y_valid > 0))
    print("Test: ", accuracy_score(predict_test > 0, y_test > 0))
    exit(0)



    model.fit(X_train, y_train.astype(int), X_val=X_valid, y_val=y_valid)
    model.save_model()

    predict_train = model.predict(X_train)[:, 0] * 2 - 1
    predict_valid = model.predict(X_valid)[:, 0] * 2 - 1
    predict_test = model.predict(X_test)[:, 0] * 2 - 1
    print("Train: ", accuracy_score(predict_train > 0, y_train > 0))
    print("Validation: ", accuracy_score(predict_valid > 0, y_valid > 0))
    print("Test: ", accuracy_score(predict_test > 0, y_test > 0))

