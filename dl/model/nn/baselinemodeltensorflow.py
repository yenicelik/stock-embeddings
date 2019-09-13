"""
    Fully connected NN taken from
        https://www.kaggle.com/christofhenkel/market-data-nn-baseline
"""
import os

import tensorflow as tf

from dl.training.params import params


class BaselineModelTensorflow:

    @property
    def name(self):
        return "BaselineModelTensorflow"

    def __init__(self, encoder_label, number_of_numerical_inputs, development=True, regression=False):

        self.savepathdir = os.getenv("MODELPATH_DIR") + self.name + ("_dev" if development else "_prod")
        self.savepath = self.savepathdir + "/model"

        self.y = tf.placeholder(tf.float32, shape=[None], name="input_y")

        self.input_label = tf.placeholder(tf.int32, shape=[None], name="input_label")  # Shape batch_size
        self.embedding_matrix = tf.Variable(tf.random_uniform((len(encoder_label), params.embedding_dimension), -1, 1),
                                            name="Embedding")  # Shape (len(encoder_label),embedding_size)
        label_layer_1 = tf.nn.embedding_lookup(self.embedding_matrix,
                                               self.input_label)  # shape (batch_size,embedding_dimension)
        label_layer_2 = tf.layers.dense(label_layer_1, 32, activation=tf.nn.relu)  # shape (batch_size,32)

        self.input_numericals = tf.placeholder(tf.float32, shape=[None, number_of_numerical_inputs],
                                               name="input_numericals")  # shape (batch_size,len(numerical_feature_cols))
        numericals_layer_1 = tf.layers.dense(self.input_numericals, 128, activation=tf.nn.relu)  # shape batch_size,128
        numericals_layer_2 = tf.layers.dense(numericals_layer_1, 64, activation=tf.nn.relu)  # shape batch_size,64

        concat_layer = tf.concat([label_layer_2, numericals_layer_2], 1)  # shape batch_size,96
        out_layer_1 = tf.layers.dense(concat_layer, 64, activation=tf.nn.relu)  # shape batch_size,64
        if regression:
            out_layer_2 = tf.layers.dense(out_layer_1, 1)
        else:
            #   logits = tf.layers.dense(logits_layer, 1, activation=tf.nn.sigmoid)
            out_layer_2 = tf.layers.dense(out_layer_1, 1)  # shape (batch_size,1)
        self.output = tf.reshape(out_layer_2, [-1])
        self.sigmoid_output = tf.math.sigmoid(self.output)
        self.losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.output)
        self.losses_1 = -self.y * tf.math.log(self.sigmoid_output) - (1 - self.y) * tf.math.log(1 - self.sigmoid_output)
        self.loss = tf.reduce_mean(self.losses)

        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

    def fit(self, sess, X_train, y_train, X_val, y_val, num_epochs=1, batch_size=64):
        """
            NOTE! You can also load them model instead of training it!
        :param X: Full dataset
        :param Y:
        :return:
        """
        X_train_label = X_train.get("label_input")
        X_train_numerical = X_train.get('num_input')

        number_of_training_examples = len(X_train_label)
        numer_of_batches_per_epoch = number_of_training_examples // batch_size
        for epoch in range(num_epochs):
            for batch_loop in range(numer_of_batches_per_epoch):
                X_input_label_batch = X_train_label[batch_loop * batch_size:(batch_loop + 1) * batch_size]
                X_input_numericals_batch = X_train_numerical[batch_loop * batch_size:(batch_loop + 1) * batch_size, :]
                y_batch = y_train[batch_loop * batch_size:(batch_loop + 1) * batch_size]
                feed_dict = {
                    self.input_label: X_input_label_batch,
                    self.input_numericals: X_input_numericals_batch,
                    self.y: y_batch, }
                _, step, loss = sess.run(
                    [self.train_op, self.global_step, self.loss],
                    feed_dict)
        self.fitted = True
        return True

    def predict(self, sess, X):

        feed_dict = {self.input_label: X.get("label_input"),
                     self.input_numericals: X.get("num_input")}
        return sess.run(self.sigmoid_output, feed_dict)

    def save_weights(self, sess):
        """
            Saves the model
        :return:
        """
        ret = model.saver.save(sess, self.savepath)
        return ret

    def load_weights(self, sess):
        """
            Loads the model
        :return:
        """
        ckpt = tf.train.get_checkpoint_state(self.savepathdir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(sess, self.savepath)
            self.fitted = True
            return True
        return False
