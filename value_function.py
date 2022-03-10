"""
State-Value Function

Written by Patrick Coady (pat-coady.github.io)
Referring largely to https://github.com/mark-gluzman/MulticlassQueuingNetworkPolicyOptimization/blob/master/value_function.py
"""

import numpy as np
# print(np.__file__)
import tensorflow as tf
# print(tf.__file__)
from tensorflow.python.keras.layers import Embedding, Flatten
from tensorflow.python.keras.regularizers import l2
from sklearn.utils import shuffle


class NNValueFunction(object):
    """ NN-based state-value function """

    def __init__(self, obs_dim, hid1_mult, hid3_mult, sz_voc, embed_dim, reg_str=5e-3):
        """
        obs_dim: number of dimensions in observation vector (int)
        hid1_mult: size of first hidden layer, multiplier of obs_dim + embed_dim
        """

        self.replay_buffer_x = None
        self.replay_buffer_x_t = None
        self.replay_buffer_y = None
        self.obs_dim = obs_dim
        self.hid1_mult = hid1_mult
        self.hid3_mult = hid3_mult
        self.sz_voc = sz_voc  # H
        self.embed_dim = embed_dim
        self.reg_str = reg_str
        self.epochs = 10
        self.lr = None  # learning rate set in _build_graph()
        self._build_graph()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def _build_graph(self):
        """ Construct TensorFlow graph, including loss function, init op and train op """

        self.g = tf.Graph()
        with self.g.as_default():
            self.obs_ph = tf.compat.v1.placeholder(tf.float32, (None, self.obs_dim), 'obs_valfunc')
            self.time_ph = tf.compat.v1.placeholder(tf.int32, (None,), 'time_valfunc')
            self.val_ph = tf.compat.v1.placeholder(tf.float32, (None,), 'val_valfunc')
            # hid1 layer size is 10x obs_dim, hid3 size is 10, and hid2 is geometric mean

            hid1_size = (self.obs_dim + self.embed_dim) * self.hid1_mult
            # default multipler 10 chosen empirically on 'Hopper-v1'
            hid3_size = self.hid3_mult  # 5 chosen empirically on 'Hopper-v1'
            hid2_size = int(np.sqrt(hid1_size * hid3_size))
            # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
            self.lr = 1e-4  # 1. * 10 ** (-4)
            # 3 hidden layers with tanh activations
            embed = Embedding(self.sz_voc, self.embed_dim, name='simple_embedding',
                              embeddings_regularizer=l2(self.reg_str), trainable=True)(self.time_ph)
            embed = Flatten()(embed)
            # print('Embedding layer output shape: ', embed.shape)
            # print()
            # x = tf.concat([self.obs_ph, embed], axis=1)
            # print('Input layer shape: ', x.shape)
            out = tf.layers.dense(tf.concat([self.obs_ph, embed], axis=1), hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / (self.obs_dim + self.embed_dim))),
                                  name="h1")
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid1_size)), name="h2")
            out = tf.layers.dense(out, hid3_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid2_size)), name="h3")
            out = tf.layers.dense(out, 1, None,  # tf.nn.relu,  # our true expected rewards-to-go is always non-negative
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid3_size)), name='output')
            # None activation instead of relu activation, because our targets will be standardized!!!
            self.out = out[:, 0]
            # print()
            # print('Value network output shape: ', self.out.shape)
            # print('Value placeholder shape: ', self.val_ph.shape)
            # print()
            self.loss = tf.reduce_mean(tf.square(self.out - self.val_ph))  # squared loss
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def fit(self, x, x_t, y, logger):
        """ Fit model to current data batch + previous data batch
        Args:
            x: features
            x_t: time component; it needs a separate treatment than the rest, e.g., normalization does not apply
            y: target
            logger: logger to save training loss and % explained variance
        """

        num_batches = max(x.shape[0] // 254, 1)
        batch_size = x.shape[0] // num_batches

        if self.replay_buffer_x is None:
            x_train, x_t_train, y_train = x, x_t, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            x_t_train = np.concatenate([x_t, self.replay_buffer_x_t])
            y_train = np.concatenate([y, self.replay_buffer_y])
            # !!!!!!!!!!!!!!!!!!
        self.replay_buffer_x = x  # update the replay buffer
        self.replay_buffer_x_t = x_t
        self.replay_buffer_y = y  # update the replay buffer
        # !!!!!!!!!!!!!!!!!!!!!!
        for e in range(self.epochs):
            x_train, x_t_train, y_train = shuffle(x_train, x_t_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.time_ph: x_t_train[start:end],
                             self.val_ph: y_train[start:end]}
                _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
            y_hat = self.predict(x, x_t)
            loss = np.mean(np.square(y_hat - y))
            print('epoch = ', e)
            print('mean error: ', loss)

        logger.log({'ValFuncLoss': loss})  # loss from last epoch

    def predict(self, x, x_t):
        """ Predict method """

        feed_dict = {self.obs_ph: x, self.time_ph: x_t}
        y_hat = self.sess.run(self.out, feed_dict=feed_dict)
        return y_hat

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()


def test():
    val_func = NNValueFunction(obs_dim=400, hid1_mult=1, hid3_mult=5, sz_voc=360, embed_dim=6)

    print('Changes: \n'
          '1) hid1_mult (from 10 to 1); \n'
          '2) hid3_mult (from 10 to 5); \n'
          '3) Output layer activation (from None to relu); \n' 
          '4) To the input features, add the time component: 0, 1, ..., H-1; ' 
          'this needs a separate treatment compared to the other features, e.g., standardization does not apply; \n')

    print('To tune: \n'
          '1) lr (learning rate); 2) hid1_mult; 3) hid3_mult; 4) embed_dim; 5) reg_str; \n')

    print('Keep in mind: \n'
          '1) We are not using regularization, as by default kernel_regularizer = None! \n'
          '2) We are not using batch normalization! \n')


if __name__ == '__main__':
    test()

