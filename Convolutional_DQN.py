import numpy as np
import tensorflow as tf


class DQN:
    def __init__(self, session, input_row,input_col, output_size, conv1_filter_size=16, conv2_filter_size=16, conv3_filter_size=16, l_rate=0.0002, name="main"):
        self.session = session
        self.input_row = input_row
        self.input_col = input_col
        self.output_size = output_size
        self.net_name = name

        self._build_network(conv1_filter_size,conv2_filter_size,conv3_filter_size,l_rate)

    def _build_network(self, conv1_filter_size, conv2_filter_size,conv3_filter_size, l_rate):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_row,self.input_col], name="input_x")
            self._X_img = tf.reshape(self._X, [-1, self.input_row,self.input_col, 1])

            # First layer of weights
            self._conv1 = tf.layers.conv2d(inputs=self._X_img, filters=conv1_filter_size, kernel_size=[3, 3],padding="SAME", activation=tf.nn.relu)
            # self._pool1 = tf.layers.max_pooling2d(inputs=self._conv1, pool_size=[2, 2], padding="same", strides=2)

            self._conv2 = tf.layers.conv2d(inputs=self._conv1, filters=conv2_filter_size, kernel_size=[3, 3], padding="SAME",activation=tf.nn.relu)
            # self._pool2 = tf.layers.max_pooling2d(inputs=self._conv2, pool_size=[2, 2], padding="same", strides=2)

            self._conv3 = tf.layers.conv2d(inputs=self._conv2, filters=conv3_filter_size, kernel_size=[3, 3],padding="SAME", activation=tf.nn.relu)
            # self._pool3 = tf.layers.max_pooling2d(inputs=self._conv3, pool_size=[2, 2],padding="same", strides=2)
            # Dense Layer with Relu
            self._flat = tf.reshape(self._conv3, [-1, conv3_filter_size * self.input_row * self.input_col])
            # self._flat = tf.reshape(self._pool3, [-1, int(self._pool3.shape[1]) * int(self._pool3.shape[2]) * int(self._pool3.shape[3])])
            # self._flat2 = np.flatten_dtype(self._conv2, [-1, conv2_filter_size * self.input_row * self.input_col])
            self._dense = tf.layers.dense(inputs=self._flat,units=128, activation=tf.nn.relu)
            # hypothesis = fully_connected(h4_drop, final_output_size, activation_fn=None, scope="hypothesis")

            # Q prediction
            self._Qpred = tf.layers.dense(inputs=self._dense, units=self.output_size,activation=tf.nn.softmax)

        # We need to define the parts of the network needed for learning a policy
        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)
        # Loss function
        # self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._Qpred, labels=self._Y))
        # Learning
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)
        self._saver = tf.train.Saver()

    def predict(self, state):
        x = np.reshape(state, [1, self.input_row, self.input_col])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})
