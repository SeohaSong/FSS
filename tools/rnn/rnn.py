
import os

import numpy as np
import tensorflow as tf


class RecurrentNeuralNetwork():
    
    def __init__(self, opt, term):

        def get_lstm(xs, start, end):
            xs = xs[..., start:end]
            dim_size = end-start
            with tf.variable_scope(str(start)):
                rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(dim_size)                
                outputs, _ = tf.nn.dynamic_rnn(rnn_cell, xs, dtype=tf.float32)
            tensor = outputs[:, -1, :]
            tensor = tf.layers.dense(tensor, 16)
            return tensor

        os.environ["CUDA_VISIBLE_DEVICES"]="1"

        if opt == 'ours':
            argses = [[0, 29], [29, 59], [59, 109], [109, 159], [159, 209]]
            dim_n = 209
        elif opt == 'text':
            argses = [[0, 2], [2, 32], [32, 82], [82, 132], [132, 182]]
            dim_n = 182
        elif opt == 'base':
            argses = [[0, 27]]
            dim_n = 27

        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            tf.random.set_random_seed(0)
            xs_ = tf.placeholder(shape=(None, term, dim_n), dtype=tf.float32)
            ys_ = tf.placeholder(shape=(None,), dtype=tf.float32)
            tensor_ = tf.concat([
                get_lstm(xs_, *args) for args in argses
            ], axis=1)
            tensor_ = tf.nn.dropout(tensor_, 0.8)
            tensor_ = tf.layers.dense(tensor_, 1, activation=tf.nn.leaky_relu)
            probs_ = tf.reshape(tf.nn.sigmoid(tensor_), [-1])
            reg_ = tf.reduce_sum([tf.reduce_sum(tf.abs(v))
                                  for v in tf.trainable_variables()])
            loss_ = tf.losses.mean_squared_error(ys_, probs_)+reg_/180
            train_ = tf.train.AdamOptimizer(0.01).minimize(loss_)
            init_ = tf.global_variables_initializer()
        
        self.term = term
        self.sess = tf.Session(graph=graph)
        self.tr_tensors = [xs_, ys_, loss_, train_, init_]
        self.te_tensors = [xs_, ys_, loss_, train_, probs_]

    def _get_seq(self, xs):
        term = self.term
        dim = xs.shape[1]//term
        seq_xs = np.array([xs[:, i*dim:(i+1)*dim] for i in range(term)])
        seq_xs = seq_xs.transpose([1, 0, 2])
        return seq_xs

    def fit(self, tr_xs, tr_ys):
        sess = self.sess
        tr_xs = self._get_seq(tr_xs)
        xs_, ys_, loss_, train_, init_= self.tr_tensors
        log = "\r% 2d | loss: %0.05f | % 3d"
        sess.run(init_)
        for _ in range(150):        
            tr_feed_dict = {xs_: tr_xs, ys_: tr_ys}
            loss, _ = sess.run([loss_, train_], feed_dict=tr_feed_dict)
        self.tr_xs, self.tr_ys = tr_xs, tr_ys

    def predict(self, te_xs):
        sess = self.sess
        te_xs = self._get_seq(te_xs)
        tr_xs, tr_ys = self.tr_xs, self.tr_ys
        xs_, ys_, loss_, train_, probs_ = self.te_tensors
        prob_mat = []
        for _ in range(50):        
            tr_feed_dict = {xs_: tr_xs, ys_: tr_ys}
            loss, _ = sess.run([loss_, train_], feed_dict=tr_feed_dict)
            prob_mat.append(sess.run(probs_, feed_dict={xs_: te_xs}))
        preds = np.mean(prob_mat, axis=0).round()
        sess.close()
        return preds
