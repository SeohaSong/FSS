
import os
import sys

import numpy as np
import tensorflow as tf

from sklearn.metrics import f1_score


class RecurrentNeuralNetwork():
    
    def __init__(self, opt, term, epoch_n, learning_rate, regularization_rate):

        def get_lstm(xs, start, end):
            xs = xs[..., start:end]
            dim_size = end-start
            with tf.variable_scope(str(start)):
                rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(dim_size)                
                outputs, _ = tf.nn.dynamic_rnn(rnn_cell, xs, dtype=tf.float32)
            t = outputs[:, -1, :]
            t = tf.layers.dense(t, max(2, dim_size//2))
            t = tf.layers.dense(t, max(2, dim_size//4))
            t = tf.layers.dense(t, 1)
            return t

        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

        if opt == 'base':
            argses = [[0, 27]]
        elif opt == 'ours':
            argses = [[0, 27], [27, 29], [29, 59], [59, 109], [109, 159],
                      [159, 209]]
        elif opt == 'text':
            argses = [[0, 2], [2, 32], [32, 82], [82, 132], [132, 182]]
        dim_n = argses[-1][-1]

        tf.reset_default_graph()
        graph = tf.get_default_graph()
        tf.random.set_random_seed(0)
        xs_ = tf.placeholder(shape=(None, term, dim_n), dtype=tf.float32)
        ys_ = tf.placeholder(shape=(None,), dtype=tf.float32)
        t_ = tf.concat([get_lstm(xs_, *args) for args in argses], axis=1)
        t_ = tf.layers.dense(t_, 1, activation=tf.nn.sigmoid)
        probs_ = tf.reshape(t_, [-1])
        reg_ = tf.reduce_sum(
            [tf.reduce_sum(tf.abs(v)) for v in tf.trainable_variables()]
        )*regularization_rate
        loss_ = tf.losses.mean_squared_error(ys_, probs_)+reg_
        train_ = tf.train.AdamOptimizer(learning_rate).minimize(loss_)
        init_ = tf.global_variables_initializer()
        
        self._epoch_n = epoch_n
        self._term = term
        self._sess = tf.Session(graph=graph)
        self._tensors = [xs_, ys_, train_, probs_, init_]

    
    def predict(self, tr_xs, tr_ys, te_xs, te_ys, iter_):

        def get_seq(xs, term):
            dim = xs.shape[1]//term
            seq_xs = np.array([xs[:, i*dim:(i+1)*dim] for i in range(term)])
            seq_xs = seq_xs.transpose([1, 0, 2])
            return seq_xs

        sess = self._sess
        term = self._term
        epoch_n = self._epoch_n
        xs_, ys_, train_, probs_, init_ = self._tensors

        tr_xs, te_xs =  get_seq(tr_xs, term), get_seq(te_xs, term)

        prob_mat = []
        max_score = 0
        log = '\r'+("% 3d" % iter_)+' | score: %0.04f | max-score: %0.04f'
        
        sess.run(init_)
        for iter_ in range(epoch_n+50):        
            tr_feed_dict = {xs_: tr_xs, ys_: tr_ys}
            sess.run(train_, feed_dict=tr_feed_dict)
            probs = sess.run(probs_, feed_dict={xs_: te_xs})
            preds = probs.round()
            score = f1_score(te_ys, preds) if sum(preds) else 0
            if score > max_score:
                max_score = score
            sys.stdout.write(log % (score, max_score))
            if iter_ < epoch_n:
                continue
            prob_mat.append(probs)
        sess.close()

        preds = np.mean(prob_mat, axis=0).round()
        score = f1_score(te_ys, preds) if sum(preds) else 0
        print(log % (score, max_score))
        
        return preds
