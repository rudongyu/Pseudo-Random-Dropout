from simulator import Simulator
import tensorflow as tf
import numpy as np
import input_data
import random

class sudoDNN(object):

    def __init__(self, hidden_dim, p):
        self.p = p
        self.sim = Simulator()
        self.hidden_dim = hidden_dim
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.x = tf.placeholder(tf.float32, [None, 784])

        self.idxx = 0
        self.idx = 0
        self.x_dropout_list = self._get_dropout_list(784, p)
        self.dropout_list = self._get_dropout_list(hidden_dim, p)
        self.idx_list = []
        for i in range(9):
            li = range(len(self.dropout_list))
            random.shuffle(li)
            self.idx_list.append(li)

        self.y, self.y_test, self.l2_loss = self._build_network()
        self.y_ = tf.placeholder(tf.float32, [None,10])
        cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y))
        self.loss = cross_entropy + 0.00001*self.l2_loss
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        correct_prediction = tf.equal(tf.argmax(self.y_test,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        self._init()

    def _get_dropout_list(self, hidden_dim, keep_prob):
        x = self.sim.get_random_vector(hidden_dim, keep_prob)
        length = x.sum()
        res = []
        for i in range(len(x)):
            if(length > 1000):
                res += int(x[i]*1000.0/length)*[i+1]
            else:
                res += x[i]*[i+1]
        return res

    def _build_network(self):
        hidden_dim = self.hidden_dim
        #self.A = tf.constant(1.7159)
        #self.B = tf.constant(0.6666)
        x = pseudo_dropout(self.x, self.x_dropout_list[self.idxx], 784)
        x_test = self.p * self.x
        #x = tf.nn.dropout(self.x, self.keep_prob)
        W = tf.Variable(_init_matrix([784, hidden_dim]))
        b = tf.Variable(_init_matrix([hidden_dim]))
        loss = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
        hidden0 = tf.matmul(x, W) + b
        hidden0_test = tf.matmul(x_test, W) + b
        hidden = pseudo_dropout(hidden0, self.dropout_list[self.idx_list[0][self.idx]], self.hidden_dim)
        hidden_test = self.p * hidden0_test
        act = tf.nn.sigmoid(hidden)
        act_test = tf.nn.sigmoid(hidden_test)
        for i in range(1, 9):
            W = tf.Variable(_init_matrix([hidden_dim, hidden_dim]))
            b = tf.Variable(_init_matrix([hidden_dim]))
            loss = loss + tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
            hidden0 = tf.matmul(act, W) + b
            hidden0_test = tf.matmul(act_test, W) + b
            hidden = pseudo_dropout(hidden0, self.dropout_list[self.idx_list[i][self.idx]], self.hidden_dim)
            hidden_test = self.p * hidden0_test
            act = tf.nn.sigmoid(hidden)
            act_test = tf.nn.sigmoid(hidden_test)
        W = tf.Variable(_init_matrix([hidden_dim, 10]))
        b = tf.Variable(_init_matrix([10]))
        loss = loss + tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
        y = tf.nn.softmax(tf.matmul(act,W) + b)
        y_test = tf.nn.softmax(tf.matmul(act_test, W) + b)
        return y, y_test, loss

    def _init(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _train_epoch(self):
        for i in range(1000):
            batch_xs, batch_ys = self.mnist.train.next_batch(50)
            self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})
            self.idx = (self.idx + 1)%len(self.dropout_list)
            self.idxx += (self.idxx + 1)%len(self.x_dropout_list)
        validation_x, validation_y = self.mnist.train.next_batch(5000)
        validation_accu = self.sess.run(self.accuracy,
            feed_dict={self.x: validation_x, self.y_: validation_y})
        return validation_accu

    def _test(self):
        return self.sess.run(self.accuracy, feed_dict={self.x: self.mnist.test.images, self.y_: self.mnist.test.labels})

def _init_matrix(shape):
    return tf.truncated_normal(shape, stddev = 0.05)

def pseudo_dropout(x, period, dim):
    w = np.zeros([dim])
    idx0 = np.random.randint(period)
    while idx0 < dim:
        w[idx0] = 1
        idx0 += period
    return x*w

if __name__=="__main__":
    sudodnn = sudoDNN(1000, 0.75)
    file = open("sudod 0.75", "w")
    for epoch in range(300):
        tra = sudodnn._train_epoch()
        tea = sudodnn._test()
        print epoch, tra, tea
        file.write("{0}\t{1}\t{2}\n".format(epoch, tra, tea))
    file.close()
