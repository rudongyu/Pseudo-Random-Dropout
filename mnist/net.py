import tensorflow as tf
import numpy as np
import input_data

class DNN(object):

    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.keep_prob = tf.placeholder(tf.float32)
        self.y, self.l2_loss = self._build_network()
        self.y_ = tf.placeholder(tf.float32, [None,10])
        cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y))
        self.loss = cross_entropy + 0.00001*self.l2_loss
        #self.loss = cross_entropy + 0.01*self.l2_loss
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        #self.train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(self.loss)
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        self._init()

    def _build_network(self):
        hidden_dim = self.hidden_dim
        #self.A = tf.constant(1.7159)
        #self.B = tf.constant(0.6666)
        x = tf.nn.dropout(self.x, self.keep_prob)
        W = tf.Variable(_init_matrix([784, hidden_dim]))
        b = tf.Variable(_init_matrix([hidden_dim]))
        loss = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
        hidden0 = tf.matmul(x, W) + b
        hidden = tf.nn.dropout(hidden0, self.keep_prob)
        act = tf.nn.sigmoid(hidden)
        for i in range(1, 9):
            W = tf.Variable(_init_matrix([hidden_dim, hidden_dim]))
            b = tf.Variable(_init_matrix([hidden_dim]))
            loss = loss + tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
            hidden0 = tf.matmul(act, W) + b
            hidden = tf.nn.dropout(hidden0, self.keep_prob)
            act = tf.nn.sigmoid(hidden)
        W = tf.Variable(_init_matrix([hidden_dim, 10]))
        b = tf.Variable(_init_matrix([10]))
        loss = loss + tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
        y = tf.nn.softmax(tf.matmul(hidden,W) + b)
        return y, loss

    def _init(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _train_epoch(self):
        for i in range(1000):
            batch_xs, batch_ys = self.mnist.train.next_batch(50)
            self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys, self.keep_prob: 0.7})
        validation_x, validation_y = self.mnist.train.next_batch(5000)
        validation_accu = self.sess.run(self.accuracy,
            feed_dict={self.x: validation_x, self.y_: validation_y, self.keep_prob: 1.0})
        print "validation accuracy ", validation_accu
        return validation_accu

    def _test(self):
        return self.sess.run(self.accuracy, feed_dict={self.x: self.mnist.test.images, self.y_: self.mnist.test.labels, self.keep_prob:1.0})

def _init_matrix(shape):
    return tf.truncated_normal(shape, stddev = 0.05)

if __name__ == '__main__':
    nn = DNN(1000)
    file = open("dnn 1.0", "w")
    for i in range(200):
        print "epoch ", i
        tr = nn._train_epoch()
        te = nn._test()
        print i, tr, te
        file.write("{0}\t{1}\t{2}\n".format(i, tr, te))
    file.close()
