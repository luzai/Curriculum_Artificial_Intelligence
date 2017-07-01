from config import Config
import tensorflow as tf
import numpy as np
import os, sys
from utils import SVHN
DATASET = 'svhn'
if DATASET == 'mnist':
    mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir='data/', one_hot=True)
else:
    mnist = SVHN()


def conv_layer(input, size_in, size_out, conv_size=5, name="conv", activation='relu'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([conv_size, conv_size, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        if activation.lower() == 'relu':
            act = tf.nn.relu(conv + b)
        elif activation.lower() == 'tanh':
            act = tf.nn.tanh(conv + b)
        elif activation.lower() == 'sigmoid':
            act = tf.nn.sigmoid(conv + b)
        else:
            act = conv + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(input, size_in, size_out, name="fc", activation='relu'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        if activation.lower == 'relu':

            act = tf.nn.relu(tf.matmul(input, w) + b)
        elif activation.lower() == 'tanh':
            act = tf.nn.tanh(tf.matmul(input, w) + b)
        else:
            act = tf.matmul(input, w) + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


class LeNet:
    def __init__(self, config):
        self.dataset = DATASET
        self.epochs = 1201
        self.config = config
        self.x = None
        self.y = None
        self.train_step = None
        self.keep_prob = None
        self.build()

    def build(self):
        tf_graph = tf.get_default_graph()
        _sess_config = tf.ConfigProto(allow_soft_placement=True)
        _sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=_sess_config, graph=tf_graph)
        # Setup placeholders, and reshape the data
        if self.dataset == 'mnist':
            self.x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
            x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        else:
            assert self.dataset == 'svhn'
            self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
            x_image = self.x

        tf.summary.image('input', x_image, 3)
        self.y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
        self.keep_prob = tf.placeholder(tf.float32)
        if self.dataset == 'mnist':
            conv, size_in = x_image, 1
        else:
            conv, size_in = x_image, 3

        for nb_conv_, conv_size_, nb_conv_chnl_ in zip(range(self.config.nb_pool), self.config.conv_size,
                                                       self.config.nb_conv_chnl):
            conv = conv_layer(conv, size_in, nb_conv_chnl_, conv_size_, 'conv' + str(nb_conv_))
            size_in = nb_conv_chnl_
            conv = tf.nn.dropout(conv, self.keep_prob)

        shape = np.array(conv.get_shape().as_list()[1:]).prod()
        fc = tf.reshape(conv, [-1, shape])
        size_in = shape
        for nb_fc_, fc_size_ in zip(range(self.config.nb_fc), self.config.fc_size):
            fc = fc_layer(fc, size_in, fc_size_, 'fc' + str(nb_fc_))
            size_in = fc_size_
            fc = tf.nn.dropout(fc, self.keep_prob)
        logits = fc_layer(fc, size_in, 10, 'logits', activation='None')

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=self.y), name="loss")
            summ_train_loss = tf.summary.scalar("train_loss", self.loss)
            summ_test_loss = tf.summary.scalar('test_loss', self.loss)

        with tf.name_scope("train"):
            # self.train_step = tf.train.GradientDescentOptimizer(self.config.lr).minimize(self.loss)
            self.train_step = tf.train.AdamOptimizer(self.config.lr).minimize(self.loss)

        with tf.name_scope("acc"):
            self.pred = tf.argmax(logits, 1)
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            summ_train_acc = tf.summary.scalar("train_acc", self.accuracy)
            summ_test_acc = tf.summary.scalar('test_acc', self.accuracy)

        # self.summ = tf.summary.merge_all()
        self.summ_train = tf.summary.merge([summ_train_acc, summ_train_loss])
        self.summ_test = tf.summary.merge([summ_test_acc, summ_test_loss])
        self.sess.run(tf.global_variables_initializer())

    def inference(self, x):
        [pred] = self.sess.run([self.pred],
                               feed_dict={self.x: x})
        return pred

    def train(self):
        train_writer = tf.summary.FileWriter('log/' + self.config.hparam_str + '/train/')
        # train_writer.add_graph(self.sess.graph)
        test_writer = tf.summary.FileWriter('log/' + self.config.hparam_str + '/test/')
        saver = tf.train.Saver()
        for i in range(self.epochs):  # 2001
            batch = mnist.train.next_batch(1024)
            if i % 10 == 0:
                [train_accuracy, s, loss] = self.sess.run([self.accuracy, self.summ_train, self.loss],
                                                          feed_dict={self.x: batch[0], self.y: batch[1],
                                                                     self.keep_prob: 1 - self.config.dp_ratio})
                train_writer.add_summary(s, i)
                # print '-> train loss is {} acc is {}'.format(loss, train_accuracy)
            if i % 3 == 0:
                [test_acc, s, loss] = self.sess.run([self.accuracy, self.summ_test, self.loss],
                                                    feed_dict={self.x: mnist.test.images,
                                                               self.y: mnist.test.labels,
                                                               self.keep_prob: 1})
                test_writer.add_summary(s, i)

                # saver.save(self.sess, 'log/model.ckpt', i)
                print "test loss {} acc {}".format(loss, test_acc)
            train_writer.flush(), test_writer.flush()
            self.sess.run(self.train_step,
                          feed_dict={self.x: batch[0], self.y: batch[1],
                                     self.keep_prob: 1 - self.config.dp_ratio})
