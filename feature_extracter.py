# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import pickle
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import os
from preprocess import *
#from resize import Preprocess
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
model = 'model_ae_comp.ckpt'
p = Preprocess('images', (60, 20))
p.preprocess()
x,y = p.generateTrainingSamples(p.image1DArray, p.labels)
input = p.image1DArray
# Parameters
learning_rate = 0.01
training_epochs = 2
batch_size = 20
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 512 # 1st layer num features
n_hidden_2 = 256 # 2nd layer num features
#n_input = shape[0]*shape[1] # MNIST data input (img shape: 28*28)
n_input = 60*20*2 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
	# Encoder Hidden layer with sigmoid activation #1
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
				   biases['encoder_b1']))
	# Decoder Hidden layer with sigmoid activation #2
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
				   biases['encoder_b2']))
	return layer_2


# Building the decoder
def decoder(x):
	# Encoder Hidden layer with sigmoid activation #1
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
				   biases['decoder_b1']))
	# Decoder Hidden layer with sigmoid activation #2
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
				   biases['decoder_b2']))
	return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()
saver = tf.train.Saver()
# Launch the graph
with tf.Session() as sess:
	sess.run(init)
	saver.restore(sess, model)
	print('Starting encode')
	encode = sess.run( encoder_op, feed_dict={X: x})
	print('Done')
	hickle.dump(encode, 'encode_ae_comp.hk1', mode='w', compression='gzip')


