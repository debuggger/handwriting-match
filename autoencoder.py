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
import os
from preprocess import *

def autoEncoderTrain(model, data, shape):
	# Parameters
	learning_rate = 0.01
	training_epochs = 2
	batch_size = 20
	display_step = 1
	examples_to_show = 10

	n_hidden_1 = 512 # 1st layer num features
	n_hidden_2 = 256 # 2nd layer num features
	n_input = shape[0]*shape[1] # MNIST data input (img shape: 28*28)
	
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

	def encoder(x):
		# Encoder Hidden layer with sigmoid activation #1
		layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
									   biases['encoder_b1']))
		# Decoder Hidden layer with sigmoid activation #2
		layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
									   biases['encoder_b2']))
									   
		return layer_2

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
	total_batch = int(len(data)/batch_size)+1
	with tf.Session() as sess:
		# Training cycle
		sess.run(init)
		for epoch in range(training_epochs):
			start = 0
			# Loop over all batches
			for i in range(total_batch):
				batch_xs = np.array(data[start: start+batch_size])
				start += batch_size
				#print(i)
				# Run optimization op (backprop) and cost op (to get loss value)
				_, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
			# Display logs per epoch step
			if epoch % display_step == 0:
				print("Epoch:", '%04d' % (epoch+1),
					  "cost=", "{:.9f}".format(c))
		save_path = saver.save(sess, model)
	print("Optimization Finished!")

	
def autoEncode(model, data, shape):
	# Parameters
	learning_rate = 0.01
	training_epochs = 20
	batch_size = 20
	display_step = 1
	examples_to_show = 10

	n_hidden_1 = 256 # 1st layer num features
	n_hidden_2 = 128 # 2nd layer num features
	n_input = shape[0]*shape[1] # MNIST data input (img shape: 28*28)
	
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

	def encoder(x):
		# Encoder Hidden layer with sigmoid activation #1
		layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
									   biases['encoder_b1']))
		# Decoder Hidden layer with sigmoid activation #2
		layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
									   biases['encoder_b2']))
									   
		return layer_2

	# Construct model
	encoder_op = encoder(X)
	
	# Initializing the variables
	saver = tf.train.Saver()
	init = tf.initialize_all_variables()
	total_batch = int(len(data)/batch_size)+1
	with tf.Session() as sess:
		# Training cycle
		sess.run(init)
		saver.restore(sess, model)
		encode = sess.run( encoder_op, feed_dict={X: data})
	
	return encode

