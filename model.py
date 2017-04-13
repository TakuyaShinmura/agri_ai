# -*- coding:utf-8 -*-

import tensorflow as tf
import math

NUM_INPUT = 6


def loop_function(prev, num_units):

	with tf.variable_scope("output"):
		weight_init = tf.truncated_normal_initializer(stddev=1.0/math.sqrt(num_units))
		weight = tf.get_variable('output_weight', [num_units, NUM_INPUT], initializer = weight_init)

		bias_init = tf.constant_initializer(value=0.0)
		bias = tf.get_variable('output_bias', [NUM_INPUT], initializer=bias_init)

		out = tf.matmul(prev, weight) + bias

	return out





def _prediction_decoder(decoder_inputs, initial_state, cell, num_units, feed_previous, scope=None):

	with tf.variable_scope(scope or "prediction") as scope:

		state = initial_state
		outputs = []
		prev = None
		for i, inp in enumerate(decoder_inputs):
			if i > 0:
				scope.reuse_variables()
				if feed_previous:
					inp = prev
			out, state = cell(inp, state)
			pred = loop_function(out, num_units)
			outputs.append(pred)
			prev = pred
		return outputs, state


def inference(encoder_inputs, decoder_inputs, num_units, feed_previous=True):

	encoder_cell = tf.contrib.rnn.LSTMCell(num_units=num_units, use_peepholes=True)
	encoder_cell = tf.contrib.rnn.MultiRNNCell(cells=[encoder_cell]*3)

	_, state = tf.contrib.rnn.static_rnn(encoder_cell, encoder_inputs, dtype=tf.float32)

	

	decoder_cell = tf.contrib.rnn.LSTMCell(num_units=num_units, use_peepholes=True)
	decoder_cell = tf.contrib.rnn.MultiRNNCell(cells=[decoder_cell]*3)


	outputs , _ = _prediction_decoder(decoder_inputs, state, decoder_cell, num_units=num_units, feed_previous=feed_previous)

	return outputs

def loss(outputs, corrects):
	
	with tf.name_scope("loss"): 

		loss_array = []
		for output, correct in zip(outputs, corrects):
			loss_array.append(tf.reduce_mean(tf.abs(output - correct)))

		loss = tf.add_n(loss_array)
	tf.summary.scalar("loss", loss)

	return loss

def train_step(loss, learning_rate):

	with tf.name_scope("train"):
		train = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
	return train


def evaluate_error(outputs, corrects, std, mean):

	maximum_error = None
	average_error = []
	counter = 0

	for output, correct in zip(outputs, corrects):
		true_out = output * std + mean
		true_correct = correct * std + mean
		abs_error = tf.abs(true_correct - true_out)
		max_error = tf.reduce_max(abs_error, axis=0)
		mean_error = tf.reduce_mean(abs_error, axis=0)
		average_error.append(mean_error)

		if counter == 0:
			maximum_error = max_error
		else:
			maximum_error = tf.maximum(max_error, maximum_error)

		counter +=1

	average_error = tf.add_n(average_error)/counter

	tf.summary.scalar("MAX HD Error", maximum_error[0])
	tf.summary.scalar("MAX TP Error", maximum_error[1])
	tf.summary.scalar("MAX HM Error", maximum_error[2])
	tf.summary.scalar("MAX SM Error", maximum_error[3])
	tf.summary.scalar("MAX CO Error", maximum_error[4])
	tf.summary.scalar("MAX SR Error", maximum_error[5])

	tf.summary.scalar("HD Error", average_error[0])
	tf.summary.scalar("TP Error", average_error[1])
	tf.summary.scalar("HM Error", average_error[2])
	tf.summary.scalar("SM Error", average_error[3])
	tf.summary.scalar("CO Error", average_error[4])
	tf.summary.scalar("SR Error", average_error[5])

	return maximum_error, average_error





