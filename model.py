# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
from six.moves import xrange


def loop_function(prev, num_units):

    with tf.variable_scope("output"):
        weight_init = tf.truncated_normal_initializer(stddev=1.0/math.sqrt(num_units))
        weight = tf.get_variable('output_weight', [num_units, 6], initializer = weight_init)

        bias_init = tf.constant_initializer(value=0.0)
        bias = tf.get_variable('output_bias', [6], initializer=bias_init)

        out = tf.matmul(prev, weight) + bias
    return out



def _prediction_decoder(decoder_inputs, initial_state, cell, num_units, feed_previous=False ,scope=None):

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




def inference(encoder_inputs, decoder_inputs, num_units, feed_previous=False):

    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
    cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell]*5)
    _, state = tf.nn.rnn(cell=cell, inputs=encoder_inputs, dtype=tf.float32)

    outputs, state = _prediction_decoder(decoder_inputs, state,cell, num_units=num_units, feed_previous=feed_previous)
    return outputs

def loss(outputs, corrects):

    loss_array = []
    for output, correct in zip(outputs, corrects):
        loss_array.append(tf.reduce_mean(tf.abs(output-correct)))

    loss = tf.add_n(loss_array)
    tf.summary.scalar("loss", loss)

    return loss

def train_step(loss, learning_rate):
    return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

def average_error(outputs, corrects, stddev, mean):

    error = []
    counter = 0
    for output, correct in zip(outputs, corrects):
        true_out = output * stddev + mean
        true_correct = correct * stddev + mean

        accuracy = tf.abs(true_correct - true_out)
        accuracy = tf.reduce_mean(accuracy, axis=0)

        error.append(accuracy)
        counter +=1

    each_error = tf.add_n(error)/counter

    tf.summary.scalar("HD Error", each_error[0])
    tf.summary.scalar("TP Error", each_error[1])
    tf.summary.scalar("HM Error", each_error[2])
    tf.summary.scalar("SM Error", each_error[3])
    tf.summary.scalar("CO Error", each_error[4])
    tf.summary.scalar("SR Error", each_error[5])

    return each_error
