#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data_sets import DataSets
from six.moves import xrange
import tensorflow as tf
import random
import model
import numpy as np


tf.app.flags.DEFINE_string("train_dir","data/","Work directory.")
tf.app.flags.DEFINE_string("log_dir","data/logs/","Log directory.")
tf.app.flags.DEFINE_string("model_dir", "data/ckpt/", "Model directory.")
tf.app.flags.DEFINE_integer("num_units",128, "State size.")
tf.app.flags.DEFINE_integer("train_step", 1000, "Step size to train.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size.")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")



FLAGS = tf.app.flags.FLAGS
DATA_SETS_FILE = 'data.csv'
NUM_INPUT = 6
INPUT_STEP = 24
OUTPUT_STEP = 24

def main(_):
    data_sets = DataSets(FLAGS.train_dir+DATA_SETS_FILE)

    stdev = data_sets.std
    mean = data_sets.average

    print(mean)
    print(stdev)
    print(data_sets.max)
    print(data_sets.min)


    encoder_inputs = []
    for i in xrange(INPUT_STEP):
        encoder_input = tf.placeholder(tf.float32, [None, NUM_INPUT])
        encoder_inputs.append(encoder_input)
        tf.add_to_collection("encoder_input", encoder_input)

    decoder_inputs = []
    for i in xrange(OUTPUT_STEP):
        decoder_input = tf.placeholder(tf.float32, [None, NUM_INPUT])
        decoder_inputs.append(decoder_input)
        tf.add_to_collection("decoder_input", decoder_input)

    corrects = []
    for i in xrange(OUTPUT_STEP):
        correct = tf.placeholder(tf.float32, [None, NUM_INPUT])
        corrects.append(correct)
        tf.add_to_collection("correct", correct)

    outputs = model.inference(encoder_inputs, decoder_inputs, FLAGS.num_units)

    loss = model.loss(outputs, corrects)

    train_step = model.train_step(loss, FLAGS.learning_rate)

    average_error = model.average_error(outputs, corrects, stdev, mean)

    init_op = tf.global_variables_initializer()

    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:

    	file_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        valid_feed = {}

        v_inputs, v_corrects = data_sets.validation_data
        valid_size = v_inputs.shape[0]
        v_inputs = np.transpose(v_inputs,(1,0,2))
        v_corrects = np.transpose(v_corrects,(1,0,2))

        for i in xrange(INPUT_STEP):
            valid_feed[tf.get_collection("encoder_input")[i]] = v_inputs[i]


        for i in xrange(OUTPUT_STEP):
            valid_feed[tf.get_collection("correct")[i]] = v_corrects[i]
            if i != 0:
                valid_feed[tf.get_collection("decoder_input")[i]] = v_corrects[i-1]


            else:
                valid_feed[tf.get_collection("decoder_input")[i]] = np.zeros((valid_size, NUM_INPUT))

		saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt:
        	last_model = ckpt.model_checkpoint_path
        	print("load " + last_model)
         	saver.restore(sess, last_model)
        else:
        	sess.run(init_op)

        for step in xrange(FLAGS.train_step):

            train_feed = {}
            t_inputs, t_corrects = data_sets.get_next_batch(FLAGS.batch_size)
            t_inputs = np.transpose(t_inputs,(1,0,2))
            t_corrects = np.transpose(t_corrects,(1,0,2))

            for i in xrange(INPUT_STEP):
                train_feed[tf.get_collection("encoder_input")[i]] = t_inputs[i]

            for i in xrange(OUTPUT_STEP):
                train_feed[tf.get_collection("correct")[i]] = t_corrects[i]
                if i != 0:
                    train_feed[tf.get_collection("decoder_input")[i]] = t_corrects[i-1]
                else:
                    train_feed[tf.get_collection("decoder_input")[i]] = np.zeros((FLAGS.batch_size, NUM_INPUT))

            print("step%d"%step)
            sess.run(train_step, feed_dict=train_feed)

            if step % 100 == 0:
                summary, error, loss_val = sess.run([summary_op, average_error, loss], feed_dict=valid_feed)
                file_writer.add_summary(summary, step)
                print("step%d loss=%f average error: HD=%f TP=%f HM=%f SM=%f CO=%f SR=%f"%(step, loss_val, error[0], error[1], error[2], error[3], error[4] ,error[5]))
                saver.save(sess, "model.ckpt")


        



if __name__ == "__main__":
    tf.app.run()