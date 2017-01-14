# -*- coding:utf-8 -*-



import tensorflow as tf


from data_sets import DataSets
from six.moves import xrange
import time
import tensorflow as tf
import random
import model
import numpy as np
import csv





tf.app.flags.DEFINE_string("train_dir","data/","Work directory.")
tf.app.flags.DEFINE_string("log_dir","data/logs/","Log directory.")
tf.app.flags.DEFINE_string("model_dir", "data/ckpt/", "Model directory.")
tf.app.flags.DEFINE_integer("num_units",128, "State size.")
tf.app.flags.DEFINE_integer("train_step",1000, "Step size to train.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size.")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")



FLAGS = tf.app.flags.FLAGS
DATA_SETS_FILE = 'data.csv'
TEST_SETS_FILE = 'test_data.csv'
NUM_INPUT = 6
TIME_STEP = 24
LOG_STEP = 100


def main(_):
    data_sets = DataSets(FLAGS.train_dir+DATA_SETS_FILE, FLAGS.train_dir+TEST_SETS_FILE)

    stdev = data_sets.std
    mean = data_sets.average
    test_keys = data_sets.test_keys
    print(mean)
    print(stdev)
    print(data_sets.max)
    print(data_sets.min)

    print("create model...")
    encoder_inputs = []
    decoder_inputs = []
    corrects = []
    for i in xrange(TIME_STEP):
        encoder_input = tf.placeholder(tf.float32, [None, NUM_INPUT])
        decoder_input = tf.placeholder(tf.float32, [None, NUM_INPUT])
        correct = tf.placeholder(tf.float32, [None, NUM_INPUT])

        encoder_inputs.append(encoder_input)
        decoder_inputs.append(decoder_input)
        corrects.append(correct)

        tf.add_to_collection("encoder_input", encoder_input)
        tf.add_to_collection("decoder_input", decoder_input)
        tf.add_to_collection("correct", correct)

    outputs = model.inference(encoder_inputs, decoder_inputs, FLAGS.num_units)


    with tf.Session() as sess:

        test_feed = {}

        t_inputs, t_corrects = data_sets.test_data
        test_size = t_inputs.shape[0]
        t_inputs = np.transpose(t_inputs,(1,0,2))
        t_corrects = np.transpose(t_corrects,(1,0,2))

        for i in xrange(TIME_STEP):
            test_feed[tf.get_collection("encoder_input")[i]] = t_inputs[i]


        for i in xrange(TIME_STEP):
            test_feed[tf.get_collection("correct")[i]] = t_corrects[i]
            if i != 0:
                test_feed[tf.get_collection("decoder_input")[i]] = t_corrects[i-1]
            else:
                test_feed[tf.get_collection("decoder_input")[i]] = np.zeros((test_size, NUM_INPUT))

		saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)

        last_model = ckpt.model_checkpoint_path
        print("load " + last_model)
        saver.restore(sess, last_model)

        predicts = sess.run(outputs, feed_dict=test_feed)

        predicts = np.transpose(predicts, (1,0,2))
        result = []
        hd_mean = mean[0][0]
        hd_stdev= stdev[0][0]

        counter = 0
        for i in predicts:
        	hds =[]
        	for j in i:
        		hds.append(str(j[0]*hd_stdev + hd_mean))

        	hds.insert(0, test_keys[counter])
        	result.append(hds)
        	counter +=1

        with open('data/result.csv', 'w') as f:
        	writer = csv.writer(f, lineterminator='\n')
        	for i in result:
        		writer.writerow(i)

if __name__ == "__main__":
    tf.app.run()
