# -*- coding:utf-8 -*-

from data_set import DataSets
import model
import time
import tensorflow as tf
import random
import numpy as np

tf.app.flags.DEFINE_string("train_dir","data/","Work directory.")
tf.app.flags.DEFINE_string("log_dir","data/logs/","Log directory.")
tf.app.flags.DEFINE_string("model_dir", "data/ckpt/", "Model directory.")
tf.app.flags.DEFINE_boolean("attention", False, "Use attention or not.")
tf.app.flags.DEFINE_integer("num_units",128, "State size.")
tf.app.flags.DEFINE_integer("train_step", 1000, "Step size to train.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size.")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")



FLAGS = tf.app.flags.FLAGS
DATA_SETS_FILE = 'train.csv'
TEST_SETS_FILE = 'test.csv'

TIME_STEP = 24
NUM_INPUT = 6
LOG_STEP = 100

def main(_):

	data_sets = DataSets(FLAGS.train_dir+DATA_SETS_FILE, FLAGS.train_dir+TEST_SETS_FILE)

	std = data_sets.std
	mean = data_sets.average

	print("create model...")

	encoder_inputs = []
	decoder_inputs = []
	corrects = []

	for i in range(TIME_STEP):
		#各時間用のplaceholderを定義
		encoder_input = tf.placeholder(tf.float32, [None, NUM_INPUT])
		decoder_input = tf.placeholder(tf.float32, [None, NUM_INPUT])
		correct = tf.placeholder(tf.float32, [None, NUM_INPUT])
		#tf.contrib.static_rnnに渡せるようにlistに突っ込む
		encoder_inputs.append(encoder_input)
		decoder_inputs.append(decoder_input)
		corrects.append(correct)
		#実行時に各placeholderを参照するためにcollectionに突っ込む
		tf.add_to_collection("encoder_input", encoder_input)
		tf.add_to_collection("decoder_input", decoder_input)
		tf.add_to_collection("correct", correct)

	outputs = model.inference(encoder_inputs, decoder_inputs, FLAGS.num_units)

	loss = model.loss(outputs, corrects)

	train_step = model.train_step(loss, FLAGS.learning_rate)

	maximum_error, average_error = model.evaluate_error(outputs, corrects, std, mean)

	init_op = tf.global_variables_initializer()

	summary_op = tf.summary.merge_all()

	with tf.Session() as sess:

		file_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

		valid_feed = {}
		test_feed = {}

		v_inputs, v_corrects = data_sets.validation_data
		t_inputs, t_corrects = data_sets.test_data

		valid_size = v_inputs.shape[0]
		test_size = t_inputs.shape[0]

		v_inputs = np.transpose(v_inputs,(1,0,2))
		t_inputs = np.transpose(t_inputs,(1,0,2))
		v_corrects = np.transpose(v_corrects,(1,0,2))
		t_corrects = np.transpose(t_corrects,(1,0,2))

		for i in range(TIME_STEP):
			valid_feed[tf.get_collection("encoder_input")[i]] = v_inputs[i]
			test_feed[tf.get_collection("encoder_input")[i]] = t_inputs[i]
			valid_feed[tf.get_collection("correct")[i]] = v_corrects[i]
			test_feed[tf.get_collection("correct")[i]] = t_corrects[i]

			if i != 0:
		  		valid_feed[tf.get_collection("decoder_input")[i]] = v_corrects[i-1]
		  		test_feed[tf.get_collection("decoder_input")[i]] = t_corrects[i-1]
			else:
				valid_feed[tf.get_collection("decoder_input")[i]] = np.zeros((valid_size, NUM_INPUT))
				test_feed[tf.get_collection("decoder_input")[i]] = np.zeros((test_size, NUM_INPUT))

		saver = tf.train.Saver()
		ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
		if ckpt:
			last_model = ckpt.model_checkpoint_path
			print("load " + last_model)
			saver.restore(sess, last_model)
		else:
			print("no ckpt file. initialize variables.")
			sess.run(init_op)

		print("start training.")
		start_time = time.time()
		for step in range(FLAGS.train_step + 1):

			train_feed = {}
			t_inputs, t_corrects = data_sets.get_next_batch(FLAGS.batch_size)
			t_inputs = np.transpose(t_inputs,(1,0,2))
			t_corrects = np.transpose(t_corrects,(1,0,2))

			for i in range(TIME_STEP):
				train_feed[tf.get_collection("encoder_input")[i]] = t_inputs[i]
				train_feed[tf.get_collection("correct")[i]] = t_corrects[i]
				if i != 0:
					train_feed[tf.get_collection("decoder_input")[i]] = t_corrects[i-1]
				else:
					train_feed[tf.get_collection("decoder_input")[i]] = np.zeros((FLAGS.batch_size, NUM_INPUT))

			
			if step % LOG_STEP == 99:
				run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				run_metadata = tf.RunMetadata()
				summary, _ = sess.run([summary_op, train_step], feed_dict=train_feed, options=run_options, run_metadata=run_metadata)
				file_writer.add_run_metadata(run_metadata, 'step%d' % step)
			else:
				sess.run(train_step, feed_dict=train_feed)

			if step > 0 and step % LOG_STEP == 0:
				duration = time.time() - start_time
				print("step%d-%d: %f sec"%(step-LOG_STEP, step, duration))

				summary, loss_val, average, maximum = sess.run([summary_op,loss, average_error, maximum_error], feed_dict=valid_feed)

				file_writer.add_summary(summary, step)
				print("step%d loss=%f average error: HD=%f TP=%f HM=%f SM=%f CO=%f SR=%f"%(step, loss_val, average[0], average[1], average[2], average[3], average[4] ,average[5]))
				print("max error: HD=%f TP=%f HM=%f SM=%f CO=%f SR=%f"%(maximum[0], maximum[1], maximum[2], maximum[3], maximum[4] , maximum[5]))
				start_time = time.time()
			

		saver.save(sess, FLAGS.model_dir+"model.ckpt")
		test_loss, test_average, test_maximum = sess.run([loss, average_error, maximum_error], feed_dict=test_feed)
		print("test loss=%f average error: HD=%f TP=%f HM=%f SM=%f CO=%f SR=%f"%(test_loss, test_average[0], test_average[1], test_average[2], test_average[3], test_average[4] ,test_average[5]))
		print("max error: HD=%f TP=%f HM=%f SM=%f CO=%f SR=%f"%(test_maximum[0], test_maximum[1], test_maximum[2], test_maximum[3], test_maximum[4] , test_maximum[5]))


if __name__ == '__main__':
	tf.app.run()

