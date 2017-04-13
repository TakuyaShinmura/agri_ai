# -*- coding: utf-8 -*-

import csv
import numpy as np
import random


VALIDATION_STSRT=10
VALIDATION_END=13
TIME_STEP=24

class DataSets(object):

	def __init__(self, train_path, test_path):

		all_data = []
		train_inded = []
		validation_index = []

		train_data, train_indexes, validation_indexes, _ = self._read_file(train_path)
		test_data, test_indexes, _, test_keys = self._read_file(test_path, False)

		self.train_data = np.array(train_data)
		self.test_data = np.array(test_data)
		self.train_indexes = train_indexes
		self.validation_indexes = validation_indexes
		self.test_indexes = test_indexes
		self.test_keys = test_keys
		print("train:%d, validation:%d, test:%d"%(len(self.train_indexes), len(self.validation_indexes), len(self.test_indexes)))

		self.average = np.mean(self.train_data, axis=0, keepdims=True)
		self.std = np.std(self.train_data, axis=0, keepdims=True)

		self.stand_train_data = (self.train_data - self.average)/self.std
		self.stand_test_data = (self.test_data - self.average)/self.std

		self.validation_data = self._create_batch(self.validation_indexes)
		self.test_data = self._create_batch(self.test_indexes, True)

	def _read_file(self, file_path, train=True):

		all_data = []
		keys = []
		input_indexes = []
		v_input_indexes = []

		with open(file_path, 'r') as f:
			reader = csv.reader(f)
			header = next(reader)

			num_line = 0
			for row in reader:
				key = row[0]+'/'+row[1]+'/'+row[2]+':'+row[3]
				str_row = row[5:]
				float_row = [float(elem) for elem in str_row]
				all_data.append(float_row)

				if row[4] == str(1):
					if train and int(row[2]) >= VALIDATION_STSRT and int(row[2]) <= VALIDATION_END:
						v_input_indexes.append(num_line)
					else:
						input_indexes.append(num_line)
						keys.append(key)
				num_line += 1
		return all_data, input_indexes, v_input_indexes, keys

	def _create_batch(self, indexes, test=False, batch_size=None):

		target = indexes
		if batch_size:
			target = random.sample(target, batch_size)

		input_batch = []
		correct_batch = []

		for index in target:
			input_time_step = []
			correct_time_step = []

			for i in range(TIME_STEP):
				if test:
					input_time_step.append(self.stand_test_data[index+i])
					correct_time_step.append(self.stand_test_data[index+TIME_STEP+i])
				else:
					input_time_step.append(self.stand_train_data[index+i])
					correct_time_step.append(self.stand_train_data[index+TIME_STEP+i])

			input_batch.append(input_time_step)
			correct_batch.append(correct_time_step)
		return (np.array(input_batch), np.array(correct_batch))

	def get_next_batch(self, batch_size):
		return self._create_batch(self.train_indexes, batch_size=batch_size)

