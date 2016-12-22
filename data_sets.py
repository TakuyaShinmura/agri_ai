# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import numpy as np
from six.moves import xrange
import random


VALIDATION_START = 10
VALIDATTION_END = 15
NUM_INPUT = 2
NUM_PREDICT = 2

class DataSets(object):

    '''
    @param data_dir: Path of datasets csv file. 
    '''

    def __init__(self,data_dir):

        all_data=[]
        train_index = []
        validation_index = []

        with open(data_dir, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            num_line = 0
            for row in reader:
                float_row = row[4:-1]
                float_row = [float(elem) for elem in float_row]
                all_data.append(float_row)

                if row[12] == str(1):
                    if int(row[2]) >= VALIDATION_START and int(row[2]) <= VALIDATTION_END:
                        validation_index.append(num_line)
                    else:
                        train_index.append(num_line)
                num_line += 1

        self.num_line = num_line + 1

        self.all_data = np.array(all_data)
        self.train_index = train_index
        self.validation_index = validation_index

        self.average = np.mean(self.all_data, axis=0, keepdims=True)
        self.std = np.std(self.all_data, axis=0, keepdims=True)
        self.max = np.max(self.all_data, axis=0, keepdims=True)
        self.min = np.min(self.all_data, axis=0, keepdims=True)

        self.stand_data = (self.all_data - self.average)/self.std
        self.validation_data = self._create_validation_data()

    def _create_validation_data(self):

        input_batch = []
        correct_batch = []

        for index in self.validation_index:
            time_step = []
            for i in xrange(NUM_INPUT):
                time_step.append(self.stand_data[index+1])
            input_batch.append(time_step)

        for index in self.validation_index:
            time_step = []
            for i in xrange(NUM_PREDICT):
                time_step.append(self.stand_data[index+NUM_INPUT+i])
            correct_batch.append(time_step)

        return (input_batch, correct_batch)            

    '''
    @param batch_size: Num of batch size.
    '''
    def get_next_batch(self, batch_size):

        target = random.sample(self.train_index, batch_size)
        input_batch = []
        correct_batch = []
        for index in target:
            time_step = []
            for i in xrange(NUM_INPUT):
                time_step.append(self.stand_data[index+i])
            input_batch.append(time_step)

        for index in target:
            time_step = []
            for i in xrange(NUM_PREDICT):
                time_step.append(self.stand_data[index+NUM_INPUT+i])
            correct_batch.append(time_step)


        input_batch = np.array(input_batch)
        correct_batch = np.array(correct_batch)

        return input_batch, correct_batch






        