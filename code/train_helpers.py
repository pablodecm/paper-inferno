from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from memory_profiler import profile


class InputFactory(object): 

  def __init__(self, file_name,
               batch_size, keys,
               val_batch_size=10000,
               random_state=17):

    self.file_name = file_name
    self.random_state = random_state 

    self.batch_size = batch_size
    self.val_batch_size = val_batch_size
    self.keys = keys


  def train_input_fn(self):


    data =  np.load(self.file_name)

    train_data = {}
    val_data = {}
    for key, value in data.items():
      train_data[key], val_data[key] = train_test_split(value,
          random_state=self.random_state) 

    self.train_data = {k : tf.convert_to_tensor(v) for k,v in train_data.items()}
    self.val_data = {k : tf.convert_to_tensor(v) for k,v in val_data.items()}


    components = {}
    for key, value in self.train_data.items():
      if key in self.keys:
        components[key] = tf.data.Dataset.from_tensor_slices(value)\
                                         .shuffle(buffer_size=10000)\
                                         .batch(self.batch_size)
  
    dataset = tf.data.Dataset.zip({"components" : components})
  
    dataset_it = dataset.make_one_shot_iterator()
    next_batch = dataset_it.get_next()

  
    return next_batch, None 

  def val_input_fn(self):

    data =  np.load(self.file_name)

    train_data = {}
    val_data = {}
    for key, value in data.items():
      train_data[key], val_data[key] = train_test_split(value,
          random_state=self.random_state) 

    self.train_data = {k : tf.convert_to_tensor(v) for k,v in train_data.items()}
    self.val_data = {k : tf.convert_to_tensor(v) for k,v in val_data.items()}


    components = {}
    for key, value in self.val_data.items():
      if key in self.keys:
        components[key] = tf.data.Dataset.from_tensor_slices(value)\
                                         .shuffle(buffer_size=10000)\
                                         .batch(self.val_batch_size)
  
    dataset = tf.data.Dataset.zip({"components" : components})
  
    dataset_it = dataset.make_one_shot_iterator()
    next_batch = dataset_it.get_next()
  
    return next_batch, None 

