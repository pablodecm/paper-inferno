from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json

import tensorflow as tf


class MixtureBatcher(object):

  def __init__(self, c_names, name="batcher",
               buffer_size=100000):

    self.batch_size = tf.placeholder(dtype=tf.int64, shape=(),
                                     name=f"batch_size_{name}")
    self.buffer_size = buffer_size
    self.seed = tf.placeholder(dtype=tf.int64, shape=(),
                               name=f"seed_{name}")
    self.name = name
    self.c_names = c_names

    self.tensors = {}
    components = {}
    for c_name in c_names:
      self.tensors[c_name] = tf.placeholder(dtype=tf.float32,
                                            name=f"{c_name}_{self.name}_ph",
                                            shape=(None, None))
      dat = tf.data.Dataset.from_tensor_slices(self.tensors[c_name])
      dat = dat.shuffle(buffer_size=self.buffer_size, seed=self.seed)
      components[c_name] = dat.batch(self.batch_size)

    self.dataset = tf.data.Dataset.zip(components)

    self.iterator = self.dataset.make_initializable_iterator()
    self.batch = self.iterator.get_next()

  def init_iterator(self, dict_arr, batch_size, seed):

    sess = tf.get_default_session()
    feed_dict = {}
    if batch_size == -1:
      batch_size = np.max([arr.shape[0] for arr in dict_arr.values()])
    feed_dict[self.batch_size] = batch_size
    for c_name, c_arr in dict_arr.items():
      feed_dict[self.tensors[c_name]] = c_arr
    feed_dict[self.seed] = seed

    sess.run(self.iterator.initializer, feed_dict)



def filter_by_column(column_name, value):
  """ Returns a filter function for an Ordered Dict,
  filterting only the elements whose column_name is
  equal to value """

  if type(value) == bytes:
    def filter_func(input_dict):
      return tf.equal(input_dict[column_name][0],
             tf.convert_to_tensor(str_value.encode()))
  else:
   def filter_func(input_dict):
      return tf.equal(input_dict[column_name][0],
             tf.convert_to_tensor(value))

  return filter_func

class HiggsBatcher(object):
  """ Creates a batch tensor for the Higgs ML dataset, using
  TensorFlow dataset pipelines.
  """
  def __init__(self, c_names = ["s","b"],
               name="batcher",
               buffer_size=100000,
               higgs_data_path="../data/atlas-higgs-challenge-2014-v2.csv"):

    self.batch_size = tf.placeholder(dtype=tf.int64, shape=(),
                                     name=f"batch_size_{name}")
    self.buffer_size = buffer_size
    self.seed = tf.placeholder(dtype=tf.int64, shape=(),
                               name=f"seed_{name}")
    self.name = name
    self.c_names = c_names

    self.set_name = tf.placeholder(dtype=tf.string, shape=(),
                                   name=f"set_name_{name}")

    csv_dataset = tf.contrib.data.make_csv_dataset(higgs_data_path, batch_size=1,
                                                   shuffle=False, num_epochs=1)

    sub_dataset = csv_dataset.filter(filter_by_column("KaggleSet",
                                                      self.set_name))

    components = {}
    for c_name in c_names:
      dat = sub_dataset.filter(filter_by_column("Label", c_name))
      dat = dat.shuffle(buffer_size=self.buffer_size, seed=self.seed)
      components[c_name] = dat.batch(self.batch_size)

    self.dataset = tf.data.Dataset.zip(components)

    self.iterator = self.dataset.make_initializable_iterator()
    self.batch = self.iterator.get_next()

  def init_iterator(self, set_name, batch_size, seed):

    sess = tf.get_default_session()
    feed_dict = {}
    if batch_size == -1:
      batch_size = np.max([arr.shape[0] for arr in dict_arr.values()])
    feed_dict[self.batch_size] = batch_size
    if type(set_name) == bytes:
      feed_dict[self.set_name] = set_name
    else:
      feed_dict[self.set_name] = set_name.encode()
    feed_dict[self.seed] = seed

    sess.run(self.iterator.initializer, feed_dict)

def softmax(x, axis=None):
  """ Simple Numpy softmax implementation"""
  e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
  return e_x / np.sum(e_x, axis=axis, keepdims=True)


class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    elif isinstance(obj, np.floating):
      return float(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    else:
      return super(NumpyEncoder, self).default(obj)
