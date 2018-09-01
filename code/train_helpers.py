from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.model_selection import train_test_split as d_split
import numpy as np
import tensorflow as tf
import json


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

    data = np.load(self.file_name)

    train_data = {}
    val_data = {}
    for key, value in data.items():
      train_data[key], val_data[key] = d_split(value,
                                               random_state=self.random_state)

    self.train_data = {k: tf.convert_to_tensor(
        v) for k, v in train_data.items()}
    self.val_data = {k: tf.convert_to_tensor(v) for k, v in val_data.items()}

    components = {}
    for key, value in self.train_data.items():
      if key in self.keys:
        components[key] = tf.data.Dataset.from_tensor_slices(value)\
                                         .shuffle(buffer_size=10000)\
                                         .batch(self.batch_size)

    dataset = tf.data.Dataset.zip({"components": components})

    dataset_it = dataset.make_one_shot_iterator()
    next_batch = dataset_it.get_next()

    return next_batch, None

  def val_input_fn(self):

    data = np.load(self.file_name)

    train_data = {}
    val_data = {}
    for key, value in data.items():
      train_data[key], val_data[key] = d_split(value,
                                               random_state=self.random_state)

    self.train_data = {k: tf.convert_to_tensor(
        v) for k, v in train_data.items()}
    self.val_data = {k: tf.convert_to_tensor(v) for k, v in val_data.items()}

    components = {}
    for key, value in self.val_data.items():
      if key in self.keys:
        components[key] = tf.data.Dataset.from_tensor_slices(value)\
                                         .shuffle(buffer_size=10000)\
                                         .batch(self.val_batch_size)

    dataset = tf.data.Dataset.zip({"components": components})

    dataset_it = dataset.make_one_shot_iterator()
    next_batch = dataset_it.get_next()

    return next_batch, None
