from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import json

import tensorflow as tf


class MixtureBatcher(object):

  def __init__(self, c_names, name="batcher",
               buffer_size=100000):

    self.batch_size = tf.placeholder(dtype=tf.int64, shape=(),
                                     name="batch_size_{name}".format(name=name))
    self.buffer_size = buffer_size
    self.seed = tf.placeholder(dtype=tf.int64, shape=(),
                               name="seed_{name}".format(name=name))
    self.name = name
    self.c_names = c_names

    self.tensors = {}
    components = {}
    for c_name in c_names:
      self.tensors[c_name] = tf.placeholder(dtype=tf.float32,
                                            name="{c_name}_{name}_ph".format(name=name, c_name=c_name),
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
  """ Creates a batched tensor for the Higgs ML dataset, using
  TensorFlow dataset pipelines.
  """
  def __init__(self, features, c_names = ["s","b"],
               name="batcher",
               buffer_size=100000):

    self.features = features
    self.extra_columns =  ["Weight","PRI_jet_num"]

    self.batch_size = tf.placeholder(dtype=tf.int64, shape=(),
                                     name="batch_size_{name}".format(name=name))
    self.buffer_size = buffer_size
    self.seed = tf.placeholder(dtype=tf.int64, shape=(),
                               name="seed_{name}".format(name=name))
    self.name = name
    self.c_names = c_names

    self.tensors = {}
    components = {}
    for c_name in self.c_names:
      self.tensors[c_name] = {}
      for feature in (self.features+self.extra_columns):
        ph_name = name="{c_name}_{name}_ph".format(name=name, c_name=c_name)
        ph = tf.placeholder(dtype=tf.float32, name=ph_name,
                            shape=(None, None))
        self.tensors[c_name][feature] = ph

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
      batch_size = np.max([c_dict["Weight"].shape[0] for c_dict in dict_arr.values()])
    feed_dict[self.batch_size] = batch_size
    for c_name, c_dict in dict_arr.items():
      for feature, c_arr in c_dict.items():
        feed_dict[self.tensors[c_name][feature]] = c_arr
    feed_dict[self.seed] = seed

    sess.run(self.iterator.initializer, feed_dict)

  def kaggle_sets(self, set_name,
                  higgs_data_path="../data/atlas-higgs-challenge-2014-v2.csv"):

    df = pd.read_csv(higgs_data_path)

    s_df =  df.loc[(df.KaggleSet == set_name)]
    # remove -999.0 missing values by 0
    s_df = s_df.replace(-999.0,0.0)
    # compute mean and stds for set
    s_f_matrix = s_df.loc[:, self.features].values
    mean_and_std = (s_f_matrix.mean(axis=0),s_f_matrix.std(axis=0))

    dict_arr = {}
    for c_name in self.c_names:
      dict_arr[c_name] = {}
      c_df =  s_df.loc[(df.Label == c_name)]
      for feature in (self.features+self.extra_columns):
        dict_arr[c_name][feature] = c_df.loc[:,[feature]].values

    return dict_arr, mean_and_std


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
