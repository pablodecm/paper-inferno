
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from higgs_example import HiggsExample
from train_helpers import softmax, HiggsBatcher

import numpy as np
import tensorflow as tf
import itertools as it
import json

k = tf.keras

default_par_scan = {"tau_energy": [1.0, 1.03, 0.97]}
default_bins = np.linspace(0, 1, 11)


class HiggsSummaryStatisticComputer(object):
  """ Helper class that computes shape-like summary statistics for different
  parameter variations and summary statistics types: optimal classifier,
  trained classifer or inferno model."""

  def __init__(self, pars_scan=default_par_scan,
               set_name="b"):

    self.problem = HiggsExample()
    self.batcher = HiggsBatcher(features=self.problem.features,
                                buffer_size = 10000)
    
    n_features = len(self.problem.features)
    self.scale_means = tf.placeholder(dtype=tf.float32, shape=(n_features))
    self.scale_stds = tf.placeholder(dtype=tf.float32, shape=(n_features))


    self.pars_scan = {self.problem.all_pars[k]: v for k, v in pars_scan.items()}

    _ , mean_and_std = self.batcher.kaggle_sets("t")
    self.dict_arr, _ = self.batcher.kaggle_sets(set_name)


    self.phs_scale = {self.scale_means : mean_and_std[0],
                      self.scale_stds :  mean_and_std[1]}

    self.b_sizes = {}
    self.dense_batches = {}
    self.scaled_dense_batches = {}
    self.weights = {}
    for c_name in self.batcher.c_names:
      transform_batch = self.problem.transform(self.batcher.batch[c_name])
      dense_batch = self.problem.make_dense(transform_batch)
      scaled_dense_batch = (dense_batch - self.scale_means)/self.scale_stds
      weight =  self.problem.get_weight(transform_batch, c_name)
      self.b_sizes[c_name] = tf.shape(dense_batch)[0]
      self.dense_batches[c_name] = dense_batch
      self.scaled_dense_batches[c_name] = scaled_dense_batch
      self.weights[c_name] = weight

  def classifier_shapes(self, model_path, batch_size=10000,
                        bins=default_bins, sess=None):

    shapes = {}
    outputs = {}
    batch_size = 10000

    with open('{}/model.json'.format(model_path)) as f:
      model_json = json.load(f)
    model = k.models.model_from_json(model_json)

    if sess is None:
       sess = tf.Session()
    with sess.as_default():
      k.backend.set_session(sess)
      model.load_weights('{}/model.h5'.format(model_path))
      
      for c_name in self.batcher.c_names:
        for p_n, pars_val in enumerate(it.product(*self.pars_scan.values())):

          self.batcher.init_iterator(dict_arr=self.dict_arr,
                                          batch_size=batch_size, seed=20)

          pars_phs = {par: val for par, val in zip(self.pars_scan.keys(),
                                                   pars_val)}
          dense_x_arr, weight_arr = sess.run([self.scaled_dense_batches[c_name],
                                              self.weights[c_name]],
                                             {**pars_phs, **self.phs_scale})
          c_clf = softmax(model.predict(dense_x_arr,
                                      batch_size=batch_size),
                        axis=1)[:, 1]
          if p_n==0:
            outputs[c_name] = c_clf
            

          key = (c_name,) + pars_val
          c_clf_hist = np.histogram(c_clf,weights=weight_arr[:,0], bins=bins)[0]
          shapes[key] = c_clf_hist

      k.backend.set_session(None)
      

    return shapes, outputs

  def inferno_shapes(self, model_path, batch_size = 10000, sess=None):

    shapes = {}

    with open('{}/model.json'.format(model_path)) as f:
      model_json = json.load(f)
    model = k.models.model_from_json(model_json)
    n_outputs = model.get_output_shape_at(0)[1]

    if sess is None:
       sess = tf.Session()
    with sess.as_default():
      k.backend.set_session(sess)
      
      model.load_weights('{}/model.h5'.format(model_path))
      
      for c_name in self.batcher.c_names:
        for pars_val in it.product(*self.pars_scan.values()):

          self.batcher.init_iterator(dict_arr=self.dict_arr,
                                     batch_size=batch_size, seed=20)

          pars_phs = {par: val for par, val in zip(self.pars_scan.keys(),
                                                   pars_val)}
          dense_x_arr, weight_arr = sess.run([self.scaled_dense_batches[c_name],
                                              self.weights[c_name]],
                                             {**pars_phs, **self.phs_scale})
          c_clf = model.predict(dense_x_arr, batch_size=batch_size)
          key = (c_name,) + pars_val
          c_clf_hist = np.bincount(np.argmax(c_clf, axis=-1),
                                   weights=weight_arr[:,0],
                                   minlength=n_outputs)
          shapes[key] = c_clf_hist

      k.backend.set_session(None)

      return shapes
