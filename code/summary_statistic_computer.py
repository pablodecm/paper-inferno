
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from synthetic_3D_example import SyntheticThreeDimExample
from train_helpers import softmax

import numpy as np
import tensorflow as tf
import itertools as it
import json

k = tf.keras

default_par_scan = {"r_dist": [2.0, 2.2, 1.8],
                    "b_rate": [3.0, 3.5, 2.5]}
default_bins = np.linspace(0, 1, 11)


class SummaryStatisticComputer(object):
  """ Helper class that computes shape-like summary statistics for different
  parameter variations and summary statistics types: optimal classifier,
  trained classifer or inferno model."""

  def __init__(self, pars_scan=default_par_scan,
               dataset="valid"):

    problem = SyntheticThreeDimExample()
    self.pars_scan = {problem.all_pars[k]: v for k, v in pars_scan.items()}

    with tf.Session() as sess:
      if "valid" in dataset:
        self.data = sess.run(problem.valid_data())
      elif "test" in dataset:
        self.data = sess.run(problem.test_data())
      else:
        print("dataset argument has to be valid or test")

    self.b_vals = tf.placeholder(dtype=tf.float32, shape=(None, 3),
                                 name="b_vals")
    self.b_vals_shifted = problem.transform_bkg(self.b_vals)
    self.x_vals = tf.placeholder(dtype=tf.float32, shape=(None, 3),
                                 name="x_vals")
    self.x_opt_clf = problem.optimal_classifier(self.x_vals)

  def optimal_shapes(self, bins=default_bins, sess=None):

    shapes = {}

    if sess is None:
       sess = tf.Session()
    with sess.as_default():
      s_clf = sess.run(self.x_opt_clf,
                       {self.x_vals: self.data["sig"]})
      shapes[("sig",)] = np.histogram(s_clf, bins)[0]
      for pars_val in it.product(*self.pars_scan.values()):
        pars_phs = {par: val for par, val in zip(self.pars_scan.keys(),
                                                 pars_val)}
        b_vals_arr = sess.run(self.b_vals_shifted,
                              {**pars_phs, **{self.b_vals: self.data["bkg"]}})
        b_clf = sess.run(self.x_opt_clf, {self.x_vals: b_vals_arr})
        key = ("bkg",) + pars_val
        b_clf_hist = np.histogram(b_clf, bins)[0]
        shapes[key] = b_clf_hist

    return shapes

  def classifier_shapes(self, model_path, bins=default_bins,
                        sess=None):

    shapes = {}

    batch_size = 10000

    with open(f'{model_path}/model.json') as f:
      model_json = json.load(f)
    model = k.models.model_from_json(model_json)

    if sess is None:
       sess = tf.Session()
    with sess.as_default():
      k.backend.set_session(sess)
      model.load_weights(f'{model_path}/model.h5')

      s_clf = softmax(model.predict(self.data["sig"],
                                    batch_size=batch_size),
                      axis=1)[:, 1]
      shapes[("sig",)] = np.histogram(s_clf, bins)[0]
      for pars_val in it.product(*self.pars_scan.values()):
        pars_phs = {par: val for par, val in zip(self.pars_scan.keys(),
                                                 pars_val)}
        b_vals_arr = sess.run(self.b_vals_shifted,
                              {**pars_phs, **{self.b_vals: self.data["bkg"]}})
        b_clf = softmax(model.predict(b_vals_arr,
                                      batch_size=batch_size),
                        axis=1)[:, 1]

        key = ("bkg",) + pars_val
        b_clf_hist = np.histogram(b_clf, bins)[0]
        shapes[key] = b_clf_hist

      k.backend.set_session(None)

    return shapes

  def generic_shapes(self, transformation_f, bins, sess=None):

    shapes = {}

    if sess is None:
       sess = tf.Session()
    with sess.as_default():
      k.backend.set_session(sess)
      model.load_weights(f'{model_path}/model.h5')

      s_clf = transformation_f(self.data["sig"])
      shapes[("sig",)] = np.histogram(s_clf, bins)[0]
      for pars_val in it.product(*self.pars_scan.values()):
        pars_phs = {par: val for par, val in zip(self.pars_scan.keys(),
                                                 pars_val)}
        b_vals_arr = sess.run(self.b_vals_shifted,
                              {**pars_phs, **{self.b_vals: self.data["bkg"]}})
        b_clf = transformation_f(b_vals_arr)
        key = ("bkg",) + pars_val
        b_clf_hist = np.histogram(b_clf, bins)[0]
        shapes[key] = b_clf_hist

      k.backend.set_session(None)

    return shapes

  def inferno_shapes(self, model_path, sess=None):

    shapes = {}

    batch_size = 10000

    with open(f'{model_path}/model.json') as f:
      model_json = json.load(f)
    model = k.models.model_from_json(model_json)
    n_outputs = model.get_output_shape_at(0)[1]

    if sess is None:
       sess = tf.Session()
    with sess.as_default():
      k.backend.set_session(sess)
      model.load_weights(f'{model_path}/model.h5')

      s_clf = model.predict(self.data["sig"], batch_size=batch_size)

      shapes[("sig",)] = np.bincount(np.argmax(s_clf, axis=-1),
                                     minlength=n_outputs)

      for pars_val in it.product(*self.pars_scan.values()):
        pars_phs = {par: val for par, val in zip(self.pars_scan.keys(),
                                                 pars_val)}
        b_vals_arr = sess.run(self.b_vals_shifted,
                              {**pars_phs, **{self.b_vals: self.data["bkg"]}})
        b_clf = model.predict(b_vals_arr, batch_size=batch_size)
        key = ("bkg",) + pars_val
        b_clf_hist = np.bincount(np.argmax(b_clf, axis=-1),
                                 minlength=n_outputs)
        shapes[key] = b_clf_hist

      k.backend.set_session(None)

      return shapes
