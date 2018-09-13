from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from neyman.inferences import batch_hessian
from ast import literal_eval
import json
import itertools as it
import tensorflow_probability as tfp
from collections import OrderedDict
from fisher_matrix import FisherMatrix

ds = tfp.distributions


def int_quad_lin(alpha, c_nom, c_up, c_dw, multiple_pars=False):
  "Three-point interpolation, quadratic inside and linear outside"

  if multiple_pars:
    tiling_shape = [1, 1, 1, tf.shape(c_nom)[0]]
    expand_axis = 1
  else:
    tiling_shape = [1, 1, tf.shape(c_nom)[0]]
    expand_axis = 0
  # alpha dimensions are (1, n_par_types, n_par_inst)
  # c_dw and c_up are (n_bins, n_par_types)
  alpha_t = tf.tile(tf.expand_dims(alpha, axis=-1), tiling_shape)
  # alpha_t dimensions are (1, n_par_types, n_par_inst, n_bins)
  # if multiple_pars is True or (1, n_par_types, n_bins)
  a = tf.expand_dims(0.5 * (c_up + c_dw) - c_nom,
                     axis=expand_axis, name="a")
  b = tf.expand_dims(0.5 * (c_up - c_dw),
                     axis=expand_axis, name="b")
  ones = tf.ones_like(alpha_t)
  #  (1, n_par_types, n_par_inst, n_bins) broadcast when multiple_pars
  #      (n_par_types,         1, n_bins) (if expand axis 1)
  #  (1, n_par_types, n_bins) broadcast when not multiple_pars
  #      (n_par_types,n_bins) (if expand axis 0)
  switch = tf.where(alpha_t < 0.,
                    ones * tf.expand_dims(c_dw - c_nom, axis=expand_axis),
                    ones * tf.expand_dims(c_up - c_nom, axis=expand_axis))
  abs_var = tf.where(tf.abs(alpha_t) > 1.,
                     (2 * b + tf.sign(alpha_t) * a) *
                     (alpha_t - tf.sign(alpha_t)) + switch,
                     a * tf.pow(alpha_t, 2) + b * alpha_t)
  # abs_var is (1, n_par_types, n_par_inst, n_bins) or (1, n_par_types, n_bins)
  return c_nom + tf.reduce_sum(abs_var, axis=1)


class TemplateModel(object):

  def __init__(self, multiple_pars=False):

    self.multiple_pars = multiple_pars
    if multiple_pars:
      shape_pars = (None,)
    else:
      shape_pars = ()

    def default_ph_value(val):
      if multiple_pars:
        return [val, ]
      else:
        return val

    self.r_dist_init = tf.placeholder_with_default(
        2., shape=(), name="r_dist_init")
    self.b_rate_init = tf.placeholder_with_default(
        3., shape=(), name="b_rate_init")
    self.r_dist_shift = tf.placeholder_with_default(
        0.2, shape=(), name="r_dist_shift")
    self.b_rate_shift = tf.placeholder_with_default(
        0.5, shape=(), name="b_rate_shift")

    self.r_dist = tf.placeholder_with_default(default_ph_value(2.),
                                              shape=shape_pars,
                                              name="r_dist")
    self.b_rate = tf.placeholder_with_default(default_ph_value(3.),
                                              shape=shape_pars,
                                              name="b_rate")

    # background templates
    self.c_nom = tf.placeholder(dtype=tf.float32, shape=(None,), name="c_nom")
    self.c_up = tf.placeholder(
        dtype=tf.float32, shape=(None, None), name="c_up")
    self.c_dw = tf.placeholder(
        dtype=tf.float32, shape=(None, None), name="c_dw")

    # signal template
    self.sig_shape = tf.placeholder(
        dtype=tf.float32, shape=(None,), name="sig_shape")

    self.alpha_pars = [[(self.r_dist - self.r_dist_init) / self.r_dist_shift,
                        (self.b_rate - self.b_rate_init) / self.b_rate_shift]]

    # bkg_shape shape is (n_par_inst, n_bins, 1) if multiple_pars
    self.bkg_shape = int_quad_lin(self.alpha_pars,
                                  self.c_nom, self.c_up, self.c_dw,
                                  multiple_pars=multiple_pars)[0]
    # expected amount of signal
    self.s_exp = tf.placeholder_with_default(default_ph_value(50.),
                                             shape=shape_pars, name="s_exp")
    # expected amount of background
    self.b_exp = tf.placeholder_with_default(default_ph_value(1000.),
                                             shape=shape_pars, name="b_exp")

    if multiple_pars:
      sig_shape = tf.expand_dims(self.sig_shape, axis=0,
                                 name="expanded_sig_shape")
      s_exp = tf.expand_dims(self.s_exp, axis=-1, name="expanded_s_exp")
      b_exp = tf.expand_dims(self.b_exp, axis=-1, name="expanded_b_exp")
    else:
      sig_shape = self.sig_shape
      s_exp = self.s_exp
      b_exp = self.b_exp

    self.t_exp = tf.cast(s_exp * sig_shape +
                         b_exp * self.bkg_shape,
                         dtype=tf.float64, name="t_exp")

    # placeholder for observed data
    self.obs = tf.placeholder(dtype=tf.float64, shape=(None,), name="obs")

    self.h_pois = ds.Poisson(self.t_exp)
    self.h_nll = - \
        tf.cast(tf.reduce_sum(self.h_pois.log_prob(self.obs), axis=-1),
                dtype=tf.float32)

    self.all_pars = OrderedDict([('s_exp', self.s_exp),
                                 ('r_dist', self.r_dist),
                                 ('b_rate', self.b_rate),
                                 ('b_exp', self.b_exp)])

    pars = list(self.all_pars.values())

    self.h_hess, self.h_grad = batch_hessian(self.h_nll, pars)

  def templates_from_dict(self, templates,
                          r_dist=[2.0, 2.2, 1.8],
                          b_rate=[3.0, 3.5, 2.5]):

    def normalise(arr):
        arr = np.array(arr, dtype=np.float32)
        return arr / arr.sum()

    templates = {k: normalise(v) for k, v in templates.items()}

    shift_phs = {self.r_dist_init: r_dist[0],
                 self.r_dist_shift: (r_dist[1] - r_dist[2]) / 2.,
                 self.b_rate_init: b_rate[0],
                 self.b_rate_shift: (b_rate[1] - b_rate[2]) / 2.}

    c_nom = templates[('bkg', r_dist[0], b_rate[0])]
    c_up = np.array([templates[('bkg', r_dist[1], b_rate[0])],
                     templates[('bkg', r_dist[0], b_rate[1])]])
    c_dw = np.array([templates[('bkg', r_dist[2], b_rate[0])],
                     templates[('bkg', r_dist[0], b_rate[2])]])
    sig_shape = templates[('sig',)]

    # remove zeroes
    zero_filter = np.all([(sig_shape != 0.), (c_nom != 0.)], axis=0)
    templates = {k: v[zero_filter] for k, v in templates.items()
                 if not ('pars' in k[0])}

    self.shape_phs = {self.c_nom: c_nom[zero_filter],
                      self.c_up: c_up[:, zero_filter],
                      self.c_dw: c_dw[:, zero_filter],
                      self.sig_shape: sig_shape[zero_filter],
                      **shift_phs}

    return templates

  def templates_from_json(self, json_path,
                          r_dist=[2.0, 2.2, 1.8],
                          b_rate=[3.0, 3.5, 2.5]):

    with open(json_path) as f:
      templates = json.load(f)

    templates = {literal_eval(k): v for k, v in templates.items()}
    templates = self.templates_from_dict(templates, r_dist=r_dist,
                                         b_rate=b_rate)

    return templates

  def asimov_data(self, par_phs={}, sess=None):

    if sess is None:
      sess = tf.get_default_session()
    asimov_data = sess.run(self.t_exp, {**par_phs, **self.shape_phs})
    if self.multiple_pars:
      asimov_data = asimov_data[0]
    return asimov_data

  def asimov_hess(self, par_phs={}, sess=None):

    if sess is None:
      sess = tf.get_default_session()
    obs_phs = {self.obs: self.asimov_data(par_phs, sess=sess)}
    h_hess = sess.run(self.h_hess, {**par_phs, **obs_phs, **self.shape_phs})
    return FisherMatrix(h_hess, par_names=list(self.all_pars.keys()))

  def hessian_and_gradient(self, pars, par_phs={}, obs_phs={}, sess=None):

    if sess is None:
      sess = tf.get_default_session()

    pars = tuple(pars)
    nll, hess, grad = sess.run([self.h_nll, self.h_hess, self.h_grad],
                               feed_dict={**par_phs, **obs_phs,
                                          **self.shape_phs})

    indices = [list(self.all_pars.keys()).index(par) for par in pars]
    idx_subset = np.reshape(list(it.product(indices, indices)),
                            (len(pars), len(pars), -1)).T

    sub_hess = hess[:, idx_subset[0], idx_subset[1]]
    sub_grad = grad[:, indices]
    return nll, sub_hess, sub_grad
