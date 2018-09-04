from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

from neyman.inferences import batch_hessian
from fisher_matrix import FisherMatrix

import tensorflow as tf
import tensorflow_probability as tfp

ds = tfp.distributions


class ExtendedModel(object):

  def __init__(self, problem, aux={}):

    self.problem = problem

    self.all_pars = OrderedDict([('s_exp', self.problem.s_exp),
                                 ('r_dist', self.problem.r_dist),
                                 ('b_rate', self.problem.b_rate),
                                 ('b_exp', self.problem.b_exp)])

    self.n_exp = tf.cast((problem.s_exp + problem.b_exp), dtype=tf.float64,
                         name="n_exp")

    self.m_norm = ds.Poisson(rate=self.n_exp)
    self.s_n_exp = tf.placeholder(shape=(), dtype=tf.float64, name="s_n_exp")
    self.b_n_exp = tf.placeholder(shape=(), dtype=tf.float64, name="b_n_exp")
    self.s_data = tf.placeholder(
        shape=(None, 3), dtype=tf.float32, name="s_data")
    self.b_data = tf.placeholder(
        shape=(None, 3), dtype=tf.float32, name="b_data")

    self.e_nll = -tf.cast(self.s_n_exp, tf.float32) * tf.reduce_mean(
        self.problem.m_dist.log_prob(self.s_data)) \
        - tf.cast(self.b_n_exp, tf.float32) * tf.reduce_mean(
        self.problem.m_dist.log_prob(self.b_data)) \
        - tf.cast(self.m_norm.log_prob(self.s_n_exp + self.b_n_exp),
                  tf.float32)

    all_pars_list = list(self.problem.all_pars.values())
    self.e_hess, self.e_grad = batch_hessian(
        self.e_nll, all_pars_list)

    self.aux = aux

    self.nll_aux = {}
    self.hess_nll_aux = {}
    for par, dist in self.aux.items():
        self.nll_aux[par] = -dist.log_prob(self.problem.all_pars[par])
        self.hess_nll_aux[par], _ = batch_hessian(self.nll_aux[par],
                                                  all_pars_list)

    self.t_hess = sum([self.e_hess] + list(self.hess_nll_aux.values()))

  def hess(self, par_phs={}, obs_phs={}, sess=None):

    if sess is None:
      sess = tf.get_default_session()
    all_pars_names = list(self.problem.all_pars.keys())
    e_hess, e_hess_aux = sess.run(
        [self.e_hess, self.hess_nll_aux], {**par_phs, **obs_phs})
    e_hess = FisherMatrix(e_hess, all_pars_names)
    e_hess_aux = FisherMatrix(e_hess_aux, all_pars_names)
    return e_hess, e_hess_aux
