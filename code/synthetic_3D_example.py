from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from neyman.models import IndependentList
import tensorflow_probability as tfp

from collections import OrderedDict

ds = tfp.distributions


class SyntheticThreeDimExample(object):

  def __init__(self):

    # background nuisance parameters
    self.r_dist = tf.placeholder_with_default(2., shape=(), name="r_dist")
    self.b_rate = tf.placeholder_with_default(3., shape=(), name="b_rate")

    # 2D normal  with variable mean
    MultNormal = ds.MultivariateNormalFullCovariance
    self.b_01 = MultNormal(loc=[self.r_dist, 0.],
                           covariance_matrix=[[5., 0.], [0., 9.]], name="b_01")

    # 1D exponential in third dimension (rate is a nuisance)
    self.b_2 = ds.Exponential(rate=self.b_rate, name="b_2")

    # full background distribution
    self.b_dist = IndependentList(distributions=[self.b_01, self.b_2],
                                  name="b_dist")

    # 2D normal fully defined
    self.s_01 = ds.MultivariateNormalDiag(loc=[0., 0.],
                                          scale_diag=[1., 1.], name="s_01")

    # 1D expopential fully defined
    self.s_2 = ds.Exponential(rate=2.0, name="s_2")

    # full signal distributions
    self.s_dist = IndependentList(distributions=[self.s_01, self.s_2],
                                  name="s_dist")

    # expected amount of signal
    self.s_exp = tf.placeholder_with_default(50., shape=(), name="s_exp")
    # expected amount of background
    self.b_exp = tf.placeholder_with_default(1000., shape=(), name="b_exp")

    # compute signal fraction from s_exp and b_exp
    self.mu = self.s_exp / (self.s_exp + self.b_exp)

    # full mixture distribution
    self.m_dist = ds.Mixture(cat=ds.Categorical(probs=[1. - self.mu, self.mu]),
                             components=[self.b_dist, self.s_dist],
                             name="mixture")

    # ordered dict with all model parameters
    self.all_pars = OrderedDict([('s_exp', self.s_exp),
                                 ('r_dist', self.r_dist),
                                 ('b_rate', self.b_rate),
                                 ('b_exp', self.b_exp)])

  def generate_data(self, n_samples, seed):

    components = {"bkg": self.b_dist,
                  "sig": self.s_dist}

    dataset = {}
    for c_name, c_dist in components.items():
      dataset[c_name] = c_dist.sample(n_samples, seed=seed,
                                      name=f'{c_dist.name}_sample')

    return dataset

  def train_data(self):

    return self.generate_data(n_samples=100000, seed=27)

  def valid_data(self):

    return self.generate_data(n_samples=100000, seed=37)

  def test_data(self):

    return self.generate_data(n_samples=500000, seed=47)

  def transform_bkg(self, x, r_dist_g=2., b_rate_g=3.):

    # apply transformations over each dim
    x_prime_0 = x[:, 0] - r_dist_g + self.r_dist
    x_prime_1 = x[:, 1]
    x_prime_2 = x[:, 2] * (b_rate_g / self.b_rate)

    x_prime = tf.stack([x_prime_0, x_prime_1, x_prime_2], 1,
                       name="x_prime")
    return x_prime

  def log_density_ratio(self, x):

    log_dr = self.s_dist.log_prob(x) - self.b_dist.log_prob(x)

    return log_dr

  def optimal_classifier(self, x):

    log_dr = self.log_density_ratio(x)
    opt_clf = tf.exp(log_dr) / (1. + tf.exp(log_dr))

    return opt_clf
