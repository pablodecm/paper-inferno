from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import neyman.models as nm

def int_quad_lin(alpha, c_nom, c_up, c_dw):
  "Three-point interpolation, quadratic inside and linear outside"

  alpha_t = tf.tile(tf.expand_dims(alpha,axis=-1),[1, tf.shape(c_nom)[0]])
  a = 0.5*(c_up+c_dw)-c_nom
  b = 0.5*(c_up-c_dw)
  ones = tf.ones_like(alpha_t)
  switch = tf.where(alpha_t < 0.,
                    ones*tf.expand_dims(c_dw-c_nom, axis=0),
                    ones*tf.expand_dims(c_up-c_nom, axis=0))
  abs_var = tf.where(tf.abs(alpha_t) > 1.,
                    (2*b+tf.sign(alpha_t)*a)*(alpha_t-tf.sign(alpha_t))+switch,
                    a*tf.pow(alpha_t,2)+b*alpha_t)
  return c_nom+abs_var

def poisson(x, rate):
  "float64  poisson pdf (avoid numerical inacurracies)"
  x_d = tf.cast(x, tf.float64)
  rate_d = tf.cast(rate, tf.float64)
  log_rate_d = tf.log(rate_d)
  p_d = x_d*log_rate_d - tf.lgamma(tf.convert_to_tensor(1.,dtype=tf.float64)+x_d)-rate_d
  return tf.cast( p_d, tf.float32)

class TemplateLikelihood(object):

  def __init__(self, nuis_loc=2.0, nuis_scale=0.2):

    bkg_suffs = [f'bkg_{v}' for v in ['nom','up','dw']]
    sig_suffs = [f'sig']
    par_names = ['mu', 'r_dist']
    self.phs = {}
    # placeholders for signal and background shapes
    for name in (sig_suffs+bkg_suffs):
      self.phs[f'c_{name}'] = tf.placeholder(dtype=tf.float32,
                                             shape=(None,),
                                             name=f'c_{name}_ph')

    # expected number of signal and background events
    for name in ['sig', 'bkg']:
      self.phs[f'n_{name}'] = tf.placeholder(dtype=tf.float32,
                                            shape=(),
                                            name=f'n_{name}_ph')

    # model parameters (input specified by placeholders)
    for name in par_names:
      self.phs[name] = tf.placeholder(dtype=tf.float32,
                                      shape=(None,),
                                      name=f'{name}_ph')


    for name, def_value in [['nuis_loc', nuis_loc],
                            ['nuis_scale', nuis_scale]]:
      self.phs[name] = tf.placeholder_with_default(def_value,
                                                   shape=(),
                                                   name=f'{name}_ph')

    nuis_t_shape = tf.shape(self.phs['r_dist'])
    # distribution of nuissance parameters
    self.nuis_rv = nm.Normal(loc=tf.fill(nuis_t_shape, self.phs['nuis_loc']),
                        scale=tf.fill(nuis_t_shape, self.phs['nuis_scale']),
                        value=self.phs['r_dist'],
                        name="nuis_rv")

    # background shape as a function of r_dist
    norm_nuis_rv = (self.nuis_rv - self.phs['nuis_loc'])/self.phs['nuis_scale']
    c_bkg = int_quad_lin(norm_nuis_rv,
                         *[self.phs[f'c_{name}'] for name in bkg_suffs])

    # expected events ([batch, bin])
    mu = tf.expand_dims(self.phs['mu'],-1, name='mu_expanded')
    self._expected = mu*self.phs['n_sig']*self.phs['c_sig'] + \
                       self.phs['n_bkg']*c_bkg

    # placeholder for data/asimov
    self.phs['observed'] = tf.placeholder(dtype=tf.float32, shape=(None,),
                                          name='observed')

    self.pars = [self.phs[name] for name in par_names]

  def expected(self):
    return self._expected

  def nll(self, c_term = True):
    poisson_pdf = poisson(self.phs['observed'], self._expected)
    nll = -tf.reduce_sum(poisson_pdf ,-1)
    if c_term:
      r_dist_ext = -self.nuis_rv.log_prob(self.nuis_rv)
      nll_ext = nll+r_dist_ext
      return nll_ext
    else:
      return nll

