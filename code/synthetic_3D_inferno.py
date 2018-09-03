from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tqdm import trange

from neyman.inferences import batch_hessian

from synthetic_3D_example import SyntheticThreeDimExample
from train_helpers import MixtureBatcher
import os
import json
import itertools as it
from fisher_matrix import FisherMatrix
import tensorflow as tf
import tensorflow_probability as tfp

k = tf.keras
ds = tfp.distributions
ge = tf.contrib.graph_editor


class SyntheticThreeDimInferno(object):

  def __init__(self, model_path, poi, pars, seed, aux={}):

    tf.set_random_seed(seed)

    self.problem = SyntheticThreeDimExample()

    self.batcher = MixtureBatcher(["sig", "bkg"])

    s_batch = self.batcher.batch["sig"]
    b_batch = self.problem.transform_bkg(self.batcher.batch["bkg"])
    b_sizes = [tf.shape(s_batch)[0], tf.shape(b_batch)[0]]
    train_batch = tf.concat([s_batch, b_batch], axis=0,
                            name="input_batch")

    k_init = "he_normal"
    Dense = k.layers.Dense
    self.nn_model = k.Sequential([Dense(units=100, activation="relu",
                                        kernel_initializer=k_init,
                                        input_shape=(3,)),
                                  Dense(units=100, activation="relu",
                                        kernel_initializer=k_init),
                                  Dense(units=10, activation="linear")])

    self.logits = self.nn_model(train_batch)
    self.temperature = tf.placeholder_with_default(1., shape=())
    self.probs = tf.nn.softmax(self.logits / self.temperature)

    s_probs, b_probs = tf.split(self.probs, b_sizes, axis=0)

    s_counts = tf.reduce_mean(s_probs, axis=0)
    b_counts = tf.reduce_mean(b_probs, axis=0),

    self.exp_counts = tf.cast(self.problem.s_exp * s_counts +
                              self.problem.b_exp * b_counts,
                              dtype=tf.float64)

    self.pois = ds.Poisson(self.exp_counts, name="poisson")
    self.asimov = tf.stop_gradient(self.exp_counts, name="asimov")

    self.nll = - tf.cast(tf.reduce_sum(self.pois.log_prob(self.asimov)),
                         name="nll", dtype=tf.float32)

    all_pars = list(self.problem.all_pars.values())
    self.hess_nll, self.grad_nll = batch_hessian(self.nll,
                                                 all_pars)

    self.aux = aux

    self.nll_aux = {}
    self.hess_nll_aux = {}
    for par, dist in self.aux.items():
        self.nll_aux[par] = -dist.log_prob(self.problem.all_pars[par])
        self.hess_nll_aux[par], _ = batch_hessian(self.nll_aux[par],
                                                  all_pars)

    self.ext_nll = sum([self.hess_nll] + list(self.hess_nll_aux.values()))

    self.cov_nll = self.cov_matrix(pars)
    idx_poi = pars.index(poi)
    self.loss = self.cov_nll[idx_poi, idx_poi]

    # remove stop gradient after loss is computed
    ge.edit.bypass(self.asimov.op)

    self.lr = tf.placeholder(shape=(), dtype=tf.float32)
    self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.train_op = self.optimizer.minimize(
        self.loss, global_step=self.global_step)

    self.init_op = tf.global_variables_initializer()

    self.model_path = model_path

    if not os.path.exists(self.model_path):
      os.makedirs(self.model_path)

    json_str = self.nn_model.to_json()
    with open(f'{self.model_path}/model.json', 'w') as f:
      json.dump(json_str, f)

    self.saver = tf.train.Saver()
    self.history = {}

  def cov_matrix(self, pars):

    pars = tuple(pars)

    indices = [list(self.problem.all_pars.keys()).index(par) for par in pars]
    idx_subset = np.reshape(list(it.product(indices, indices)),
                            (len(pars), len(pars), -1))
    hess_subset = tf.gather_nd(self.ext_nll, idx_subset)
    cov_nll = tf.matrix_inverse(hess_subset)

    return cov_nll

  def fit(self, n_epochs, lr, temperature, batch_size, seed, par_phs={}):

    with tf.Session() as sess:
      train_arrays = sess.run(self.problem.train_data())
      valid_arrays = sess.run(self.problem.valid_data())

    phs_train = {self.lr: lr,
                 self.temperature: temperature}

    phs_val = {self.temperature: temperature}

    rs = np.random.RandomState(seed=seed)

    with tf.Session() as sess:
      k.backend.set_session(sess)
      sess.run(self.init_op)
      batch_n = 0
      with trange(n_epochs) as t:
        for i in t:
          shuffle_seed = rs.randint(np.iinfo(np.int32).max)
          self.batcher.init_iterator(train_arrays,
                                     batch_size=batch_size, seed=shuffle_seed)
          while True:
            try:
              batch_n += 1
              loss_t, _ = sess.run([self.loss, self.train_op], phs_train)
              self.history.setdefault("loss_train", []).append(
                  [batch_n, float(np.sqrt(loss_t))])
            except tf.errors.OutOfRangeError:
              break
          # fix seed for validation set (no need to shuffle)
          self.batcher.init_iterator(valid_arrays,
                                     batch_size=batch_size, seed=20)
          val_losses = []
          while True:
            try:
              loss_t = sess.run([self.loss], phs_val)
              val_losses.append(np.sqrt(loss_t))
            except tf.errors.OutOfRangeError:
              break
          val_loss = np.mean(val_losses)
          val_loss_std = np.std(val_losses, ddof=1)
          t.set_postfix({"mean_val_loss": val_loss})
          self.history.setdefault("loss_valid", []).append(
              [batch_n, float(val_loss)])
          self.history.setdefault("loss_std_valid", []).append(
              [batch_n, float(val_loss_std)])

      self.nn_model.save_weights(f'{self.model_path}/model.h5')
      self.saver.save(sess, f'{self.model_path}/model.ckpt',
                      global_step=self.global_step)
      with open(f'{self.model_path}/history.json', 'w') as fp:
        json.dump(self.history, fp)

  def load_weights(self):
    sess = tf.get_default_session()
    last_ckpt = tf.train.latest_checkpoint(f'{self.model_path}')
    print("loading_vars_from", last_ckpt)
    self.saver.restore(sess, last_ckpt)

  def eval_hessian(self, temperature):

    phs_val = {self.temperature: temperature}

    with tf.Session() as sess:
      valid_arrays = sess.run(self.problem.valid_data())
      self.load_weights()
      self.batcher.init_iterator(valid_arrays,
                                 batch_size=-1, seed=20)
      hess, hess_aux = sess.run([self.hess_nll, self.hess_nll_aux], phs_val)

    pars = list(self.problem.all_pars.keys())
    fisher = FisherMatrix(hess, pars)
    aux_fisher = FisherMatrix(sum(hess_aux.values()), pars)

    return fisher, aux_fisher


def main(_):

  pars = ["s_exp", "r_dist", "b_rate", "b_exp"]

  aux = {"b_rate": ds.Normal(loc=3.0, scale=0.02),
         "b_exp": ds.Normal(loc=1000.0, scale=20.)}

  inferno = SyntheticThreeDimInferno(model_path="default_b_exp_small",
                                     poi="s_exp", pars=pars, seed=7, aux=aux)
  inferno.fit(n_epochs=1, lr=1e-6, temperature=0.1, seed=7)

  hess, hess_aux = inferno.eval_hessian(temperature=0.1)


if __name__ == "__main__":
  tf.app.run()
