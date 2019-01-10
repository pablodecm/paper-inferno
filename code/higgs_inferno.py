from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tqdm import trange

from neyman.inferences import batch_hessian

from higgs_example import HiggsExample
from train_helpers import HiggsBatcher
import os
import json
import itertools as it
from fisher_matrix import FisherMatrix
import tensorflow as tf
import tensorflow_probability as tfp

k = tf.keras
ds = tfp.distributions
ge = tf.contrib.graph_editor


class HiggsInferno(object):

  def __init__(self, model_path, poi, pars, seed, aux={}):

    tf.set_random_seed(seed)

    self.problem = HiggsExample()
    self.batcher = HiggsBatcher(buffer_size = 10000)

    b_sizes = []
    dense_batches = []
    weights = []
    for c_name in self.batcher.c_names:
      transform_batch = self.problem.transform(self.batcher.batch[c_name])
      dense_batch = self.problem.make_dense(transform_batch)
      weight =  self.problem.get_weight(transform_batch, c_name)
      b_sizes.append(tf.shape(dense_batch)[0])
      dense_batches.append(dense_batch)
      weights.append(weight)

    train_batch = tf.concat(dense_batches, axis=0, name="input_batch")

    k_init = "he_normal"
    Dense = k.layers.Dense
    self.nn_model = k.Sequential([Dense(units=100, activation="relu",
                                        kernel_initializer=k_init,
                                        input_shape=(len(self.problem.features),)),
                                  Dense(units=100, activation="relu",
                                        kernel_initializer=k_init),
                                  Dense(units=10, activation="linear")])

    self.logits = self.nn_model(train_batch)
    self.temperature = tf.placeholder_with_default(1., shape=())
    self.probs = tf.nn.softmax(self.logits / self.temperature)

    s_probs, b_probs = tf.split(self.probs, b_sizes, axis=0)

    s_counts = tf.reduce_sum(s_probs*weights[0], axis=0)
    b_counts = tf.reduce_sum(b_probs*weights[1], axis=0)

    # add constant small term to avoid NaNs
    small_const = tf.constant(1e-3)
    self.exp_counts = tf.cast(self.problem.mu * s_counts + b_counts+small_const,
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
    with open('{model_path}/model.json'.format(model_path=self.model_path), 'w') as f:
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
          self.batcher.init_iterator("t",
                                     batch_size=batch_size, seed=shuffle_seed)
          while True:
            try:
              batch_n += 1
              hess_t, loss_t, _ = sess.run([self.hess_nll,self.loss, self.train_op], phs_train)
              print(hess_t, loss_t)
              self.history.setdefault("loss_train", []).append(
                  [batch_n, float(np.sqrt(loss_t))])
            except tf.errors.OutOfRangeError:
              break
          # fix seed for validation set (no need to shuffle)
          self.batcher.init_iterator("b", batch_size=batch_size, seed=20)
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

      self.nn_model.save_weights('{model_path}/model.h5'.format(model_path=self.model_path))
      self.saver.save(sess, '{model_path}/model.ckpt'.format(model_path=self.model_path),
                      global_step=self.global_step)
      with open('{model_path}/history.json'.format(model_path=self.model_path), 'w') as fp:
        json.dump(self.history, fp)

  def load_weights(self):
    sess = tf.get_default_session()
    last_ckpt = tf.train.latest_checkpoint(self.model_path)
    print("loading_vars_from", last_ckpt)
    self.saver.restore(sess, last_ckpt)

  def eval_hessian(self, temperature):

    phs_val = {self.temperature: temperature}

    with tf.Session() as sess:
      self.load_weights()
      self.batcher.init_iterator("b",
                                 batch_size=-1, seed=20)
      hess, hess_aux = sess.run([self.hess_nll, self.hess_nll_aux], phs_val)

    pars = list(self.problem.all_pars.keys())
    fisher = FisherMatrix(hess, pars)
    aux_fisher = FisherMatrix(sum(hess_aux.values()), pars)

    return fisher, aux_fisher


def main(_):

  pars = ["mu","tau_energy"]
  aux = {}

  inferno = HiggsInferno(model_path="higgs_default",
                                     poi="mu", pars=pars, seed=17, aux=aux)
  inferno.fit(n_epochs=1, lr=1e-6,
              temperature=0.1, batch_size=256, seed=17)

  hess, hess_aux = inferno.eval_hessian(temperature=0.1)


if __name__ == "__main__":
  tf.app.run()
