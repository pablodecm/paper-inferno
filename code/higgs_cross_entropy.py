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


class HiggsCrossEntropy(object):

  def __init__(self, model_path, seed, aux={}):

    tf.set_random_seed(seed)

    self.problem = HiggsExample()
    self.batcher = HiggsBatcher(features=self.problem.features,
                                buffer_size = 10000)

    self.scale_means = tf.placeholder(dtype=tf.float32,
                                      shape=(len(self.problem.features)))
    self.scale_stds = tf.placeholder(dtype=tf.float32,
                                     shape=(len(self.problem.features)))
    b_sizes = []
    dense_batches = []
    weights = []
    for c_name in self.batcher.c_names:
      transform_batch = self.problem.transform(self.batcher.batch[c_name])
      dense_batch = self.problem.make_dense(transform_batch)
      weight =  self.problem.get_balanced_weight(transform_batch)
      b_sizes.append(tf.shape(dense_batch)[0])
      dense_batches.append(dense_batch)
      weights.append(weight)

    b_labels = [1, 0]
    label_batch = tf.concat([tf.ones((b_size,), dtype=tf.int32) * b_label
                            for b_label, b_size in zip(b_labels, b_sizes)],
                            axis=0, name="input_batch_l")

    train_batch = tf.concat(dense_batches, axis=0, name="input_batch")
    scaled_train_batch = (train_batch - self.scale_means)/self.scale_stds
    weight_batch = tf.concat(weights, axis=0, name="input_batch_w")[:,0]
    self.labels = label_batch
    k_init = "he_normal"
    Dense = k.layers.Dense
    self.nn_model = k.Sequential([Dense(units=100, activation="relu",
                                        kernel_initializer=k_init,
                                        input_shape=(len(self.problem.features),)),
                                  Dense(units=100, activation="relu",
                                        kernel_initializer=k_init),
                                  Dense(units=10, activation="linear")])

    self.logits = self.nn_model(scaled_train_batch)
    self.temperature = tf.placeholder_with_default(1., shape=())
    self.probs = tf.nn.softmax(self.logits / self.temperature)
    
    self.loss = tf.losses.sparse_softmax_cross_entropy(labels=label_batch,
                                                       weights=weight_batch,
                                                       logits=self.logits)

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

  def fit(self, n_epochs, lr, batch_size, seed):

    train_dict_arr, mean_and_std = self.batcher.kaggle_sets("t")
    valid_dict_arr, _ = self.batcher.kaggle_sets("b")

    self.phs_scale = {self.scale_means : mean_and_std[0],
                      self.scale_stds : mean_and_std[1]}

    phs_train = {self.lr: lr,
                 **self.phs_scale}
    phs_val = {**self.phs_scale}

    rs = np.random.RandomState(seed=seed)

    with tf.Session() as sess:
      k.backend.set_session(sess)
      sess.run(self.init_op)
      batch_n = 0
      with trange(n_epochs) as t:
        for i in t:
          shuffle_seed = rs.randint(np.iinfo(np.int32).max)
          self.batcher.init_iterator(dict_arr=train_dict_arr,
                                     batch_size=batch_size, seed=shuffle_seed)
          while True:
            try:
              batch_n += 1
              loss_t, _ = sess.run([self.loss, self.train_op], phs_train)
              self.history.setdefault("loss_train", []).append(
                  [batch_n, float(loss_t)])
            except tf.errors.OutOfRangeError:
              break
          # fix seed for validation set (no need to shuffle)
          self.batcher.init_iterator(dict_arr=valid_dict_arr,
                                     batch_size=batch_size, seed=20)
          val_losses = []
          while True:
            try:
              loss_t = sess.run([self.loss], phs_val)
              val_losses.append(loss_t)
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


def main(_):


  clf  = HiggsCrossEntropy(model_path="cross_entropy_32", seed=7)
  clf.fit(n_epochs=1, lr=1.e2,
          batch_size=32, seed=17)

if __name__ == "__main__":
  tf.app.run()
