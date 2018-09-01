from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tqdm import trange

from synthetic_3D_example import SyntheticThreeDimExample
from train_helpers import MixtureBatcher
import os
import json

k = tf.keras
ds = tfp.distributions
ge = tf.contrib.graph_editor


class SyntheticThreeDimCrossEntropy(object):

  def __init__(self, model_path, seed):

    tf.set_random_seed(seed)

    self.problem = SyntheticThreeDimExample()

    self.batcher = MixtureBatcher(["sig", "bkg"])

    s_batch = self.batcher.batch["sig"]
    b_batch = self.problem.transform_bkg(self.batcher.batch["bkg"])
    b_sizes = [tf.shape(s_batch)[0], tf.shape(b_batch)[0]]
    b_labels = [1, 0]
    train_batch = tf.concat([s_batch, b_batch], axis=0,
                            name="input_batch")
    label_batch = tf.concat([tf.ones((b_size,), dtype=tf.int32) * b_label
                             for b_label, b_size in zip(b_labels, b_sizes)],
                            axis=0)

    k_init = "he_normal"
    Dense = k.layers.Dense
    self.nn_model = k.Sequential([Dense(units=100, activation="relu",
                                        kernel_initializer=k_init,
                                        input_shape=(3,)),
                                  Dense(units=100, activation="relu",
                                        kernel_initializer=k_init),
                                  Dense(units=2, activation="linear")])

    self.logits = self.nn_model(train_batch)
    self.temperature = tf.placeholder_with_default(1., shape=())
    self.probs = tf.nn.softmax(self.logits / self.temperature)

    self.loss = tf.losses.sparse_softmax_cross_entropy(labels=label_batch,
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
    with open(f'{self.model_path}/model.json', 'w') as f:
      json.dump(json_str, f)

    self.saver = tf.train.Saver()
    self.history = {}

  def fit(self, n_epochs, lr, batch_size, seed):

    with tf.Session() as sess:
      train_arrays = sess.run(self.problem.train_data())
      valid_arrays = sess.run(self.problem.valid_data())

    phs_train = {self.lr: lr}

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
          self.batcher.init_iterator(valid_arrays,
                                     batch_size=5000, seed=20)
          val_losses = []
          while True:
            try:
              loss_t = sess.run([self.loss])
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


def main(_):

  clf = SyntheticThreeDimCrossEntropy(model_path="cross_entropy_32", seed=7)
  clf.fit(n_epochs=100, lr=1e-3, seed=7)


if __name__ == "__main__":
  tf.app.run()
