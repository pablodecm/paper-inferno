from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import time
timestr = time.strftime("%Y%m%d-%H%M%S")


tf.flags.DEFINE_string("train_data", default="toy_1D_cat_train_samples.npz",
                       help="path of the training data in npz format")
tf.flags.DEFINE_string("model_dir", default="clf_dir/{}".format(timestr),
                       help="path to save checkpoints and results")
tf.flags.DEFINE_integer("num_epochs", default=1,
                       help="path to save checkpoints and results")


flags = tf.flags.FLAGS


def small_nn():
  layers = OrderedDict() 
  activation = tf.nn.relu
  initializer = tf.glorot_normal_initializer
  layers["dense_0"] = tf.layers.Dense(10, activation=activation,
                              kernel_initializer=initializer(),
                              name="dense_0")
  layers["dense_1"] = tf.layers.Dense(10, activation=activation,
                              kernel_initializer=initializer(),
                              name="dense_1")
  layers["output"] = tf.layers.Dense(2, activation=None, name="output")
  return layers


def clf_model(features, labels, mode, params):

    inputs = tf.reshape(features["X"], (-1,1))

    layers = small_nn()

    net = inputs
    for layer in layers.values():
      net = layer(net) 
    logits = net  

    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          "probabilities" : tf.nn.softmax(logits)
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec( mode, loss=loss)

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)    

def main(_):

  data = np.load(flags.train_data)
  X_train, X_val, y_train, y_val = train_test_split(data["X"], data["y"])

  clf = tf.estimator.Estimator(model_fn=clf_model, model_dir=flags.model_dir)

  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"X" : X_train},
      y=y_train,
      num_epochs=True,
      batch_size=32,
      shuffle=True)

  val_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"X" : X_val},
      y=y_val,
      batch_size=1024,
      shuffle=False)

  for i in range(flags.num_epochs):
    clf.train(input_fn=train_input_fn)
    clf.evaluate(input_fn=val_input_fn)

if __name__ == "__main__":
  tf.app.run()
