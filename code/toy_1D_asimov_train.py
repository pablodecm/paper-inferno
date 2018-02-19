from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import time
from neyman.inferences import batch_hessian
timestr = time.strftime("%Y%m%d-%H%M%S")


tf.flags.DEFINE_string("train_data", default="toy_1D_ind_train_samples.npz",
                       help="path of the training data in npz format")
tf.flags.DEFINE_string("model_dir", default="clf_dir/asimov_{}".format(timestr),
                       help="path to save checkpoints and results")
tf.flags.DEFINE_integer("num_epochs", default=1,
                       help="path to save checkpoints and results")


flags = tf.flags.FLAGS

ds = tf.contrib.distributions

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


def asimov_model(features, labels, mode, params):

    layers = small_nn()

    inputs = {}
    logits = {}
    probs = {}
    input_keys =  ["sig", "bkg"]

    for key in input_keys:
    
      net = tf.reshape(features[key], (-1,1))
      for layer in layers.values():
        net = layer(net) 
      logits[key] = net  

      probs[key] = tf.nn.softmax(logits[key])


    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = probs
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    means = { key : tf.reduce_mean(probs[key], axis=0) for key in input_keys}
    exp_sig = tf.convert_to_tensor(20., name="exp_bkg")
    exp_bkg = tf.convert_to_tensor(400., name="exp_sig")
    mu = tf.convert_to_tensor(1., name="s_exp")
    norms = { "sig" : exp_sig*mu,
              "bkg" : exp_bkg }
       
    exp_counts = sum([means[key]*norms[key] for key in input_keys]) 

    pois = ds.Poisson(exp_counts[1])
    nll = - pois.log_prob(tf.stop_gradient(exp_counts[1]))

    hess = batch_hessian(nll, [mu])

    loss = tf.reduce_sum(tf.sqrt(1./hess[0][0]))

    if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec( mode, loss=loss)

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)    

def main(_):

  data = np.load(flags.train_data)

  clf = tf.estimator.Estimator(model_fn=asimov_model, model_dir=flags.model_dir)

  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"sig" : data["sig"], "bkg" : data["bkg"]},
      y=np.ones(data["sig"].shape[0]),
      num_epochs=None,
      batch_size=4000,
      shuffle=True)

  for i in range(100):
    clf.train(input_fn=train_input_fn, steps=1)

if __name__ == "__main__":
  tf.app.run()
