from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import time
from neyman.inferences import batch_hessian
import neyman.models as nm
from inference_estimator import InferenceEstimator
from train_helpers import make_input_fn


tf.flags.DEFINE_string("train_data", default="toy_1D_train_samples.npz",
                       help="path of the training data in npz format")
tf.flags.DEFINE_string("problem_name", default="toy_three_components_c1_shape",
                       help="path to save checkpoints and results")
tf.flags.DEFINE_integer("n_epochs", default=10,
                       help="number of steps for each train call")


flags = tf.flags.FLAGS

ds = tf.contrib.distributions


def toy_three_components_no_nuis():

  def c_norm_dists_fn():
    norm_dict = { "sig" : 20.,
                  "c0_bkg" : 200.,
                  "c1_bkg" : 200. }
    nuis_pars = []

    return norm_dict, nuis_pars

  par_dict = {"c_norm_dists_fn" : c_norm_dist_fn,
              "c_transforms_fn" : None}

  return par_dict

def toy_three_components_c1_shape():

  def c_norm_dists_fn():
    norm_dict = { "sig" : 20.,
                  "c0_bkg" : 200.,
                  "c1_bkg" : 200. }
    nuis_pars = []

    return norm_dict, nuis_pars


  def c_transforms_fn():
    c1_shift = nm.Normal(loc=0.0,scale=0.50,value=0.0, name="c1_shift")
    trans_dict = {"c1_bkg" : lambda t: t+c1_shift}
    nuis_pars = [c1_shift]

    return trans_dict, nuis_pars

  par_dict = {"c_norm_dists_fn" : c_norm_dists_fn,
              "c_transforms_fn" : c_transforms_fn}

  return par_dict

problem_dict = {"toy_three_components_no_nuis" : toy_three_components_no_nuis,
                "toy_three_components_c1_shape" : toy_three_components_c1_shape}   


def main(_):

  data = np.load(flags.train_data)

  train_data = {}
  val_data = {}
  for key, value in data.items():
    train_data[key], val_data[key] = train_test_split(value, random_state=17) 

  bkg_names = ["c0_bkg", "c1_bkg"]
  train_data["bkg"] = np.vstack([train_data[n] for n in bkg_names])
  val_data["bkg"] = np.vstack([val_data[n] for n in bkg_names])
  np.random.shuffle(train_data["bkg"])
  np.random.shuffle(val_data["bkg"])


  learning_rates = [1.e-3, 1.e-4]
  learning_rates_x_entropy = [1.e-2, 1.e-3]
  batch_sizes = [128]
  batch_sizes_x_entropy = [128] 
  optimizer="SGD"
  config = tf.estimator.RunConfig(save_summary_steps=100)
  problem_name = flags.problem_name
  par_dict = problem_dict[problem_name]()
  n_bins = None
  epsilon = 1.e-5
  keys = ["sig","c0_bkg", "c1_bkg"]
  n_epochs = flags.n_epochs

  for learning_rate in learning_rates:
    for batch_size in batch_sizes:

      train_input_fn = make_input_fn(train_data, keys, batch_size)
      val_input_fn = make_input_fn(val_data, keys, 10000)


      model_dir = "{}/asimov_lr_{:.2E}_batch_{}".format(problem_name,
          learning_rate, batch_size)

      clf = InferenceEstimator(
          use_cross_entropy=False,
          model_dir=model_dir,
          learning_rate=learning_rate,
          optimizer=optimizer,
          n_bins=n_bins,
          epsilon=epsilon,
          config=config,
          **par_dict)

      print(clf.model_dir)

      for i in range(n_epochs):
        clf.train(input_fn=train_input_fn)
        clf.evaluate(input_fn=val_input_fn)

  for learning_rate_x_ent in learning_rates_x_entropy:
    for batch_size_x_ent in batch_sizes_x_entropy:
      
      train_input_fn = make_input_fn(train_data, keys, batch_size_x_ent)
      val_input_fn = make_input_fn(val_data, keys, 10000)


      model_dir = "{}/x_entropy_lr_{:.2E}_batch_{}".format(problem_name,
          learning_rate, batch_size_x_ent)

      clf = InferenceEstimator(
          use_cross_entropy=True,
          model_dir=model_dir,
          learning_rate=learning_rate_x_ent,
          optimizer=optimizer,
          n_bins=n_bins,
          epsilon=epsilon,
          config=config,
          **par_dict)

      print(clf.model_dir)

      for i in range(n_epochs):
        clf.train(input_fn=train_input_fn)
        clf.evaluate(input_fn=val_input_fn)


if __name__ == "__main__":
  tf.app.run()

