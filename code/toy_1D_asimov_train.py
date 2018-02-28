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
    c1_shift = nm.Normal(loc=0.0,scale=1.0,value=0.0, name="c1_shift")
    trans_dict = {"c1_bkg" : lambda t: t+c1_shift}
    nuis_pars = [c1_shift]

    return trans_dict, nuis_pars

  par_dict = {"c_norm_dists_fn" : c_norm_dists_fn,
              "c_transforms_fn" : c_transforms_fn}

  return par_dict

def toy_three_components_four_nuis():

  def c_norm_dists_fn():
    c0_norm = nm.Normal(loc=200.,scale=40.,value=200., name="c0_norm")
    c1_norm = nm.Normal(loc=200.,scale=40.,value=200., name="c1_norm")
    norm_dict = { "sig" : 20.,
                  "c0_bkg" : c0_norm,
                  "c1_bkg" : c1_norm }
    nuis_pars = [c0_norm, c1_norm]

    return norm_dict, nuis_pars


  def c_transforms_fn():
    c0_shift = nm.Normal(loc=0.0,scale=1.0,value=0.0, name="c0_shift")
    c1_shift = nm.Normal(loc=0.0,scale=1.0,value=0.0, name="c1_shift")
    trans_dict = {"c0_bkg" : lambda t: t+c0_shift,
                  "c1_bkg" : lambda t: t+c1_shift}
    nuis_pars = [c0_shift, c1_shift]

    return trans_dict, nuis_pars

  par_dict = {"c_norm_dists_fn" : c_norm_dists_fn,
              "c_transforms_fn" : c_transforms_fn}

  return par_dict
problem_dict = {"toy_three_components_no_nuis" : toy_three_components_no_nuis,
                "toy_three_components_c1_shape" : toy_three_components_c1_shape,
                "toy_three_components_four_nuis" : toy_three_components_four_nuis}   


def main(_):

  data =  np.load(flags.train_data)

  train_data = {}
  val_data = {}
  for key, value in data.items():
    train_data[key], val_data[key] = train_test_split(value, random_state=17) 


  bkg_names = ["c0_bkg", "c1_bkg"]
  train_data["bkg"] = np.vstack([train_data[n] for n in bkg_names])
  val_data["bkg"] = np.vstack([val_data[n] for n in bkg_names])
  np.random.shuffle(train_data["bkg"])
  np.random.shuffle(val_data["bkg"])

  #train_data = {k : tf.convert_to_tensor(v) for k,v in train_data.items()}
  #val_data = {k : tf.convert_to_tensor(v) for k,v in val_data.items()}


  learning_rates = []
  learning_rates_x_entropy = [2.1e-2]
  batch_sizes = [1024]
  batch_sizes_x_entropy = [64] 
  optimizer="SGD"
  config = tf.estimator.RunConfig(save_summary_steps=10)
  problem_name = flags.problem_name
  par_dict = problem_dict[problem_name]()
  n_bins = 10 
  epsilon = 1.e-5
  keys = ["sig","c0_bkg", "c1_bkg"]
  n_epochs = flags.n_epochs
  clip_gradients=5.0

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
          clip_gradients=clip_gradients,
          **par_dict)

      print(clf.model_dir)

      for i in range(n_epochs):

        with tf.Graph().as_default():
          clf.train(input_fn=train_input_fn)
          clf.evaluate(input_fn=val_input_fn)

  for learning_rate_x_ent in learning_rates_x_entropy:
    for batch_size_x_ent in batch_sizes_x_entropy:
      
      train_input_fn = make_input_fn(train_data, keys, batch_size_x_ent)
      val_input_fn = make_input_fn(val_data, keys, 10000)


      model_dir = "{}/x_entropy_lr_{:.2E}_batch_{}".format(problem_name,
          learning_rate_x_ent, batch_size_x_ent)

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
        
        with tf.Graph().as_default():
          clf.train(input_fn=train_input_fn)
          clf.evaluate(input_fn=val_input_fn)


if __name__ == "__main__":
  tf.app.run()

