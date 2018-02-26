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


timestr = time.strftime("%Y%m%d-%H%M%S")

tf.flags.DEFINE_string("train_data", default="toy_1D_train_samples.npz",
                       help="path of the training data in npz format")
tf.flags.DEFINE_string("val_data", default="toy_1D_ind_train_samples.npz",
                       help="path of the training data in npz format")
tf.flags.DEFINE_string("model_dir", default="clf_dir/asimov_{}".format(timestr),
                       help="path to save checkpoints and results")
tf.flags.DEFINE_integer("n_epochs", default=1,
                       help="number of steps for each train call")
tf.flags.DEFINE_integer("batch_size", default=4000, help="")
tf.flags.DEFINE_float("temperature", default=1.0, help="")
tf.flags.DEFINE_boolean("use_cross_entropy", default=False, help="")


flags = tf.flags.FLAGS

ds = tf.contrib.distributions

def main(_):

  data = np.load(flags.train_data)


  config = tf.estimator.RunConfig(save_summary_steps=21)

  def c_norm_dists_fn():
    norm_bkg =  nm.Normal(loc=400., scale=100., value=400.)
    norm_c0_bkg =  nm.Normal(loc=200., scale=20., value=200.)
    norm_c1_bkg =  nm.Normal(loc=200., scale=20., value=200.)

    norm_dict = {"sig" : 20.,
                 "bkg" : norm_bkg,
                 "c0_bkg" : norm_c0_bkg,
                 "c1_bkg" : norm_c1_bkg}
    nuis_pars = [norm_c0_bkg, norm_c1_bkg] 

    return norm_dict, nuis_pars

  clf = InferenceEstimator(
      c_norm_dists_fn=c_norm_dists_fn,
      temperature=flags.temperature,
      use_cross_entropy=flags.use_cross_entropy,
      model_dir=flags.model_dir,
      n_bins=None,
      config=config)

  print(clf.model_dir)

  train_data = {}
  val_data = {}
  for key, value in data.items():
    train_data[key], val_data[key] = train_test_split(value, random_state=17) 

  bkg_names = ["c0_bkg", "c1_bkg"]
  train_data["bkg"] = np.vstack([train_data[n] for n in bkg_names])
  val_data["bkg"] = np.vstack([val_data[n] for n in bkg_names])
  np.random.shuffle(train_data["bkg"])
  np.random.shuffle(val_data["bkg"])
  
  def make_input_fn(data, keys):
    def input_fn():

      components = {}
      for key, value in data.items():
        if key in keys:
          components[key] = tf.data.Dataset.from_tensor_slices(value)\
                                           .shuffle(buffer_size=10000)\
                                           .batch(flags.batch_size)

      dataset = tf.data.Dataset.zip({"components" : components})

      dataset_it = dataset.make_one_shot_iterator()
      next_batch = dataset_it.get_next()

      return next_batch, None 
    return input_fn
    

  keys = ["sig","c0_bkg", "c1_bkg"]
#  keys = ["sig","bkg"]
  train_input_fn = make_input_fn(train_data, keys)
  val_input_fn = make_input_fn(val_data, keys)

  for i in range(flags.n_epochs):
    clf.train(input_fn=train_input_fn)
    clf.evaluate(input_fn=val_input_fn)

if __name__ == "__main__":
  tf.app.run()
