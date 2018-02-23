from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import time
from neyman.inferences import batch_hessian
from inference_estimator import InferenceEstimator


timestr = time.strftime("%Y%m%d-%H%M%S")

tf.flags.DEFINE_string("train_data", default="toy_1D_ind_train_samples.npz",
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

  c_norm_dists_fn = lambda : {ds.Normal(loc=400., scale=20.)}

  clf = InferenceEstimator(
      temperature=flags.temperature,
      use_cross_entropy=flags.use_cross_entropy,
      c_norm_dists_fn=c_norm_dists_fn,
      model_dir=flags.model_dir,
      n_bins=4,
      config=config)

  print(clf.model_dir)

  train_data = {}
  val_data = {}
  for key, value in data.items():
    train_data[key], val_data[key] = train_test_split(value) 

  print(data.keys())

  def make_input_fn(data):
    def input_fn():

      components = {}
      for key, value in data.items():
        components[key] = tf.data.Dataset.from_tensor_slices(value)\
                               .shuffle(buffer_size=10000)\
                               .batch(flags.batch_size)

      c_norms = tf.data.Dataset.from_tensors({"sig" : 20., "bkg" : 400.}).repeat()

      dataset = tf.data.Dataset.zip({"components" : components, "c_norms" : c_norms})

      dataset_it = dataset.make_one_shot_iterator()
      next_batch = dataset_it.get_next()

      return next_batch, None 
    return input_fn
    

  train_input_fn = make_input_fn(train_data)
  val_input_fn = make_input_fn(val_data)

  for i in range(flags.n_epochs):
    clf.train(input_fn=train_input_fn)
    clf.evaluate(input_fn=val_input_fn)

if __name__ == "__main__":
  tf.app.run()
