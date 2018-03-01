from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from neyman.inferences import batch_hessian
import neyman.models as nm
from inference_estimator import InferenceEstimator
from train_helpers import InputFactory 
from toy_1D_problems import problem_dict


tf.flags.DEFINE_string("train_data", default="toy_1D_train_samples.npz",
                       help="path of the training data in npz format")
tf.flags.DEFINE_string("problem_name", default="toy_three_components_c1_shape",
                       help="path to save checkpoints and results")
tf.flags.DEFINE_integer("n_epochs", default=10,
                       help="number of steps for each train call")


flags = tf.flags.FLAGS




def main(_):


  session_config = tf.ConfigProto()


  learning_rates = [1.1e-4]
  learning_rates_x_entropy = []
  batch_sizes = [1024]
  batch_sizes_x_entropy = [64] 
  optimizer="SGD"
  config = tf.estimator.RunConfig(save_summary_steps=10,
                                  session_config=session_config)
  problem_name = flags.problem_name
  par_dict = problem_dict[problem_name]()
  n_bins = 10 
  epsilon = 1.e-5
  keys = ["sig","c0_bkg", "c1_bkg"]
  n_epochs = flags.n_epochs
  clip_gradients=5.0
  keys = ["sig","c0_sig","c1_sig"]

  for learning_rate in learning_rates:
    for batch_size in batch_sizes:

      input_f = InputFactory(flags.train_data, batch_size, keys)

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
        clf.train(input_fn=input_f.train_input_fn)
        clf.evaluate(input_fn=input_f.val_input_fn)

  for learning_rate_x_ent in learning_rates_x_entropy:
    for batch_size_x_ent in batch_sizes_x_entropy:

      input_f = InputFactory(flags.train_data, batch_size_x_ent, keys)
      

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
        
        clf.train(input_fn=input_f.train_input_fn)
        clf.evaluate(input_fn=input_f.val_input_fn)


if __name__ == "__main__":
#  tf.app.run()
  main(None)

