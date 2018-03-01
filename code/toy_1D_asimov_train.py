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
                       help="")
tf.flags.DEFINE_string("model_dir", default=None,
                       help="path to save checkpoints and results")
tf.flags.DEFINE_integer("n_epochs", default=10,
                       help="number of steps for each train call")
tf.flags.DEFINE_float("learning_rate", default=1.e-4,
                       help="")
tf.flags.DEFINE_integer("batch_size", default=128,
                       help="")


flags = tf.flags.FLAGS


def train_and_eval(use_cross_entropy,
                   batch_size,
                   n_epochs,
                   learning_rate,
                   model_dir=None):

  print(learning_rate)

  keys = ["sig","c0_bkg","c1_bkg"]
  input_f = InputFactory(flags.train_data, batch_size, keys)
                                                                  
                                                                  
  problem_name = flags.problem_name
  optimizer="SGD"
  par_dict = problem_dict[problem_name]()
  config = tf.estimator.RunConfig(save_summary_steps=10)
  n_bins = None 

  clf = InferenceEstimator(                                        
      use_cross_entropy=use_cross_entropy,                                     
      model_dir=model_dir,                                         
      learning_rate=learning_rate,                                 
      optimizer=optimizer,                                         
      n_bins=n_bins,                                               
      config=config,                                               
      **par_dict)                                                  
                                                                  
  print(clf.model_dir)                                             
                                                                  
  for i in range(n_epochs):                                        
    clf.train(input_fn=input_f.train_input_fn)                     
    clf.evaluate(input_fn=input_f.val_input_fn)                    


def main(_):


  batch_size = flags.batch_size 

  n_epochs = flags.n_epochs
  use_cross_entropy=False

  train_and_eval(use_cross_entropy, batch_size,
                 n_epochs,
                 learning_rate=flags.learning_rate,
                 model_dir=flags.model_dir)


if __name__ == "__main__":
  tf.app.run()

