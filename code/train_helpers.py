from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def make_input_fn(data, keys, batch_size):
  def input_fn():
  
    components = {}
    for key, value in data.items():
      if key in keys:
        components[key] = tf.data.Dataset.from_tensor_slices(value)\
                                         .shuffle(buffer_size=10000)\
                                         .batch(batch_size)
  
    dataset = tf.data.Dataset.zip({"components" : components})
  
    dataset_it = dataset.make_one_shot_iterator()
    next_batch = dataset_it.get_next()
  
    return next_batch, None 
  return input_fn

