from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.estimator import estimator
from collections import OrderedDict
from neyman.inferences import batch_hessian

ds = tf.contrib.distributions

def small_nn(n_logits=2):
  layers = OrderedDict() 
  activation = tf.nn.relu
  initializer = tf.glorot_normal_initializer
  layers["dense_0"] = tf.layers.Dense(10, activation=activation,
                              kernel_initializer=initializer(),
                              name="dense_0")
  layers["dense_1"] = tf.layers.Dense(10, activation=activation,
                              kernel_initializer=initializer(),
                              name="dense_1")
  layers["output"] = tf.layers.Dense(n_logits, activation=None, name="output")
  return layers


class InferenceEstimator(estimator.Estimator):

    def __init__(self,
                 c_norm_dists_fn,
                 c_interest="sig",
                 c_transforms_fn=None,
                 temperature=1.0,
                 optimizer="SGD",
                 learning_rate=0.05,
                 n_bins = None,
                 use_cross_entropy=False,
                 model_dir=None,
                 config=None):

      def _model_fn(features, labels, mode):

        if "components" in features:
           # get name of each component
           c_names = sorted(features["components"].keys())
           c_tensors = [features["components"][c_name] for c_name in c_names] 
           c_sizes = [tf.shape(c_tensor)[0] for c_tensor in c_tensors]
           norm_dict, norm_nuis = c_norm_dists_fn()
           c_norms = [norm_dict[c_name] for c_name in c_names]
           for c_name, c_size in zip(c_names, c_sizes):
             tf.summary.scalar("c_batch_size_{}".format(c_name), c_size)
           if c_transforms_fn:
             c_transforms, trans_nuis = c_transforms_fn()
             for i, c_name in enumerate(c_names):
               if c_name in c_transforms:
                 c_tensors[i] = c_transforms[c_name](c_tensors[i])
           else:
             trans_nuis = []
           X = tf.concat(c_tensors, axis=0, name="concat_components")
           y = tf.concat([tf.ones((c_size,),dtype=tf.int32)*i 
             for i, c_size in enumerate(c_sizes)], axis=0) 
        else:
          X = features["X"]
          y = labels 

        n_logits = n_bins if n_bins else len(c_names)
        layers = small_nn(n_logits)
        inputs = tf.reshape(X, (-1,1))
        net = inputs
        for name, layer in layers.items():
          net = layer(net) 
        logits = net

        probs = tf.nn.softmax(logits/temperature)

        if mode == tf.estimator.ModeKeys.PREDICT:
          predictions = {"probabilities" : probs}
          return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        split_probs = tf.split(probs, c_sizes, name="split_probs")
        split_means = [tf.reduce_mean(split_prob, axis=0) for split_prob in split_probs]

        mu = tf.convert_to_tensor(1., name="mu")
        split_counts = [ mean*norm*mu if (c_name==c_interest) else mean*norm
             for c_name, mean, norm in zip(c_names, split_means, c_norms)]

        for mean, count, c_name in zip(split_means, split_counts, c_names):
          for n in range(n_logits):
            tf.summary.scalar("mean/{}/{}".format(c_name,n), mean[n])
            tf.summary.scalar("count/{}/{}".format(c_name,n), count[n])
          
        exp_counts = sum(split_counts) 

        # asimov loss
        nuis_pars = norm_nuis + trans_nuis
        with tf.name_scope("compute_asimov_loss"):
          pois = ds.Poisson(exp_counts, name="poisson")
          asimov = tf.stop_gradient(exp_counts, name="asimov")
          ll = tf.reduce_sum(pois.log_prob(asimov), name="likelihood")

          # add c_norm constrain terms
          constraint_terms = [] 
          for rv in norm_nuis:
            constraint_terms.append(tf.reduce_sum(rv.log_prob(rv),
                name="c_norm_{}_log_prob".format(rv.name)))

          nll = - tf.reduce_sum([ll]+constraint_terms)
          hess = batch_hessian(nll, [mu]+nuis_pars)
          cov = tf.matrix_inverse(hess[0])
          asimov_loss = tf.sqrt(cov[0,0]) 
        

        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y,
                                                               logits=logits)

        tf.summary.scalar("asimov_loss", asimov_loss) 
        tf.summary.scalar("nll", nll) 
        tf.summary.scalar("cross_entropy", cross_entropy)

        loss = cross_entropy if use_cross_entropy else asimov_loss 

        if mode == tf.estimator.ModeKeys.EVAL:
          asimov_mean = tf.metrics.mean(asimov_loss,name="asimov_mean") 
          metrics = {"asimov_loss" : asimov_mean}
          return tf.estimator.EstimatorSpec( mode, loss=loss,
              eval_metric_ops=metrics)

        assert mode == tf.estimator.ModeKeys.TRAIN

        train_op = tf.contrib.layers.optimize_loss(loss=loss,
                      global_step=tf.train.get_global_step(),
                      learning_rate=learning_rate,
                      optimizer=optimizer)

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)    

      super(InferenceEstimator, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config)
     
