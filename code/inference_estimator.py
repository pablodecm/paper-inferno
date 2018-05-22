from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.estimator import estimator
from collections import OrderedDict
from neyman.inferences import batch_hessian
import tfplot

ds = tf.contrib.distributions
ge = tf.contrib.graph_editor
k = tf.keras

def small_nn(n_logits=2, softmax_output=False):
  model = k.Sequential()
  activation = "relu" 
  initializer = "glorot_normal" 
  model.add(k.layers.Dense(10, activation=activation,
            kernel_initializer=initializer,
            name="dense_0", input_shape=(2,)))
  model.add(k.layers.Dense(10, activation=activation,
            kernel_initializer=initializer,
            name="dense_1"))
  model.add(k.layers.Dense(n_logits, activation=None,
            kernel_initializer=initializer,name="output"))
  if softmax_output:
    model.add(k.layers.Activation("softmax", name="softmax"))
  return model 


def scatter_argmax(x,y, argmax):
  fig, ax = tfplot.subplots(figsize=(8, 6))
  ax.scatter(x, y, c=argmax, alpha=0.7)
  return fig

def bar_mean(indexes, soft_means, hard_means):
  fig, ax = tfplot.subplots(figsize=(8, 6))
  for i, (s_mean, h_mean) in enumerate(zip(soft_means, hard_means)):
    ax.bar(indexes, h_mean, fill=False, linewidth=2,
           edgecolor="C{}".format(i))
    ax.bar(indexes, s_mean, fill=False, linewidth=2,
           linestyle="--", edgecolor="C{}".format(i))
  
  ax.set_ylim([-0.05, 1.05])
  return fig



class InferenceEstimator(estimator.Estimator):

    def __init__(self,
                 network_fn,
                 c_norm_dists_fn,
                 c_interest="sig",
                 c_transforms_fn=None,
                 optimizer="SGD",
                 learning_rate=0.05,
                 n_bins = None,
                 use_cross_entropy=False,
                 epsilon=1.e-4,
                 model_dir=None,
                 clip_gradients=None,
                 temperature=1.0,
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
             tf.summary.scalar("c_batch_size/{}".format(c_name), c_size)
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
        inputs = tf.reshape(X, (-1,2))
        model = network_fn(n_logits)
        logits = model(inputs)

        probs = tf.nn.softmax(logits/temperature)

        if mode == tf.estimator.ModeKeys.PREDICT:
          predictions = {"probabilities" : probs}
          return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        eps = tf.convert_to_tensor(epsilon)
        split_probs = tf.split(probs, c_sizes, name="split_probs")
        split_means = [tf.reduce_mean(split_prob, axis=0) + eps for split_prob in split_probs]

        mu = tf.convert_to_tensor(1., name="mu")
        split_counts = [ mean*norm*mu if (c_name==c_interest) else mean*norm
             for c_name, mean, norm in zip(c_names, split_means, c_norms)]

        for mean, count, c_name in zip(split_means, split_counts, c_names):
          for n in range(n_logits):
            tf.summary.scalar("mean/{}/{}".format(c_name,n), mean[n])
            tf.summary.scalar("count/{}/{}".format(c_name,n), count[n])
          
        exp_counts = tf.cast(sum(split_counts), dtype=tf.float64)
        # asimov loss
        nuis_pars = norm_nuis + trans_nuis

        with tf.name_scope("compute_asimov_loss"):
          pois = ds.Poisson(exp_counts, name="poisson")

          asimov = tf.stop_gradient(exp_counts, name="asimov")
          ll = tf.cast(tf.reduce_sum(pois.log_prob(asimov),
                       name="likelihood"), dtype=tf.float32)

          # add c_norm constrain terms
          constraint_terms = [] 
          for rv in norm_nuis:
            constraint_terms.append(tf.reduce_sum(rv.log_prob(rv),
                name="c_norm_{}_log_prob".format(rv.name)))
          for rv in trans_nuis:
            constraint_terms.append(tf.reduce_sum(rv.log_prob(rv),
                name="c_transform_{}_log_prob".format(rv.name)))

          nll = - tf.reduce_sum([ll]+constraint_terms)
          hess = batch_hessian(nll, [mu]+nuis_pars)
          cov = tf.matrix_inverse(hess[0])
          asimov_loss = cov[0,0] 

        # remove stop gradient after loss is computed
        ge.edit.bypass(asimov.op)

        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y,
                                                               logits=logits)

        tf.summary.scalar("asimov_loss", asimov_loss) 
        tf.summary.scalar("nll", nll) 
        tf.summary.scalar("cross_entropy", cross_entropy)

        loss = cross_entropy if use_cross_entropy else asimov_loss 

        if mode == tf.estimator.ModeKeys.EVAL:
          argmax = tf.argmax(probs, axis=-1)
          # add image summary
          x_variable = X[:,0]
          y_variable = X[:,1]
          tfplot.summary.plot("scatter_argmax", scatter_argmax,
                              [x_variable, y_variable, argmax])

          hard_means = [tf.bincount(tf.argmax(s_prob, axis=-1,output_type=tf.int32),
                          minlength=n_logits, maxlength=n_logits,
                          dtype=tf.float32)
                          for s_prob in split_probs]

          hard_means = tf.stack([bc/tf.reduce_sum(bc) for bc in hard_means]) 
          soft_means = tf.stack(split_means) 
          indexes = tf.range(n_logits, dtype=tf.int32)
          tfplot.summary.plot("bar_mean", bar_mean, [indexes, soft_means, hard_means])

          asimov_mean = tf.metrics.mean(asimov_loss,name="asimov_mean") 
          metrics = {"asimov_loss" : asimov_mean}


          # Create a SummarySaverHook to save eval summaries
          eval_summary_hook = tf.train.SummarySaverHook(
                                save_steps=1,
                                output_dir= self._model_dir + "/eval_core",
                                summary_op=tf.summary.merge_all())
          evaluation_hooks = [eval_summary_hook]

          return tf.estimator.EstimatorSpec( mode, loss=loss,
              eval_metric_ops=metrics,evaluation_hooks=evaluation_hooks)

        assert mode == tf.estimator.ModeKeys.TRAIN

        summaries = ["learning_rate"]

        # convert to tensor learning rate so it is not a variable
        learning_rate_t=tf.convert_to_tensor(learning_rate)

        train_op = tf.contrib.layers.optimize_loss(loss=loss,
                      global_step=tf.train.get_global_step(),
                      learning_rate=learning_rate_t,
                      optimizer=optimizer,
                      summaries=summaries,
                      clip_gradients=clip_gradients)

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)    

      super(InferenceEstimator, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config)
     
