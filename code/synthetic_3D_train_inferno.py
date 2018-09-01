from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json

from synthetic_3D_inferno import SyntheticThreeDimInferno

from train_helpers import NumpyEncoder
import tensorflow as tf
import tensorflow_probability as tfp

ds = tfp.distributions

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--b_name", help="benchmark name")
parser.add_argument("--n_epochs", default=200, help="training epochs")
parser.add_argument("--lr", default=1e-6, help="learning rate")
parser.add_argument("--n_inits", default=100, help="number of instances")
parser.add_argument("--batch_size", default=5000, help="batch size sig/bkg")

parser.add_argument("--t_train", default=0.1, help="train softmax temperature")
parser.add_argument("--t_eval", default=0.02, help="eval softmax temperature")


def main():

  args = parser.parse_args()

  # benchmark configuration
  aux_loc = [50., 2.0, 3.0, 1000.]
  aux_none = [None, None, None, None]
  aux_std = [None, 0.4, 1.0, 100.]
  all_pars = ["s_exp", "r_dist", "b_rate", "b_exp"]

  benchmarks = {"b_0": (["s_exp"], aux_none),
                "b_1": (["s_exp", "r_dist"], aux_none),
                "b_2": (["s_exp", "r_dist", "b_rate"], aux_none),
                "b_2_aux": (["s_exp", "r_dist", "b_rate"], aux_std),
                "b_3_aux": (["s_exp", "r_dist", "b_rate", "b_exp"], aux_std)}

  t_train = args.t_train
  t_eval = args.t_eval

  n_epochs = args.n_epochs
  lr = args.lr
  batch_size = args.batch_size
  n_inits = args.n_inits
  b_name = args.b_name

  results = []
  for i in range(n_inits):
    print(f"model {i}")
    seed = 777 + i
    g = tf.Graph()
    with g.as_default():
      b_config = benchmarks[b_name]
      pars, aux_config = b_config

      aux = {p: ds.Normal(l, s) for p, l, s
             in zip(all_pars, aux_loc, aux_config)
             if s is not None}

      print(aux)

      model_path = (
          f"../data/models/{b_name}/ne_{n_epochs}_lr_{lr}"
          f"_bs_{batch_size}_t_{t_train}/init_{i}"
      )

      inferno = SyntheticThreeDimInferno(model_path=model_path, poi="s_exp",
                                         pars=pars, seed=seed, aux=aux)
      inferno.fit(n_epochs=n_epochs, lr=lr, batch_size=batch_size,
                  temperature=t_train, seed=seed)

      fisher, aux_fisher = inferno.eval_hessian(temperature=t_eval)

      f_total = fisher.add_matrix(aux_fisher.matrix)
      print("results")
      print("fisher", fisher.matrix)
      print("aux_fisher", aux_fisher.matrix)
      margs = f_total.marginals(pars)
      print(pars, list(margs.values()))
      results.append(margs["s_exp"])

      info_dict = {"b_name": b_name,
                   "pars": pars,
                   "aux_std": aux_config[1],
                   "n_epochs": n_epochs,
                   "lr": lr,
                   "batch_size": batch_size,
                   "init": i,
                   "seed": seed,
                   "fisher": fisher.matrix.tolist(),
                   "margs": [[p, v] for p, v in margs.items()],
                   "t_train": t_train,
                   "t_eval": t_eval}

      with open(f'{model_path}/info.json', 'w') as fp:
        json.dump(info_dict, fp, indent=2, cls=NumpyEncoder)

  print("all", results)


if __name__ == '__main__':
    main()
