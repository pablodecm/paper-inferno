from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import tensorflow as tf

from synthetic_3D_cross_entropy import SyntheticThreeDimCrossEntropy
from train_helpers import NumpyEncoder

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("--n_epochs", default=200, help="training epochs")
parser.add_argument("--lr", default=1e-3, help="learning rate")
parser.add_argument("--n_inits", default=100, help="number of instances")
parser.add_argument("--batch_size", default=32, help="training batch size")


def main():

  args = parser.parse_args()

  n_epochs = args.n_epochs
  lr = args.lr
  batch_size = args.batch_size
  n_inits = args.n_inits

  for i in range(n_inits):
    print(f"model {i}")
    seed = 777 + i
    g = tf.Graph()
    with g.as_default():

      model_path = (
          f"../data/models/cross_entropy/ne_{n_epochs}_lr_{lr}"
          f"_bs_{batch_size}/init_{i}"
      )
      clf = SyntheticThreeDimCrossEntropy(model_path=model_path, seed=seed)
      clf.fit(n_epochs=n_epochs, lr=1e-3, batch_size=batch_size, seed=seed)

      info_dict = {"b_name": "clf",
                   "n_epochs": n_epochs,
                   "lr": lr,
                   "batch_size": batch_size,
                   "init": i,
                   "seed": seed}

      with open(f'{model_path}/info.json', 'w') as fp:
        json.dump(info_dict, fp, indent=2, cls=NumpyEncoder)


if __name__ == '__main__':
  main()
