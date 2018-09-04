from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
from os import path
from glob import glob
import argparse

from template_model import TemplateModel
from summary_statistic_computer import SummaryStatisticComputer

import numpy as np
import pandas as pd
import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument("--model_re", help="regular expression for models")
parser.add_argument("--model_type", help="inf or clf")


aux_none = [None, None, None, None]
aux_std = [None, 0.4, 1.0, 100.]

benchmarks = {"b_0": (["s_exp"], aux_none),
              "b_1": (["s_exp", "r_dist"], aux_none),
              "b_2": (["s_exp", "r_dist", "b_rate"], aux_none),
              "b_2_aux": (["s_exp", "r_dist", "b_rate"], aux_std),
              "b_3_aux": (["s_exp", "r_dist", "b_rate", "b_exp"], aux_std)}


def marginal(pars, aux_std, poi="s_exp", row_name="fisher_matrix"):
  def marginal_computer(row):
    f = row[row_name]
    aux_diag = [1. / (e**2) if e is not None else 0. for e in aux_std]
    f_total = f.add_matrix(np.diag(aux_diag))
    return f_total.marginals(pars)[poi]
  return marginal_computer


def benchmark_model(model_re, model_type):

  results = {}
  tm = TemplateModel()
  ssc = SummaryStatisticComputer()
  sess = tf.Session()
  model_types = {"clf": ssc.classifier_shapes,
                 "inf": ssc.inferno_shapes}
  model_paths = glob(model_re)
  common_path = path.dirname(path.commonprefix(model_paths))
  print(common_path)
  for model_path in tqdm(model_paths):
    shapes = model_types[model_type](model_path)
    with sess.as_default():
      tm.templates_from_dict(shapes)
      fisher_matrix = tm.asimov_hess()
      results[model_path] = {"common_path": common_path,
                             "fisher_matrix": fisher_matrix}

  df = pd.DataFrame.from_dict(results, orient="index")
  for b_name, config in benchmarks.items():
    pars, aux = config
    df.loc[:, b_name] = df.apply(marginal(pars, aux), axis=1)

  df.to_csv(f"{common_path}/results.csv")
  return df


def main():
  args = parser.parse_args()
  benchmark_model(args.model_re, args.model_type)


if __name__ == '__main__':
  main()
