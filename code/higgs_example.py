from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from collections import OrderedDict


class HiggsExample(object):

  def __init__(self, features = None):

    default_features = ['PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi',
                        'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi',
                        'PRI_met', 'PRI_met_phi', 'PRI_met_sumet',
                        #'PRI_jet_num',
                        'PRI_jet_leading_pt', 'PRI_jet_leading_eta',
                        'PRI_jet_leading_phi',
                        'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta',
                        'PRI_jet_subleading_phi',
                        'PRI_jet_all_pt',
                        #'DER_mass_MMC',
                        'DER_mass_transverse_met_lep', 'DER_mass_vis',
                        'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
                        'DER_prodeta_jet_jet', 'DER_deltar_tau_lep',
                        'DER_pt_tot', 'DER_sum_pt', 'DER_pt_ratio_lep_tau',
                        'DER_met_phi_centrality', 'DER_lep_eta_centrality']
    if features is None:
      self.features = default_features
    else:
      self.features = features

    # tau energy scale nuisance parameter
    self.tau_energy = tf.placeholder_with_default(0., shape=(), name="tau_energy")

    # signal relative to the expected amount
    self.mu = tf.placeholder_with_default(1., shape=(), name="mu")

    # ordered dict with all model parameters
    self.all_pars = OrderedDict([('mu', self.mu),
                                 ('tau_energy', self.tau_energy)])

  def transform(self, batch):

    # TODO: add transformation code

    return batch

  def make_dense(self, batch):

    dense_batch = tf.concat([batch[f] for f in self.features], axis=1,
                      name="dense_batch")

    return dense_batch

  def get_weight(self, batch, c_name):

    c_factors = {"s" : tf.convert_to_tensor(691.9886077135781),
                 "b" : tf.convert_to_tensor(410999.84732187376)}

    weight = batch["Weight"]
    weight_sum = tf.reduce_sum(weight)
    c_factor = c_factors[c_name]

    return c_factor*weight/weight_sum




