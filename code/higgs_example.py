from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import OrderedDict
from higgs_four_vector import V4, eta_centrality, METphi_centrality, if_then_else


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
    self.tau_energy = tf.placeholder_with_default(1., shape=(), name="tau_energy")

    # signal relative to the expected amount
    self.mu = tf.placeholder_with_default(1., shape=(), name="mu")

    # ordered dict with all model parameters
    self.all_pars = OrderedDict([('mu', self.mu),
                                 ('tau_energy', self.tau_energy)])

  def transform(self, batch, missing_value = -999.0):

    zeros_batch = tf.zeros_like(batch["PRI_tau_pt"])
    missing_value_batch = zeros_batch + missing_value
    missing_value_like = lambda x: tf.zeros_like(x) + missing_value
    batch = OrderedDict(batch)
    # scale tau energy scale, arbitrary but reasonable value
    batch["PRI_tau_pt"] = batch["PRI_tau_pt"] * self.tau_energy

    # now recompute the DER quantities which are affected

    # first built 4-vectors
    vtau = V4() # tau 4-vector
    vtau.setPtEtaPhiM(batch["PRI_tau_pt"], batch["PRI_tau_eta"], batch["PRI_tau_phi"], 0.8) # tau mass 0.8 like in original

    vlep = V4() # lepton 4-vector
    vlep.setPtEtaPhiM(batch["PRI_lep_pt"], batch["PRI_lep_eta"], batch["PRI_lep_phi"], 0.) # lep mass 0 (either 0.106 or 0.0005 but info is lost)

    vmet = V4() # met 4-vector
    vmet.setPtEtaPhiM(batch["PRI_met"], 0., batch["PRI_met_phi"], 0.) # met mass zero,

    # fix MET according to tau pt change
    vtauDeltaMinus = vtau.copy()
    vtauDeltaMinus.scaleFixedM( (1.-self.tau_energy)/self.tau_energy)
    vmet = vmet + vtauDeltaMinus
    vmet.pz = zeros_batch
    vmet.e = vmet.eWithM(0.)
    batch["PRI_met"] = vmet.pt()
    batch["PRI_met_phi"] = vmet.phi()

    # first jet if it exists
    # NO LINK WITH PRI tau PT. NO LINK WITH ENERGY SCALE
    vj1 = V4()
    vj1.setPtEtaPhiM(if_then_else(batch["PRI_jet_num"] > 0, batch["PRI_jet_leading_pt"]),
                     if_then_else(batch["PRI_jet_num"] > 0, batch["PRI_jet_leading_eta"]),
                     if_then_else(batch["PRI_jet_num"] > 0, batch["PRI_jet_leading_phi"]),
                         0.) # zero mass

    vj2 = V4()
    vj2.setPtEtaPhiM(if_then_else(batch["PRI_jet_num"] > 1, batch["PRI_jet_subleading_pt"]),
                     if_then_else(batch["PRI_jet_num"] > 1, batch["PRI_jet_subleading_eta"]),
                     if_then_else(batch["PRI_jet_num"] > 1, batch["PRI_jet_subleading_phi"]),
                     0.) # zero mass

    vjsum = vj1 + vj2

    batch["DER_deltaeta_jet_jet"] = if_then_else(batch["PRI_jet_num"] > 1, vj1.deltaEta(vj2), missing_value_like )
    batch["DER_mass_jet_jet"] = if_then_else(batch["PRI_jet_num"] > 1, vjsum.m(), missing_value_like )
    batch["DER_prodeta_jet_jet"] = if_then_else(batch["PRI_jet_num"] > 1, vj1.eta() * vj2.eta(), missing_value_like )

    # DOES NOT DEPEND OF ENERGY SCALE
#     eta_centrality_tmp = eta_centrality(batch["PRI_lep_eta"],batch["PRI_jet_leading_eta"],batch["PRI_jet_subleading_eta"])                       
#     batch["DER_lep_eta_centrality"] = if_then_else(batch["PRI_jet_num"] > 1, eta_centrality_tmp, missing_value_like )

    # compute many vector sum
    vtransverse = V4()
    vtransverse.setPtEtaPhiM(batch["PRI_lep_pt"], 0., batch["PRI_lep_phi"], 0.) # just the transverse component of the lepton
    vtransverse = vtransverse + vmet
    batch["DER_mass_transverse_met_lep"] = vtransverse.m()

    vltau = vlep + vtau # lep + tau
    batch["DER_mass_vis"] = vltau.m()

    vlmet = vlep + vmet # lep + met  # FIXME Seems to be unused ?
    vltaumet = vltau + vmet # lep + tau + met

    batch["DER_pt_h"] = vltaumet.pt()

    batch["DER_deltar_tau_lep"] = vtau.deltaR(vlep)

    vtot = vltaumet + vjsum
    batch["DER_pt_tot"] = vtot.pt()

    batch["DER_sum_pt"] = vlep.pt() + vtau.pt() + batch["PRI_jet_all_pt"] # sum_pt is the scalar sum
    batch["DER_pt_ratio_lep_tau"] = vlep.pt()/vtau.pt()

    batch["DER_met_phi_centrality"] = METphi_centrality(batch["PRI_lep_phi"], batch["PRI_tau_phi"], batch["PRI_met_phi"])
    
    # FIXME do not really recompute MMC, apply a simple scaling, better than nothing (but not MET dependence)
    # rescaled_mass_MMC = data["ORIG_mass_MMC"] * data["DER_sum_pt"] / data["ORIG_sum_pt"]
    # data["DER_mass_MMC"] = data["ORIG_mass_MMC"].where(data["ORIG_mass_MMC"] < 0, other=rescaled_mass_MMC)

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




