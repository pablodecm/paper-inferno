from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import neyman.models as nm

def toy_three_components_no_nuis():

  def c_norm_dists_fn(rv_collections=[]):
    norm_dict = { "sig" : 20.,
                  "c0_bkg" : 200.,
                  "c1_bkg" : 200. }
    nuis_pars = []

    return norm_dict, nuis_pars

  par_dict = {"c_norm_dists_fn" : c_norm_dist_fn,
              "c_transforms_fn" : None}

  return par_dict

def toy_three_components_c1_shape():

  def c_norm_dists_fn(rv_collections=[]):
    norm_dict = { "sig" : 20.,
                  "c0_bkg" : 200.,
                  "c1_bkg" : 200. }
    nuis_pars = []

    return norm_dict, nuis_pars


  def c_transforms_fn(rv_collections=[]):
    c1_shift = nm.Normal(loc=0.0,scale=1.0,value=0.0, name="c1_shift",
        collections=rv_collections)
    trans_dict = {"c1_bkg" : lambda t: t+c1_shift}
    nuis_pars = [c1_shift]

    return trans_dict, nuis_pars

  par_dict = {"c_norm_dists_fn" : c_norm_dists_fn,
              "c_transforms_fn" : c_transforms_fn}

  return par_dict

def toy_three_components_four_nuis():

  def c_norm_dists_fn(rv_collections=[]):
    c0_norm = nm.Normal(loc=200.,scale=40.,value=200., name="c0_norm",
        collections=rv_collections)
    c1_norm = nm.Normal(loc=200.,scale=40.,value=200., name="c1_norm",
        collections=rv_collections)
    norm_dict = { "sig" : 20.,
                  "c0_bkg" : c0_norm,
                  "c1_bkg" : c1_norm }
    nuis_pars = [c0_norm, c1_norm]

    return norm_dict, nuis_pars


  def c_transforms_fn(rv_collections=[]):
    c0_shift = nm.Normal(loc=0.0,scale=1.0,value=0.0, name="c0_shift",
        collections=rv_collections)
    c1_shift = nm.Normal(loc=0.0,scale=1.0,value=0.0, name="c1_shift",
        collections=rv_collections)
    trans_dict = {"c0_bkg" : lambda t: t+c0_shift,
                  "c1_bkg" : lambda t: t+c1_shift}
    nuis_pars = [c0_shift, c1_shift]

    return trans_dict, nuis_pars

  par_dict = {"c_norm_dists_fn" : c_norm_dists_fn,
              "c_transforms_fn" : c_transforms_fn}

  return par_dict

problem_dict = {"toy_three_components_no_nuis" : toy_three_components_no_nuis,
                "toy_three_components_c1_shape" : toy_three_components_c1_shape,
                "toy_three_components_four_nuis" : toy_three_components_four_nuis}   

