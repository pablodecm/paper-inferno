import sys
import collections
import tensorflow as tf
import numpy as np

class V4:
    """
    A simple 4-vector class to ease calculation
    """
    px=0
    py=0
    pz=0
    e=0
    def __init__(self,apx=0., apy=0., apz=0., ae=0.):
        """
        Constructor with 4 coordinates
        """
        self.px = apx
        self.py = apy
        self.pz = apz
        self.e = ae
#         if self.e + 1e-3 < self.p():
#             raise ValueError("Energy is too small! Energy: {}, p: {}".format(self.e, self.p()))

    def copy(self):
        new_v4 = V4()
        new_v4.px = tf.identity(self.px)
        new_v4.py = tf.identity(self.py)
        new_v4.pz = tf.identity(self.pz)
        new_v4.e = tf.identity(self.e)
        return new_v4
    
    def p2(self):
        return self.px**2 + self.py**2 + self.pz**2
    
    def p(self):
        return tf.sqrt(self.p2())
    
    def pt2(self):
        return self.px**2 + self.py**2
    
    def pt(self):
        return tf.sqrt(self.pt2())
    
    def m(self):
        return tf.sqrt( np.abs( self.e**2 - self.p2() ) + sys.float_info.epsilon ) # abs and epsilon are needed for protection
    
    def eta(self):
        return tf.asinh( self.pz/self.pt() )
    
    def phi(self):
        return tf.atan2(self.py, self.px)
    
    def deltaPhi(self, v):
        """delta phi with another v"""
        return (self.phi() - v.phi() + 3*np.pi) % (2*np.pi) - np.pi
    
    def deltaEta(self,v):
        """delta eta with another v"""
        return self.eta()-v.eta()
    
    def deltaR(self,v):
        """delta R with another v"""
        return tf.sqrt(self.deltaPhi(v)**2+self.deltaEta(v)**2 )

    def eWithM(self,m=0.):
        """recompute e given m"""
        return tf.sqrt(self.p2()+m**2)

    # FIXME this gives ugly prints with 1D-arrays
    def __str__(self):
        return "PxPyPzE( %s,%s,%s,%s)<=>PtEtaPhiM( %s,%s,%s,%s) " % (self.px, self.py,self.pz,self.e,self.pt(),self.eta(),self.phi(),self.m())

    def scale(self,factor=1.): # scale
        """Apply a simple scaling"""
        self.px *= factor
        self.py *= factor
        self.pz *= factor
        self.e = tf.abs( factor*self.e )
    
    def scaleFixedM(self,factor=1.): 
        """Scale (keeping mass unchanged)"""
        m = self.m()
        self.px *= factor
        self.py *= factor
        self.pz *= factor
        self.e = self.eWithM(m)
    
    def setPtEtaPhiM(self, pt=0., eta=0., phi=0., m=0):
        """Re-initialize with : pt, eta, phi and m"""
        self.px = pt*tf.cos(phi)
        self.py = pt*tf.sin(phi)
        self.pz = pt*tf.sinh(eta)
        self.e = self.eWithM(m)
    
    def sum(self, v):
        """Add another V4 into self"""
        self.px += v.px
        self.py += v.py
        self.pz += v.pz
        self.e += v.e
    
    def __iadd__(self, other):
        """Add another V4 into self"""
        try:
            self.px += other.px
            self.py += other.py
            self.pz += other.pz
            self.e += other.e
        except AttributeError: 
            # If 'other' is not V4 like object then return special NotImplemented error
            return NotImplemented
        return self
    
    def __add__(self, other):
        """Add 2 V4 vectors : v3 = v1 + v2 = v1.__add__(v2)"""
        copy = self.copy()
        try:
            copy.px += other.px
            copy.py += other.py
            copy.pz += other.pz
            copy.e += other.e
        except AttributeError: 
            # If 'other' is not V4 like object then return special NotImplemented error
            return NotImplemented
        return copy


def if_then_else(x_ok, x, safe_f=tf.zeros_like):
    safe_x = tf.where(x_ok, x, tf.ones_like(x))    
    return tf.where(x_ok, safe_x, safe_f(x))


# magic variable
# FIXME : does it really returns sqrt(2) if in dead center ?
def METphi_centrality(aPhi, bPhi, cPhi):
    """
    Calculate the phi centrality score for an object to be between two other objects in phi
    Returns sqrt(2) if in dead center
    Returns smaller than 1 if an object is not between
    a and b are the bounds, c is the vector to be tested
    """
    # Safe division see :
    # https://stackoverflow.com/questions/33712178/tensorflow-nan-bug/42497444#42497444
    x = tf.sin(bPhi - aPhi)
    x_ok = tf.not_equal(x, 0.0)
    safe_f = tf.zeros_like
    safe_x = tf.where(x_ok, x, tf.ones_like(x))
    
    caPhi = tf.sin(cPhi - aPhi)
    bcPhi = tf.sin(bPhi - cPhi)
    
    def f(x, caPhi, bcPhi):
        A = caPhi / x
        B = bcPhi / x
        res = (A+B) / tf.sqrt(A**2 + B**2)
        return res
    return tf.where(x_ok, f(safe_x, caPhi, bcPhi), safe_f(x))


# another magic variable
def eta_centrality(eta, etaJ1, etaJ2):
    """
    Calculate the eta centrality score for an object to be between two other objects in eta
    Returns 1 if in dead center
    Returns value smaller than 1/e if object is not between
    """
    center = (etaJ1 + etaJ2) / 2.
    
    x = etaJ1 - center
    x_ok = tf.not_equal(x, 0.0)
    safe_f = tf.zeros_like
    safe_x = tf.where(x_ok, x, tf.ones_like(x))
    f = lambda x : 1. / (x**2)
    width  = tf.where(x_ok, f(safe_x), safe_f(x))
    
    return tf.exp(-width * (eta - center)**2)


def transform(batch, systTauEnergyScale=1.0, missing_value=-999.0):
    zeros_batch = tf.zeros_like(batch["PRI_tau_pt"])
    missing_value_batch = zeros_batch + missing_value
    missing_value_like = lambda x: tf.zeros_like(x) + missing_value
    batch = collections.OrderedDict(batch)  # Copy to avoid modification of original Dict
    # scale tau energy scale, arbitrary but reasonable value
    batch["PRI_tau_pt"] = batch["PRI_tau_pt"] * systTauEnergyScale 

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
    vtauDeltaMinus.scaleFixedM( (1.-systTauEnergyScale)/systTauEnergyScale)
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

