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
        copy = V4()
        try:
            copy.px = self.px + other.px
            copy.py = self.py + other.py
            copy.pz = self.pz + other.pz
            copy.e = self.e + other.e
        except AttributeError: 
            # If 'other' is not V4 like object then return special NotImplemented error
            return NotImplemented
        return copy

    def __sub__(self, other):
        """sub 2 V4 vectors : v3 = v1 - v2 = v1.__sub__(v2)"""
        copy = V4()
        try:
            copy.px = self.px - other.px
            copy.py = self.py - other.py
            copy.pz = self.pz - other.pz
            copy.e = self.e + other.e
        except AttributeError:
            # If 'other' is not V4 like object then return special NotImplemented error
            return NotImplemented
        return copy

    def __isub__(self, other):
        """Sub another V4 into self"""
        try:
            self.px -= other.px
            self.py -= other.py
            self.pz -= other.pz
            self.e -= other.e
        except AttributeError:
            # If 'other' is not V4 like object then return special NotImplemented error
            return NotImplemented
        return self


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
    f = lambda x : 1. / (x*x)
    width  = tf.where(x_ok, f(safe_x), safe_f(x))
    
    return tf.exp(-width * (eta - center)**2)

# ==================================================================================
# V4 vector constructors for particles
# ==================================================================================

def V4_tau(batch):
    vtau = V4() # tau 4-vector
    vtau.setPtEtaPhiM(batch["PRI_tau_pt"], batch["PRI_tau_eta"], batch["PRI_tau_phi"], 0.8)
    # tau mass 0.8 like in original
    return vtau

def V4_lep(batch):
    vlep = V4() # lepton 4-vector
    vlep.setPtEtaPhiM(batch["PRI_lep_pt"], batch["PRI_lep_eta"], batch["PRI_lep_phi"], 0.)
    # lep mass 0 (either 0.106 or 0.0005 but info is lost)
    return vlep

def V4_met(batch):
    vmet = V4() # met 4-vector
    vmet.setPtEtaPhiM(batch["PRI_met"], 0., batch["PRI_met_phi"], 0.) # met mass zero
    return vmet

def V4_leading_jet(batch):
    vj1 = V4()
    vj1.setPtEtaPhiM(if_then_else(batch["PRI_jet_num"] > 0, batch["PRI_jet_leading_pt"]),
                     if_then_else(batch["PRI_jet_num"] > 0, batch["PRI_jet_leading_eta"]),
                     if_then_else(batch["PRI_jet_num"] > 0, batch["PRI_jet_leading_phi"]),
                         0.) # zero mass
    return vj1

def V4_subleading_jet(batch):
    vj2 = V4()
    vj2.setPtEtaPhiM(if_then_else(batch["PRI_jet_num"] > 1, batch["PRI_jet_subleading_pt"]),
                     if_then_else(batch["PRI_jet_num"] > 1, batch["PRI_jet_subleading_eta"]),
                     if_then_else(batch["PRI_jet_num"] > 1, batch["PRI_jet_subleading_phi"]),
                     0.) # zero mass
    return vj2

# ==================================================================================
# Update data batch
# ==================================================================================

def update_met(batch, vmet):
    batch["PRI_met"] = vmet.pt()
    batch["PRI_met_phi"] = vmet.phi()

def update_jet(batch, vj1, vj2, missing_value_like):
    vjsum = vj1 + vj2
    batch["DER_deltaeta_jet_jet"] = if_then_else(batch["PRI_jet_num"] > 1, vj1.deltaEta(vj2), missing_value_like )
    batch["DER_mass_jet_jet"] = if_then_else(batch["PRI_jet_num"] > 1, vjsum.m(), missing_value_like )
    batch["DER_prodeta_jet_jet"] = if_then_else(batch["PRI_jet_num"] > 1, vj1.eta() * vj2.eta(), missing_value_like )

def update_eta_centrality(batch, missing_value_like):
    eta_centrality_tmp = eta_centrality(batch["PRI_lep_eta"],batch["PRI_jet_leading_eta"],batch["PRI_jet_subleading_eta"])                       
    batch["DER_lep_eta_centrality"] = if_then_else(batch["PRI_jet_num"] > 1, eta_centrality_tmp, missing_value_like )

def update_transverse_met_lep(batch, vlep, vmet):
    vtransverse = V4()
    vtransverse.setPtEtaPhiM(vlep.pt(), 0., vlep.phi(), 0.) # just the transverse component of the lepton
    vtransverse += vmet
    batch["DER_mass_transverse_met_lep"] = vtransverse.m()

def update_mass_vis(batch, vlep, vtau):
    vltau = vlep + vtau # lep + tau
    batch["DER_mass_vis"] = vltau.m()

def update_pt_h(batch, vlep, vmet, vtau):
    vltaumet = vlep + vtau + vmet # lep + tau + met
    batch["DER_pt_h"] = vltaumet.pt()

def update_detla_R_tau_lep(batch, vlep, vtau):
    batch["DER_deltar_tau_lep"] = vtau.deltaR(vlep)

def update_pt_tot(batch, vj1, vj2, vlep, vmet, vtau):
    vtot = vlep + vtau + vmet + vj1 + vj2
    batch["DER_pt_tot"] = vtot.pt()

def update_sum_pt(batch, vlep, vtau):
    batch["DER_sum_pt"] = vlep.pt() + vtau.pt() + batch["PRI_jet_all_pt"] # sum_pt is the scalar sum

def update_pt_ratio_lep_tau(batch, vlep, vtau):
    batch["DER_pt_ratio_lep_tau"] = vlep.pt()/vtau.pt()

def update_met_phi_centrality(batch):
    batch["DER_met_phi_centrality"] = METphi_centrality(batch["PRI_lep_phi"], batch["PRI_tau_phi"], batch["PRI_met_phi"])

def update_all(batch, vj1, vj2, vlep, vmet, vtau, missing_value_like):
    update_met(batch, vmet)
    update_jet(batch, vj1, vj2, missing_value_like)
    update_eta_centrality(batch, missing_value_like)
    update_transverse_met_lep(batch, vlep, vmet)
    update_mass_vis(batch, vlep, vtau)
    update_pt_h(batch, vlep, vmet, vtau)
    update_detla_R_tau_lep(batch, vlep, vtau)
    update_pt_tot(batch, vj1, vj2, vlep, vmet, vtau)
    update_sum_pt(batch, vlep, vtau)
    update_pt_ratio_lep_tau(batch, vlep, vtau)
    update_met_phi_centrality(batch)


# ==================================================================================
# TES : Tau Energy Scale
# ==================================================================================
def tau_energy_scale(batch, scale=1.0, missing_value=0.0):
    """
    Manipulate one primary input : the PRI_tau_pt and recompute the others values accordingly.

    Args
    ----
        batch: the dataset should be a OrderedDict like object.
            This function will modify the given data inplace.
        scale : the factor applied : PRI_tau_pt <-- PRI_tau_pt * scale
        missing_value : (default=0.0) the value used to code missing value. 
            This is not used to find missing values but to write them in feature column that have some.

    Notes :
    -------
        Recompute :
            - PRI_tau_pt
            - PRI_met
            - PRI_met_phi
            - DER_deltaeta_jet_jet
            - DER_mass_jet_jet
            - DER_prodeta_jet_jet
            - DER_lep_eta_centrality
            - DER_mass_transverse_met_lep
            - DER_mass_vis
            - DER_pt_h
            - DER_deltar_tau_lep
            - DER_pt_tot
            - DER_sum_pt
            - DER_pt_ratio_lep_tau
            - DER_met_phi_centrality
            - DER_mass_MMC
    """
    zeros_batch = tf.zeros_like(batch["PRI_tau_pt"])
    missing_value_like = lambda x: tf.zeros_like(x) + missing_value
    batch = collections.OrderedDict(batch)  # Copy to avoid modification of original Dict

    # scale tau energy scale, arbitrary but reasonable value
    vtau_original = V4_tau(batch) # tau 4-vector
    batch["PRI_tau_pt"] *= scale 

    # first built 4-vectors
    vtau = V4_tau(batch) # tau 4-vector
    vlep = V4_lep(batch) # lepton 4-vector
    vmet = V4_met(batch) # met 4-vector
    vj1 = V4_leading_jet(batch) # first jet if it exists
    vj2 = V4_subleading_jet(batch) # second jet if it exists

    # fix MET according to tau pt change
    vtau_original.scaleFixedM( scale - 1.0 )
    vmet = vmet - vtau_original
    vmet.pz = zeros_batch
    vmet.e = vmet.eWithM(0.)

    update_all(batch, vj1, vj2, vlep, vmet, vtau, missing_value_like)
    return batch


# ==================================================================================
# JES : Jet Energy Scale
# ==================================================================================
def jet_energy_scale(batch, scale=1.0, missing_value=0.0):
    """
    Manipulate jet primaries input and recompute the others values accordingly.

    Args
    ----
        batch: the dataset should be a OrderedDict like object.
            This function will modify the given data inplace.
        scale : the factor applied : PRI_jet_pt <-- PRI_jet_pt * scale
        missing_value : (default=0.0) the value used to code missing value. 
            This is not used to find missing values but to write them in feature column that have some.
    """
    zeros_batch = tf.zeros_like(batch["PRI_tau_pt"])
    missing_value_like = lambda x: tf.zeros_like(x) + missing_value
    batch = collections.OrderedDict(batch)  # Copy to avoid modification of original Dict

    vj1_original = V4_leading_jet(batch) # first jet if it exists
    vj2_original = V4_subleading_jet(batch) # second jet if it exists
    # scale jet energy, arbitrary but reasonable value
    batch["PRI_jet_leading_pt"] *= scale
    batch["PRI_jet_subleading_pt"] *= scale
    batch["PRI_jet_all_pt"] *= scale

    # first built 4-vectors
    vtau = V4_tau(batch) # tau 4-vector
    vlep = V4_lep(batch) # lepton 4-vector
    vmet = V4_met(batch) # met 4-vector
    vj1 = V4_leading_jet(batch) # first jet if it exists
    vj2 = V4_subleading_jet(batch) # second jet if it exists

    # fix MET according to jet pt change
    vj1_original.scaleFixedM( scale - 1.0 )
    vj2_original.scaleFixedM( scale - 1.0 )
    vmet = vmet - (vj1_original + vj2_original)
    vmet.pz = zeros_batch
    vmet.e = vmet.eWithM(0.)

    update_all(batch, vj1, vj2, vlep, vmet, vtau, missing_value_like)
    return batch


# ==================================================================================
# LES : Lep Energy Scale
# ==================================================================================
def lep_energy_scale(batch, scale=1.0, missing_value=0.0):
    """
    Manipulate one primary input : the PRI_lep_pt and recompute the others values accordingly.

    Args
    ----
        batch: the dataset should be a OrderedDict like object.
            This function will modify the given data inplace.
        scale : the factor applied : PRI_jet_pt <-- PRI_jet_pt * scale
        missing_value : (default=0.0) the value used to code missing value. 
            This is not used to find missing values but to write them in feature column that have some.
    """
    zeros_batch = tf.zeros_like(batch["PRI_tau_pt"])
    missing_value_like = lambda x: tf.zeros_like(x) + missing_value
    batch = collections.OrderedDict(batch)  # Copy to avoid modification of original Dict

    vlep_original = V4_lep(batch) # lepton 4-vector
    # scale jet energy, arbitrary but reasonable value
    batch["PRI_lep_pt"] *= scale 

    # first built 4-vectors
    vtau = V4_tau(batch) # tau 4-vector
    vlep = V4_lep(batch) # lepton 4-vector
    vmet = V4_met(batch) # met 4-vector
    vj1 = V4_leading_jet(batch) # first jet if it exists
    vj2 = V4_subleading_jet(batch) # second jet if it exists

    # fix MET according to lep pt change
    vlep_original.scaleFixedM( scale - 1.0 )
    vmet = vmet - vlep_original
    vmet.pz = zeros_batch
    vmet.e = vmet.eWithM(0.)

    update_all(batch, vj1, vj2, vlep, vmet, vtau, missing_value_like)
    return batch

# ==================================================================================
# Soft term
# ==================================================================================
def soft_term(batch, sigma_met=3.0, missing_value=0.0):
    """
    Manipulate MET primaries input and recompute the others values accordingly.

    Args
    ----
        batch: the dataset should be a OrderedDict like object.
            This function will modify the given data inplace.
        sigma_met : the mean energy (default = 3 GeV) of the missing v4.
        missing_value : (default=0.0) the value used to code missing value.
            This is not used to find missing values but to write them in feature column that have some.
    """
    zeros_batch = tf.zeros_like(batch["PRI_tau_pt"])
    missing_value_like = lambda x: tf.zeros_like(x) + missing_value
    batch = collections.OrderedDict(batch)  # Copy to avoid modification of original Dict

    # first built 4-vectors
    vtau = V4_tau(batch) # tau 4-vector
    vlep = V4_lep(batch) # lepton 4-vector
    vmet = V4_met(batch) # met 4-vector
    vj1 = V4_leading_jet(batch) # first jet if it exists
    vj2 = V4_subleading_jet(batch) # second jet if it exists

    # Compute the missing v4 vector
    v4_soft_term = V4()
    v4_soft_term.px = tf.random.normal(zeros_batch.shape, mean=0, stddev=sigma_met)
    v4_soft_term.py = tf.random.normal(zeros_batch.shape, mean=0, stddev=sigma_met)
    v4_soft_term.pz = zeros_batch
    v4_soft_term.e = v4_soft_term.eWithM(0.)

    # fix MET according to soft term
    vmet = vmet + v4_soft_term

    update_all(batch, vj1, vj2, vlep, vmet, vtau, missing_value_like)
    return batch



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
    # eta_centrality_tmp = eta_centrality(batch["PRI_lep_eta"],batch["PRI_jet_leading_eta"],batch["PRI_jet_subleading_eta"])                       
    # batch["DER_lep_eta_centrality"] = if_then_else(batch["PRI_jet_num"] > 1, eta_centrality_tmp, missing_value_like )

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

