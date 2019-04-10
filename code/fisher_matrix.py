
import numpy as np
import itertools as it
from collections import OrderedDict


class FisherMatrix(object):

  def __init__(self, matrix, par_names=None):

    self.matrix = matrix

    if par_names is None:
      self.pars = range(matrix.shape[0])
    else:
      self.pars = par_names

  def submatrix(self, pars):

    indices = [self.pars.index(par) for par in pars]
    idx_subset = np.reshape(list(it.product(indices, indices)),
                            (len(pars), len(pars), -1))
    return self.matrix[idx_subset[:, :, 0], idx_subset[:, :, 1]]

  def marginals(self, pars):
    diag_elems = np.sqrt(np.diag(np.linalg.inv(self.submatrix(pars))))
    return OrderedDict(zip(pars, diag_elems))
    
  def covariance_matrix(self, pars):
    return np.linalg.inv(self.submatrix(pars))

  def add_matrix(self, matrix):
    return FisherMatrix(self.matrix + matrix, self.pars)
