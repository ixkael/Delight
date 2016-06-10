
import numpy as np
from copy import copy

import GPy

from paramz.transformations import Logexp
from GPy.core.parameterization import Param
from paramz.core.observable_array import ObsAr
from GPy.kern import Kern
from GPy.core import Mapping

from photoz_kernels_cy import kernelparts
from photoz_kernels_cy import kernelparts_diag

from delight.utils import approx_DL


class Photoz_mean_function(Mapping):
    """
    Mean function of photoz GP.
    """
    def __init__(self, g_AB=1.0, DL_z=None, name='photoz'):
        """ Constructor."""
        # Call standard Kern constructor with 2 dimensions (z and l).
        super(Photoz_mean_function, self).__init__(4, 1, name)
        # If luminosity_distance function not provided, use approximation
        if DL_z is None:
            self.DL_z = approx_DL()
        else:
            self.DL_z = DL_z
        self.g_AB = g_AB
        self.fourpi = 4 * np.pi

    def f(self, X):
        b = X[:, 0]
        z = X[:, 1]
        t = X[:, 3]
        l = X[:, 2]
        return ((1 + z) * l / self.fourpi / self.DL_z(z)**2.0 / self.g_AB)\
            .reshape(-1, 1)

    def gradients_X(self, dL_dF, X):
        b = X[:, 0]
        z = X[:, 1]
        t = X[:, 3]
        l = X[:, 2]
        if isinstance(self.DL_z, approx_DL):
            dDLdz = self.DL_z.derivative(z)
            DLz = self.DL_z(z)
            grad = np.zeros_like(X)
            grad[:, 1] = (l - 2 * l * (1 + z) * dDLdz / DLz)\
                / DLz**2 / self.g_AB / self.fourpi  # z
            grad[:, 2] = (1 + z) / self.fourpi / self.DL_z(z)**2.0 / self.g_AB
            return np.dot(dL_dF, grad)  # ell
        else:
            raise NotImplementedError

    def update_gradients(self, dL_dF, X):
        pass  # no parameters in this function, so nothing here


class Photoz_kernel(Kern):
    """
    Photoz kernel based on RBF kernel.
    """
    def __init__(self, fcoefs_amp, fcoefs_mu, fcoefs_sig,
                 lines_mu, lines_sig,
                 var_T, alpha_C, alpha_L, alpha_T,
                 g_AB=1.0, DL_z=None, name='photoz'):
        """ Constructor."""
        # Call standard Kern constructor with 3 dimensions (t, b and z).
        super(Photoz_kernel, self).__init__(4, None, name)
        # If luminosity_distance function not provided, use approximation
        if DL_z is None:
            self.DL_z = approx_DL()
        else:
            self.DL_z = DL_z
        # Store arrays of coefficients.
        self.g_AB = g_AB
        self.fourpi = 4 * np.pi
        self.lines_mu = np.array(lines_mu)
        self.lines_sig = np.array(lines_sig)
        self.numLines = lines_mu.size
        self.fcoefs_amp = np.array(fcoefs_amp)
        self.fcoefs_mu = np.array(fcoefs_mu)
        self.fcoefs_sig = np.array(fcoefs_sig)
        self.numCoefs = fcoefs_amp.shape[1]
        self.numBands = fcoefs_amp.shape[0]
        self.norms = np.sqrt(2*np.pi)\
            * np.sum(self.fcoefs_amp * self.fcoefs_sig, axis=1)
        # Initialize parameters and link them.
        self.var_T = Param('var_T', float(var_T))
        self.alpha_C = Param('alpha_C', float(alpha_C))
        self.alpha_L = Param('alpha_L', float(alpha_L))
        self.alpha_T = Param('alpha_T', float(alpha_T))
        self.link_parameter(self.var_T)
        self.link_parameter(self.alpha_C)
        self.link_parameter(self.alpha_L)
        self.link_parameter(self.alpha_T)

    def set_alpha_C(self, alpha_C):
        """Set alpha_C"""
        self.update_model(False)
        index = self.alpha_C._parent_index_
        self.unlink_parameter(self.alpha_C)
        self.alpha_C = Param('alpha_C', float(alpha_C))
        self.link_parameter(self.alpha_C, index=index)
        self.update_model(True)

    def set_alpha_L(self, alpha_L):
        """Set alpha_L"""
        self.update_model(False)
        index = self.alpha_L._parent_index_
        self.unlink_parameter(self.alpha_L)
        self.alpha_L = Param('alpha_L', float(alpha_L))
        self.link_parameter(self.alpha_L, index=index)
        self.update_model(True)

    def set_var_T(self, var_T):
        """Set var_T"""
        self.update_model(False)
        index = self.var_T._parent_index_
        self.unlink_parameter(self.var_T)
        self.var_T = Param('var_T', float(var_T))
        self.link_parameter(self.var_T, index=index)
        self.update_model(True)

    def set_alpha_T(self, alpha_T):
        """Set alpha_T"""
        self.update_model(False)
        index = self.alpha_T._parent_index_
        self.unlink_parameter(self.alpha_T)
        self.alpha_T = Param('alpha_T', float(alpha_T))
        self.link_parameter(self.alpha_T, index=index)
        self.update_model(True)

    def change_numlines(self, num):
        self.numLines = num
        # self.lines_mu = self.lines_mu[0:num]
        # self.lines_sig = self.lines_sig[0:num]

    def roundband(self, bfloat):
        """Cast the last dimension (band index) as integer"""
        # In GPy, numpy arrays are type ObsAr, so the values must be extracted.
        if isinstance(bfloat, ObsAr):
            b = bfloat.values.astype(int)
        else:
            b = bfloat.astype(int)
        # Check bounds. This is ok because band indices should never change
        # unless there are tiny numerical errors withint GPy.
        b[b < 0] = 0
        b[b >= self.numBands] = self.numBands - 1
        return b

    def update_gradients_diag(self, dL_dKdiag, X):
        NO1 = X.shape[0]
        b1 = self.roundband(X[:, 0])
        fz1 = (1.+X[:, 1])
        t1 = X[:, 3]
        l1 = X[:, 2]
        norm1 = np.zeros((NO1,))
        KT, KC, KL = np.zeros((NO1,)), np.zeros((NO1,)), np.zeros((NO1,))
        D_alpha_C, D_alpha_L = np.zeros((NO1,)), np.zeros((NO1,))
        kernelparts_diag(NO1, self.numCoefs, self.numLines,
                         self.alpha_C, self.alpha_L, self.alpha_T,
                         self.fcoefs_amp, self.fcoefs_mu, self.fcoefs_sig,
                         self.lines_mu[:self.numLines],
                         self.lines_sig[:self.numLines], t1, b1, fz1, True,
                         norm1, KL, KC, KT, D_alpha_C, D_alpha_L)
        prefac = (fz1 * fz1 /
                  (self.fourpi * self.g_AB * self.DL_z(X[:, 1]) *
                   self.DL_z(X2[:, 1])))**2
        self.var_T.gradient = np.sum(dL_dKdiag * KT * prefac * (KC + KL))
        self.alpha_C.gradient = np.sum(dL_dKdiag * self.var_T *
                                       KT * prefac * D_alpha_C)
        self.alpha_L.gradient = np.sum(dL_dKdiag * self.var_T *
                                       KT * prefac * D_alpha_L)
        self.alpha_T.gradient = 0

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None:
            X2 = X
        NO1, NO2 = X.shape[0], X2.shape[0]
        b1 = self.roundband(X[:, 0])
        fz1 = (1.+X[:, 1])
        t1 = X[:, 3]
        l1 = X[:, 2]
        b2 = self.roundband(X2[:, 0])
        fz2 = (1.+X2[:, 1])
        t2 = X2[:, 3]
        l2 = X2[:, 2]
        norm1, norm2 = np.zeros((NO1,)), np.zeros((NO2,))
        KT, KC, KL\
            = np.zeros((NO1, NO2)), np.zeros((NO1, NO2)), np.zeros((NO1, NO2))
        D_alpha_C, D_alpha_L, D_alpha_z\
            = np.zeros((NO1, NO2)), np.zeros((NO1, NO2)), np.zeros((NO1, NO2))
        kernelparts(NO1, NO2, self.numCoefs, self.numLines,
                    self.alpha_C, self.alpha_L, self.alpha_T,
                    self.fcoefs_amp, self.fcoefs_mu, self.fcoefs_sig,
                    self.lines_mu[:self.numLines],
                    self.lines_sig[:self.numLines],
                    t1, b1, fz1, t2, b2, fz2, True, norm1, norm2,
                    KL, KC, KT, D_alpha_C, D_alpha_L, D_alpha_z)
        prefac = (fz1[:, None] * fz2[None, :] /
                  (self.fourpi * self.g_AB * self.DL_z(X[:, 1])[:, None] *
                   self.DL_z(X2[:, 1])[None, :]))**2
        self.var_T.gradient = np.sum(dL_dK * KT * prefac * (KC + KL))
        self.alpha_C.gradient\
            = np.sum(dL_dK * self.var_T * KT * prefac * D_alpha_C)
        self.alpha_L.gradient\
            = np.sum(dL_dK * self.var_T * KT * prefac * D_alpha_L)
        self.alpha_T.gradient\
            = np.sum(dL_dK * (t1[:, None]-t2[None, :])**2 /
                     self.alpha_T**3 * self.var_T * KT * prefac * (KC + KL))

    def Kdiag(self, X):
        NO1 = X.shape[0]
        b1 = self.roundband(X[:, 0])
        fz1 = (1.+X[:, 1])
        t1 = X[:, 3]
        l1 = X[:, 2]
        norm1 = np.zeros((NO1,))
        KT, KC, KL = np.zeros((NO1,)), np.zeros((NO1,)), np.zeros((NO1,))
        D_alpha_C, D_alpha_L = np.zeros((NO1,)), np.zeros((NO1,))
        kernelparts_diag(NO1, self.numCoefs, self.numLines,
                         self.alpha_C, self.alpha_L, self.alpha_T,
                         self.fcoefs_amp, self.fcoefs_mu, self.fcoefs_sig,
                         self.lines_mu[:self.numLines],
                         self.lines_sig[:self.numLines],
                         t1, b1, fz1, False, norm1, KL, KC, KT,
                         D_alpha_C, D_alpha_L)
        prefac = fz1**2 / (self.fourpi * self.g_AB * self.DL_z(X[:, 1])**2)
        return self.var_T * KT * prefac**2 * (KC + KL)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        NO1, NO2 = X.shape[0], X2.shape[0]
        b1 = self.roundband(X[:, 0])
        fz1 = (1.+X[:, 1])
        t1 = X[:, 3]
        l1 = X[:, 2]
        b2 = self.roundband(X2[:, 0])
        fz2 = (1.+X2[:, 1])
        t2 = X2[:, 3]
        l2 = X2[:, 2]
        norm1, norm2 = np.zeros((NO1,)), np.zeros((NO2,))
        KT, KC, KL\
            = np.zeros((NO1, NO2)), np.zeros((NO1, NO2)), np.zeros((NO1, NO2))
        D_alpha_C, D_alpha_L, D_alpha_z\
            = np.zeros((NO1, NO2)), np.zeros((NO1, NO2)), np.zeros((NO1, NO2))
        kernelparts(NO1, NO2, self.numCoefs, self.numLines,
                    self.alpha_C, self.alpha_L, self.alpha_T,
                    self.fcoefs_amp, self.fcoefs_mu, self.fcoefs_sig,
                    self.lines_mu[:self.numLines],
                    self.lines_sig[:self.numLines],
                    t1, b1, fz1, t2, b2, fz2, False, norm1, norm2,
                    KL, KC, KT, D_alpha_C, D_alpha_L, D_alpha_z)
        prefac = fz1[:, None] * fz2[None, :]\
            / (self.fourpi * self.g_AB * self.DL_z(X[:, 1])[:, None] *
               self.DL_z(X2[:, 1])[None, :])
        return self.var_T * KT * prefac**2 * (KC + KL)

    def gradients_X(self, dL_dK, X, X2=None):
        if X2 is None:
            X2 = X
        NO1, NO2 = X.shape[0], X2.shape[0]
        b1 = self.roundband(X[:, 0])
        fz1 = (1.+X[:, 1])
        t1 = X[:, 3]
        l1 = X[:, 2]
        b2 = self.roundband(X2[:, 0])
        fz2 = (1.+X2[:, 1])
        t2 = X2[:, 3]
        l2 = X2[:, 2]
        norm1, norm2 = np.zeros((NO1,)), np.zeros((NO2,))
        KT, KC, KL\
            = np.zeros((NO1, NO2)), np.zeros((NO1, NO2)), np.zeros((NO1, NO2))
        D_alpha_C, D_alpha_L, D_alpha_z\
            = np.zeros((NO1, NO2)), np.zeros((NO1, NO2)), np.zeros((NO1, NO2))
        kernelparts(NO1, NO2, self.numCoefs, self.numLines,
                    self.alpha_C, self.alpha_L, self.alpha_T,
                    self.fcoefs_amp, self.fcoefs_mu, self.fcoefs_sig,
                    self.lines_mu[:self.numLines],
                    self.lines_sig[:self.numLines],
                    t1, b1, fz1, t2, b2, fz2, False,
                    norm1, norm2, KL, KC, KT,
                    D_alpha_C, D_alpha_L, D_alpha_z)

        prefac = fz1[:, None] * fz2[None, :]\
            / (self.fourpi * self.g_AB * self.DL_z(X[:, 1])[:, None] *
               self.DL_z(X2[:, 1])[None, :])

        tmp = dL_dK * KT * prefac**2 * (KC + KL)
        grad = np.zeros(X.shape, dtype=np.float64)

        tempfull = - tmp * self.var_T * (t1[:, None] - t2[None, :])\
            / self.alpha_T**2
        np.sum(tempfull, axis=1, out=grad[:, 3])  # t

        # TODO: add kernel derivatives with respect to redshift
        tempfull = dL_dK * self.var_T * KT * D_alpha_z
        np.sum(tempfull, axis=1, out=grad[:, 1])  # z

        return grad

    def gradients_X_diag(self, dL_dKdiag, X):
        return self.gradients_X(dL_dKdiag, X)
