
import numpy as np
from copy import copy

import GPy

from paramz.transformations import Logexp
from GPy.core.parameterization import Param
from paramz.core.observable_array import ObsAr

from photoz_kernels_cy import kernelparts as kernelparts_fast
from photoz_kernels_cy import kernelparts_diag as kernelparts_diag_fast

class Photoz(GPy.kern.Kern):
    """
    Photoz kernel based on RBF kernel.
    """
    def __init__(self, fcoefs_amp, fcoefs_mu, fcoefs_sig, lines_mu, lines_sig, var_T, slope_T, alpha_C, alpha_L, alpha_T, name='photoz'):
        """ Constructor."""
        # Call standard Kern constructor with 3 dimensions.
        super(Photoz, self).__init__(3, None, name)
        # Store arrays of coefficients.
        self.lines_mu = np.array(lines_mu)
        self.lines_sig = np.array(lines_sig)
        self.numLines = lines_mu.size
        self.fcoefs_amp = np.array(fcoefs_amp)
        self.fcoefs_mu = np.array(fcoefs_mu)
        self.fcoefs_sig = np.array(fcoefs_sig)
        self.numCoefs = fcoefs_amp.shape[1]
        self.numBands = fcoefs_amp.shape[0]
        self.norms = np.sqrt(2*np.pi) * np.sum(self.fcoefs_amp * self.fcoefs_sig, axis=1)
        # Initialize parameters and link them.
        self.slope_T = Param('slope_T', np.asarray(slope_T), Logexp())
        self.var_T = Param('var_T', np.asarray(var_T), Logexp())
        self.alpha_C = Param('alpha_C', np.asarray(alpha_C), Logexp())
        self.alpha_L = Param('alpha_L', np.asarray(alpha_L), Logexp())
        self.alpha_T = Param('alpha_T', np.asarray(alpha_T), Logexp())
        self.link_parameter(self.slope_T)
        self.link_parameter(self.var_T)
        self.link_parameter(self.alpha_C)
        self.link_parameter(self.alpha_L)
        self.link_parameter(self.alpha_T)

    def change_numlines(self, num):
        self.numLines = num
        #self.lines_mu = self.lines_mu[0:num]
        #self.lines_sig = self.lines_sig[0:num]

    def roundband(self, bfloat):
        """Convenience function to cast the last dimension (band index) as integer."""
        # In GPy, numpy arrays are contained in ObsAr, so the values must be extracted.
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
        t1 = X[:,0]
        b1 = self.roundband(X[:,1])
        fz1 = (1.+X[:,2])
        norm1 = np.zeros((NO1,))
        KT, KC, KL = np.zeros((NO1,)), np.zeros((NO1,)), np.zeros((NO1,))
        D_alpha_C, D_alpha_L = np.zeros((NO1,)), np.zeros((NO1,))
        kernelparts_diag_fast(NO1, self.numCoefs, self.numLines,
            self.alpha_C, self.alpha_L, self.alpha_T,
            self.fcoefs_amp, self.fcoefs_mu, self.fcoefs_sig, self.lines_mu[:self.numLines], self.lines_sig[:self.numLines],
            t1, b1, fz1, True, norm1, KL, KC, KT, D_alpha_C, D_alpha_L)
        self.slope_T.gradient = np.sum(dL_dKdiag * t1 * t1 * KT * (KC + KL))
        self.var_T.gradient = np.sum(dL_dKdiag * KT * (KC + KL))
        Tpart = (self.var_T + self.slope_T * t1 * t1) * KT
        self.alpha_C.gradient = np.sum(dL_dKdiag * Tpart * D_alpha_C)
        self.alpha_L.gradient = np.sum(dL_dKdiag * Tpart * D_alpha_L)
        self.alpha_T.gradient = 0

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None: X2 = X
        NO1, NO2 = X.shape[0], X2.shape[0]
        t1 = X[:,0]
        t2 = X2[:,0]
        b1 = self.roundband(X[:,1])
        b2 = self.roundband(X2[:,1])
        fz1 = 1 + X[:,2]
        fz2 = 1 + X2[:,2]
        norm1, norm2 = np.zeros((NO1,)), np.zeros((NO2,))
        KT, KC, KL = np.zeros((NO1, NO2)), np.zeros((NO1, NO2)), np.zeros((NO1, NO2))
        D_alpha_C, D_alpha_L = np.zeros((NO1, NO2)), np.zeros((NO1, NO2))
        kernelparts_fast(NO1, NO2, self.numCoefs, self.numLines,
            self.alpha_C, self.alpha_L, self.alpha_T,
            self.fcoefs_amp, self.fcoefs_mu, self.fcoefs_sig, self.lines_mu[:self.numLines], self.lines_sig[:self.numLines],
            t1, b1, fz1, t2, b2, fz2, True, norm1, norm2, KL, KC, KT, D_alpha_C, D_alpha_L)
        self.slope_T.gradient = np.sum(dL_dK * t1[:,None] * t2[None,:] * KT * (KC + KL))
        self.var_T.gradient = np.sum(dL_dK * KT * (KC + KL))
        Tpart = (self.var_T + self.slope_T * t1[:,None] * t2[None,:]) * KT
        self.alpha_C.gradient = np.sum(dL_dK * Tpart * D_alpha_C)
        self.alpha_L.gradient = np.sum(dL_dK * Tpart * D_alpha_L)
        self.alpha_T.gradient = np.sum(dL_dK * (t1[:,None]-t2[None,:])**2 / self.alpha_T**3 * Tpart * (KC + KL))

    def Kdiag(self, X):
        NO1 = X.shape[0]
        t1 = X[:,0]
        b1 = self.roundband(X[:,1])
        fz1 = (1.+X[:,2])
        norm1 = np.zeros((NO1,))
        KT, KC, KL = np.zeros((NO1,)), np.zeros((NO1,)), np.zeros((NO1,))
        D_alpha_C, D_alpha_L = np.zeros((NO1,)), np.zeros((NO1,))
        kernelparts_diag_fast(NO1, self.numCoefs, self.numLines,
            self.alpha_C, self.alpha_L, self.alpha_T,
            self.fcoefs_amp, self.fcoefs_mu, self.fcoefs_sig, self.lines_mu[:self.numLines], self.lines_sig[:self.numLines],
            t1, b1, fz1, False, norm1, KL, KC, KT, D_alpha_C, D_alpha_L)
        return (self.var_T + self.slope_T * t1 * t1)  * KT * (KC + KL)

    def K(self, X, X2=None):
        if X2 is None: X2 = X
        NO1, NO2 = X.shape[0], X2.shape[0]
        t1 = X[:,0]
        t2 = X2[:,0]
        b1 = self.roundband(X[:,1])
        b2 = self.roundband(X2[:,1])
        fz1 = 1 + X[:,2]
        fz2 = 1 + X2[:,2]
        norm1, norm2 = np.zeros((NO1,)), np.zeros((NO2,))
        KT, KC, KL = np.zeros((NO1, NO2)), np.zeros((NO1, NO2)), np.zeros((NO1, NO2))
        D_alpha_C, D_alpha_L = np.zeros((NO1, NO2)), np.zeros((NO1, NO2))
        kernelparts_fast(NO1, NO2, self.numCoefs, self.numLines,
            self.alpha_C, self.alpha_L, self.alpha_T,
            self.fcoefs_amp, self.fcoefs_mu, self.fcoefs_sig, self.lines_mu[:self.numLines], self.lines_sig[:self.numLines],
            t1, b1, fz1, t2, b2, fz2, False, norm1, norm2, KL, KC, KT, D_alpha_C, D_alpha_L)
        return (self.var_T + self.slope_T * t1[:,None] * t2[None,:]) * KT * (KC + KL)

    def gradients_X(self, dL_dK, X, X2in=None):

        if X2in is None:
            X2 = X
        else:
            X2 = X2in
        NO1, NO2 = X.shape[0], X2.shape[0]
        t1 = X[:,0]
        t2 = X2[:,0]
        b1 = self.roundband(X[:,1])
        b2 = self.roundband(X2[:,1])
        fz1 = 1 + X[:,2]
        fz2 = 1 + X2[:,2]
        norm1, norm2 = np.zeros((NO1,)), np.zeros((NO2,))
        KT, KC, KL = np.zeros((NO1, NO2)), np.zeros((NO1, NO2)), np.zeros((NO1, NO2))
        D_alpha_C, D_alpha_L = np.zeros((NO1, NO2)), np.zeros((NO1, NO2))
        kernelparts_fast(NO1, NO2, self.numCoefs, self.numLines,
            self.alpha_C, self.alpha_L, self.alpha_T,
            self.fcoefs_amp, self.fcoefs_mu, self.fcoefs_sig, self.lines_mu[:self.numLines], self.lines_sig[:self.numLines],
            t1, b1, fz1, t2, b2, fz2, False, norm1, norm2, KL, KC, KT, D_alpha_C, D_alpha_L)

        tmp = dL_dK * KT * (KC + KL)
        if X2in is None:
            tmp = tmp + tmp.T
            X2 = X
        t1 = X[:,0]
        t2 = X2[:,0]
        grad = np.zeros(X.shape, dtype=np.float64)
        tempfull  = tmp * self.slope_T * np.outer(np.ones(X.shape[0]), t2)
        tempfull -= tmp * (self.var_T + self.slope_T * t1[:,None] * t2[None,:]) * (t1[:,None] - t2[None,:]) / self.alpha_T**2
        np.sum(tempfull, axis=1, out=grad[:,0])
        return grad

    def gradients_X_diag(self, dL_dKdiag, X):
        return np.zeros(X.shape)




class SEDRBF(GPy.kern.Kern):
    """
    SED kernel based on RBF kernel.
    """
    def __init__(self, lines_mu, lines_sig, var_T, slope_T, alpha_C, alpha_L, alpha_T, name='sed'):
        """ Constructor."""
        # Call standard Kern constructor with 3 dimensions.
        super(SEDRBF, self).__init__(2, None, name)
        # Store arrays of coefficients.
        self.lines_mu = np.array(lines_mu)
        self.lines_sig = np.array(lines_sig)
        self.numLines = lines_mu.size
        # Initialize parameters and link them.
        self.kern_T = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=alpha_T)
        self.kern_T.variance.fix()
        self.kern_C = GPy.kern.RBF(input_dim=1, variance=V_C, lengthscale=alpha_C)
        self.kern_L = GPy.kern.RBF(input_dim=1, variance=V_L, lengthscale=alpha_L)

        self.var_T = Param('var_T', np.asarray(var_T), Logexp())
        self.slope_T = Param('slope_T', np.asarray(slope_T), Logexp())
        self.alpha_C = Param('alpha_C', np.asarray(alpha_C), Logexp())
        self.alpha_L = Param('alpha_L', np.asarray(alpha_L), Logexp())
        self.alpha_T = Param('alpha_T', np.asarray(alpha_T), Logexp())
        self.link_parameter(self.var_T)
        self.link_parameter(self.slope_T)
        self.link_parameter(self.alpha_C)
        self.link_parameter(self.alpha_L)
        self.link_parameter(self.alpha_T)

    def parameters_changed(self):
        self.kern_T.lengthscale = self.alpha_T
        self.kern_C.lengthscale = self.alpha_C
        self.kern_L.lengthscale = self.alpha_L
        self.kern_C.variance = self.V_C
        self.kern_L.variance = self.V_L
        super(SEDRBF,self).parameters_changed()

    def change_numlines(self, num):
        self.numLines = num
        self.lines_mu = self.lines_mu[0:num]
        self.lines_sig = self.lines_sig[0:num]

    def K(self, X, X2=None):
        if X2 is None: X2 = X
        KT = self.kern_T.K(X[:,0:1], X2[:,0:1])
        KC = self.kern_C.K(X[:,1:2], X2[:,1:2])
        KL = self.kern_L.K(X[:,1:2], X2[:,1:2])
        fac = 0*KT
        if self.numLines > 0:
            for mu, sig in zip(self.lines_mu, self.lines_sig):
                term = ((mu-X[:,1][:,None])/sig)**2 + ((mu-X2[:,1][None,:])/sig)**2
                fac += np.exp(-0.5*term)
        return KT * (KC + fac * KL)

    def Kdiag(self, X):
        KT = self.kern_T.Kdiag(X[:,0:1])
        KC = self.kern_C.Kdiag(X[:,1:2])
        KL = self.kern_L.Kdiag(X[:,1:2])
        fac = 0*KT
        if self.numLines > 0:
            for mu, sig in zip(self.lines_mu, self.lines_sig):
                fac += np.exp(-1.0*((mu-X[:,1])/sig)**2)
        return KT * (KC + fac * KL)


    def update_gradients_diag(self, dL_dKdiag, X):

        KT = self.kern_T.Kdiag(X[:,0:1])
        KC = self.kern_C.Kdiag(X[:,1:2])
        KL = self.kern_L.Kdiag(X[:,1:2])
        fac = 0*KT
        if self.numLines > 0:
            for mu, sig in zip(self.lines_mu, self.lines_sig):
                fac += np.exp(-1.0*((mu-X[:,1])/sig)**2)

        self.V_C.gradient = np.sum(dL_dKdiag * KT)
        self.V_L.gradient = np.sum(dL_dKdiag * KT * fac)
        self.alpha_T.gradient = 0.
        self.alpha_C.gradient = 0.
        self.alpha_L.gradient = 0.

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None: X2 = X

        KT = self.kern_T.K(X[:,0:1], X2[:,0:1])
        KC = self.kern_C.K(X[:,1:2], X2[:,1:2])
        KL = self.kern_L.K(X[:,1:2], X2[:,1:2])
        fac = 0*KT
        if self.numLines > 0:
            for mu, sig in zip(self.lines_mu, self.lines_sig):
                term = ((mu-X[:,1][:,None])/sig)**2 + ((mu-X2[:,1][None,:])/sig)**2
                fac += np.exp(-0.5*term)

        self.alpha_T.gradient = - np.sum( self.kern_C.dK_dr_via_X(X[:,0:1], X2[:,0:1]) * (KC + fac * KL)
                                         * dL_dK * self.kern_C._scaled_dist(X[:,0:1], X2[:,0:1]) ) / self.alpha_C

        self.V_C.gradient = np.sum(KT * KC * dL_dK) / self.V_C
        self.alpha_C.gradient = - np.sum(self.kern_C.dK_dr_via_X(X[:,1:2], X2[:,1:2]) * KT
                                         * dL_dK * self.kern_C._scaled_dist(X[:,1:2], X2[:,1:2])) / self.alpha_C

        self.V_L.gradient = np.sum(fac * KT * KL * dL_dK) / self.V_L
        self.alpha_L.gradient = - np.sum(self.kern_L.dK_dr_via_X(X[:,1:2], X2[:,1:2]) * fac * KT
                                         * dL_dK * self.kern_L._scaled_dist(X[:,1:2], X2[:,1:2])) / self.alpha_L
