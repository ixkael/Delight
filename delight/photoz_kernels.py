
import numpy as np
from copy import copy
from scipy.special import erf

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
    Mean function of photoz GP
    """
    def __init__(self, alpha, beta, fcoefs_amp, fcoefs_mu, fcoefs_sig,
                 g_AB=1.0, lambdaRef=4.5e3, DL_z=None, name='photoz'):
        """ Constructor."""
        # Call standard Kern constructor with 2 dimensions (z and l).
        super(Photoz_mean_function, self).__init__(4, 1, name)
        # If luminosity_distance function not provided, use approximation
        if DL_z is None:
            self.DL_z = approx_DL()
        else:
            self.DL_z = DL_z
        self.g_AB = g_AB
        self.lambdaRef = lambdaRef
        self.fourpi = 4 * np.pi
        self.sqrthalfpi = np.sqrt(np.pi/2)
        self.fcoefs_amp = np.array(fcoefs_amp)
        self.fcoefs_mu = np.array(fcoefs_mu)
        self.fcoefs_sig = np.array(fcoefs_sig)
        self.numCoefs = fcoefs_amp.shape[1]
        self.norms = np.sqrt(2*np.pi)\
            * np.sum(self.fcoefs_amp * self.fcoefs_sig, axis=1)
        self.alpha = Param('alpha', float(alpha))
        self.alpha.constrain_positive()
        self.link_parameter(self.alpha)
        self.beta = Param('beta', float(beta))
        self.beta.constrain_positive()
        self.link_parameter(self.beta)

    def set_alpha(self, alpha):
        """Set alpha"""
        self.update_model(False)
        index = self.alpha._parent_index_
        self.unlink_parameter(self.alpha)
        self.alpha = Param('alpha', float(alpha))
        self.link_parameter(self.alpha, index=index)
        self.alpha.constrain_positive()
        self.update_model(True)

    def set_beta(self, beta):
        """Set beta"""
        self.update_model(False)
        index = self.beta._parent_index_
        self.unlink_parameter(self.beta)
        self.beta = Param('beta', float(beta))
        self.link_parameter(self.beta, index=index)
        self.beta.constrain_positive()
        self.update_model(True)

    def f(self, X):
        b = X[:, 0].astype(int)
        z = X[:, 1]
        t = X[:, 3]
        l = X[:, 2]
        opz = 1. + z
        fac = l*opz/self.fourpi/self.DL_z(z)**2.0/self.g_AB/self.norms[b]
        thesum = np.zeros_like(t)
        for i in range(self.numCoefs):
            amp, mu, sig = self.fcoefs_amp[b, i],\
                           self.fcoefs_mu[b, i],\
                           self.fcoefs_sig[b, i]
            alphat = self.alpha * t**self.beta
            term1 = (mu * opz - alphat * sig**2) /\
                (1.41421356237 * sig * opz)
            term2 = alphat * (self.lambdaRef - mu/opz + alphat*(sig/opz)**2/2)
            thesum += amp * (1 + erf(term1)) * np.exp(term2) *\
                self.sqrthalfpi * sig
        return (fac * thesum).reshape((-1, 1))

    def gradients_X(self, dL_dF, X):
        grad = np.zeros_like(X)
        b = X[:, 0].astype(int)
        z = X[:, 1]
        t = X[:, 3]
        l = X[:, 2]
        opz = 1 + z
        fac = opz/self.fourpi/self.DL_z(z)**2.0/self.g_AB/self.norms[b]
        sum_t = np.zeros_like(t)
        sum_ell = np.zeros_like(t)
        for i in range(self.numCoefs):
            amp, mu, sig = self.fcoefs_amp[b, i],\
                           self.fcoefs_mu[b, i],\
                           self.fcoefs_sig[b, i]
            alphat = self.alpha * t**self.beta
            term1 = (mu * opz - alphat * sig**2) /\
                (1.41421356237 * sig * opz)
            term2 = alphat * (self.lambdaRef - mu/opz + alphat*(sig/opz)**2/2)
            sum_ell += amp * (1 + erf(term1)) * np.exp(term2) *\
                self.sqrthalfpi * sig

            Dterm1 = (self.alpha*self.beta *
                      np.exp(-((-self.alpha*sig**2*t**self.beta + mu*opz)**2 /
                               (2*sig**2*opz**2))) *
                      self.sqrthalfpi*sig*t**(self.beta-1)) / opz
            Dterm2 = ((self.alpha**2*self.beta*sig**2*t**(2*self.beta-1)) /
                      (2*opz**2) +
                      self.alpha*self.beta*t**(self.beta-1) *
                      (self.lambdaRef +
                       (self.alpha*sig**2*t**self.beta)/(2*opz**2) - mu/opz))
            sum_t += amp * np.exp(term2) * Dterm1 *\
                self.sqrthalfpi * sig
            sum_t += amp * (1 + erf(term1)) * np.exp(term2) * Dterm2 *\
                self.sqrthalfpi * sig

        grad[:, 2] = fac * sum_ell  # ell
        grad[:, 3] = l * fac * sum_t  # t
        if isinstance(self.DL_z, approx_DL):
            dDLdz = self.DL_z.derivative(z)
            # TODO: implement z gradient
        else:
            raise NotImplementedError
        return np.dot(dL_dF, grad)

    def update_gradients(self, dL_dF, X):
        b = X[:, 0].astype(int)
        z = X[:, 1]
        t = X[:, 3]
        l = X[:, 2]
        opz = 1 + z
        fac = 1. / self.fourpi / self.DL_z(z)**2.0 / self.g_AB / self.norms[b]
        sum_alpha = np.zeros_like(t)
        sum_beta = np.zeros_like(t)
        for i in range(self.numCoefs):
            amp, mu, sig = self.fcoefs_amp[b, i],\
                           self.fcoefs_mu[b, i],\
                           self.fcoefs_sig[b, i]
            alphat = self.alpha * t**self.beta
            term1 = (mu * opz - alphat * sig**2) /\
                (1.41421356237 * sig * opz)
            term2 = alphat * (self.lambdaRef - mu/opz + alphat*(sig/opz)**2/2)

            Dterm1 = np.sqrt(2/np.pi) * sig * t**self.beta / opz *\
                np.exp(-0.5*((mu*opz - alphat * sig**2 * t**self.beta) /
                       sig / opz)**2)
            Dterm2 = t**self.beta * (self.lambdaRef - mu/opz +
                                     alphat*(sig/opz)**2/2)\
                + alphat*(t**self.beta*sig/opz)**2/2
            sum_alpha += amp * np.exp(term2) * Dterm1 *\
                self.sqrthalfpi * sig
            sum_alpha += amp * (1 + erf(term1)) * np.exp(term2) * Dterm2 *\
                self.sqrthalfpi * sig

            Dterm1 = - np.sqrt(2/np.pi) * sig * t**self.beta * np.log(t) /\
                (1 + z) * np.exp(-((-self.alpha * sig**2 * t**self.beta +
                                 mu*(1 + z))**2 / (2*sig**2 * (1 + z)**2)))
            Dterm2 = ((self.alpha**2 * sig**2 * t**(2*self.beta) * np.log(t)) /
                      (2*(1 + z)**2) + self.alpha*t**self.beta *
                      (self.lambdaRef + (self.alpha * sig**2 * t**self.beta) /
                      (2*(1 + z)**2) - mu/(1 + z)) * np.log(t))
            sum_beta += amp * np.exp(term2) * Dterm1 *\
                self.sqrthalfpi * sig
            sum_beta += amp * (1 + erf(term1)) * np.exp(term2) * Dterm2 *\
                self.sqrthalfpi * sig

        self.alpha.gradient = np.dot(dL_dF.T, opz * l * fac * sum_alpha)
        ind = t > 0
        if ind.sum() > 0:
            self.beta.gradient = np.dot(dL_dF[ind].T,
                                        (opz * l * fac * sum_beta)[ind])


class Photoz_kernel(Kern):
    """
    Photoz kernel based on RBF kernel.
    """
    def __init__(self, fcoefs_amp, fcoefs_mu, fcoefs_sig,
                 lines_mu, lines_sig,
                 var_C, var_L, alpha_C, alpha_L, alpha_T,
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
        self.var_C = Param('var_C', float(var_C))
        self.var_L = Param('var_L', float(var_L))
        self.alpha_C = Param('alpha_C', float(alpha_C))
        self.alpha_L = Param('alpha_L', float(alpha_L))
        self.alpha_T = Param('alpha_T', float(alpha_T))
        self.link_parameter(self.var_C)
        self.link_parameter(self.var_L)
        self.link_parameter(self.alpha_C)
        self.link_parameter(self.alpha_L)
        self.link_parameter(self.alpha_T)
        self.var_C.constrain_positive()
        self.var_L.constrain_positive()
        self.alpha_C.constrain_positive()
        self.alpha_L.constrain_positive()
        self.alpha_T.constrain_positive()
        # TODO: addd more realistic constraints?

    def set_alpha_C(self, alpha_C):
        """Set alpha_C"""
        self.update_model(False)
        index = self.alpha_C._parent_index_
        self.unlink_parameter(self.alpha_C)
        self.alpha_C = Param('alpha_C', float(alpha_C))
        self.link_parameter(self.alpha_C, index=index)
        self.alpha_C.constrain_positive()
        self.update_model(True)

    def set_alpha_L(self, alpha_L):
        """Set alpha_L"""
        self.update_model(False)
        index = self.alpha_L._parent_index_
        self.unlink_parameter(self.alpha_L)
        self.alpha_L = Param('alpha_L', float(alpha_L))
        self.link_parameter(self.alpha_L, index=index)
        self.alpha_L.constrain_positive()
        self.update_model(True)

    def set_var_C(self, var_C):
        """Set var_C"""
        self.update_model(False)
        index = self.var_C._parent_index_
        self.unlink_parameter(self.var_C)
        self.var_C = Param('var_C', float(var_C))
        self.link_parameter(self.var_C, index=index)
        self.var_C.constrain_positive()
        self.update_model(True)

    def set_var_L(self, var_L):
        """Set var_L"""
        self.update_model(False)
        index = self.var_L._parent_index_
        self.unlink_parameter(self.var_L)
        self.var_L = Param('var_L', float(var_L))
        self.link_parameter(self.var_L, index=index)
        self.var_L.constrain_positive()
        self.update_model(True)

    def set_alpha_T(self, alpha_T):
        """Set alpha_T"""
        self.update_model(False)
        index = self.alpha_T._parent_index_
        self.unlink_parameter(self.alpha_T)
        self.alpha_T = Param('alpha_T', float(alpha_T))
        self.link_parameter(self.alpha_T, index=index)
        self.alpha_T.constrain_positive()
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
        self.var_C.gradient = np.sum(dL_dKdiag * KT * prefac * KC)
        self.var_L.gradient = np.sum(dL_dKdiag * KT * prefac * KL)
        self.alpha_C.gradient = np.sum(dL_dKdiag * self.var_C *
                                       KT * prefac * D_alpha_C)
        self.alpha_L.gradient = np.sum(dL_dKdiag * self.var_L *
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
        self.var_C.gradient = np.sum(dL_dK * KT * prefac * KC)
        self.var_L.gradient = np.sum(dL_dK * KT * prefac * KL)
        self.alpha_C.gradient\
            = np.sum(dL_dK * self.var_C * KT * prefac * D_alpha_C)
        self.alpha_L.gradient\
            = np.sum(dL_dK * self.var_L * KT * prefac * D_alpha_L)
        self.alpha_T.gradient\
            = np.sum(dL_dK * (t1[:, None]-t2[None, :])**2 / self.alpha_T**3 *
                     KT * prefac * (self.var_C*KC + self.var_L*KL))

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
        return KT * prefac**2 * (self.var_C*KC + self.var_L*KL)

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
        return KT * prefac**2 * (self.var_C*KC + self.var_L*KL)

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

        tmp = dL_dK * KT * prefac**2 * (self.var_C*KC + self.var_L*KL)
        grad = np.zeros(X.shape, dtype=np.float64)

        tempfull = - tmp * (t1[:, None] - t2[None, :])\
            / self.alpha_T**2
        np.sum(tempfull, axis=1, out=grad[:, 3])  # t

        # TODO: add kernel derivatives with respect to redshift
        if False:
            prefac = fz2[None, :]\
                / (self.fourpi * self.g_AB * self.DL_z(X[:, 1])[:, None] *
                   self.DL_z(X2[:, 1])[None, :])
            cst = dL_dK * KT * prefac
            tempfull = (2 * fz1[:, None] - 2 * fz1[:, None]**2 *
                        self.DL_z.derivative(X[:, 1])[:, None]) *\
                (self.var_C*KC + self.var_L*KL)\
                + D_alpha_z * fz1[:, None]**2
            np.sum(cst * tempfull, axis=1, out=grad[:, 1])  # z

        return grad

    def gradients_X_diag(self, dL_dKdiag, X):
        return self.gradients_X(dL_dKdiag, X)
