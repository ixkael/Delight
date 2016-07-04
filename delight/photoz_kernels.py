
import numpy as np
from copy import copy
from scipy.special import erf
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline

import GPy

from paramz.transformations import Logexp
from GPy.core.parameterization import Param
from paramz.core.observable_array import ObsAr
from GPy.kern import Kern
from GPy.core import Mapping

from photoz_kernels_cy import kernelparts, kernelparts_diag,\
    kernel_parts_interp, find_positions

from delight.utils import approx_DL

kind = "linear"


class Photoz_mean_function(Mapping):
    """
    Mean function of photoz GP
    """
    def __init__(self, alpha, beta, fcoefs_amp, fcoefs_mu, fcoefs_sig,
                 g_AB=1.0, lambdaRef=4.5e3, DL_z=None, name='photoz_mf'):
        """ Constructor."""
        # Call standard Kern constructor with 2 dimensions (z and l).
        super(Photoz_mean_function, self).__init__(4, 1, name)
        # If luminosity_distance function not provided, use approximation
        if DL_z is None:
            self.DL_z = approx_DL()
        else:
            self.DL_z = DL_z
        self.g_AB = g_AB
        assert lambdaRef > 1 and lambdaRef < 1e5
        self.lambdaRef = lambdaRef
        self.fourpi = 4 * np.pi
        self.sqrthalfpi = np.sqrt(np.pi/2)
        assert fcoefs_amp.shape[0] == fcoefs_mu.shape[0] and\
            fcoefs_amp.shape[0] == fcoefs_sig.shape[0]
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
        self.beta.constrain_bounded(0, 1)
        self.link_parameter(self.beta)
        self.hash = 0

    def set_alpha(self, alpha):
        """Set alpha"""
        self.update_model(False)
        index = self.alpha._parent_index_
        self.unlink_parameter(self.alpha)
        self.alpha = Param('alpha', float(alpha))
        self.alpha.constrain_positive()
        self.link_parameter(self.alpha, index=index)
        self.update_model(True)

    def set_beta(self, beta):
        """Set beta"""
        self.update_model(False)
        index = self.beta._parent_index_
        self.unlink_parameter(self.beta)
        self.beta = Param('beta', float(beta))
        self.beta.constrain_bounded(0, 1)
        self.link_parameter(self.beta, index=index)
        self.update_model(True)

    def chash(self, X):
        return hash(X.tostring())\
            + hash(self.alpha.tostring()) + hash(self.beta.tostring())

    def update_parts(self, X):
        b = X[:, 0].astype(int)
        z = X[:, 1]
        l = X[:, 2]
        t = X[:, 3]
        opz = 1. + z
        if self.hash != self.chash(X):
            self.sum_mf = np.zeros_like(t)
            self.sum_beta = np.zeros_like(t)
            self.sum_alpha = np.zeros_like(t)
            self.sum_t = np.zeros_like(t)
            self.sum_ell = np.zeros_like(t)
            for i in range(self.numCoefs):
                amp, mu, sig = self.fcoefs_amp[b, i],\
                               self.fcoefs_mu[b, i],\
                               self.fcoefs_sig[b, i]
                alphat = self.alpha * (t - self.beta)
                term1 = (mu * opz - alphat * sig**2) /\
                    (1.41421356237 * sig * opz)
                term2 = alphat * (self.lambdaRef - mu/opz +
                                  alphat*(sig/opz)**2/2)
                expterm = np.exp(term2)
                erfterm = (1 + erf(term1))

                self.sum_mf += amp * erfterm * expterm *\
                    self.sqrthalfpi * sig
                self.sum_ell += amp * erfterm * expterm *\
                    self.sqrthalfpi * sig

                Dterm1 = (self.alpha *
                          np.exp(-((-self.alpha*sig**2*(t - self.beta) +
                                    mu*opz)**2 /
                                   (2*sig**2*opz**2))) *
                          self.sqrthalfpi*sig) / opz
                Dterm2 = (self.alpha**2*self.beta*sig**2*(t-self.beta)**2 /
                          (2*opz**2) +
                          self.alpha * (self.lambdaRef +
                          (self.alpha*sig**2*(t-self.beta)) /
                           (2*opz**2) - mu/opz))
                self.sum_t += amp * expterm * Dterm1 *\
                    self.sqrthalfpi * sig
                self.sum_t += amp * erfterm * expterm *\
                    Dterm2 * self.sqrthalfpi * sig

                Dterm1 = np.sqrt(2/np.pi) * sig * (t-self.beta) / opz *\
                    np.exp(-0.5*((mu*opz - alphat * sig**2 * (t-self.beta)) /
                           sig / opz)**2)
                Dterm2 = (t-self.beta) * (self.lambdaRef - mu/opz +
                                          alphat*(sig/opz)**2/2)\
                    + self.alpha*((t-self.beta)*sig/opz)**2/2
                self.sum_alpha += amp * expterm * Dterm1 *\
                    self.sqrthalfpi * sig
                self.sum_alpha += amp * erfterm * expterm *\
                    Dterm2 * self.sqrthalfpi * sig

                Dterm1 = np.sqrt(2/np.pi) * sig * self.alpha / opz *\
                    np.exp(-0.5*((mu*opz - alphat * sig**2 * (t-self.beta)) /
                           sig / opz)**2)
                Dterm2 = - self.alpha * (self.lambdaRef - mu/opz +
                                         alphat*(sig/opz)**2/2)\
                    - self.alpha**2*((t-self.beta)*sig/opz)**2/2
                self.sum_beta += amp * expterm * Dterm1 *\
                    self.sqrthalfpi * sig
                self.sum_beta += amp * erfterm * expterm *\
                    Dterm2 * self.sqrthalfpi * sig

            self.hash = self.chash(X)

    def f(self, X):
        b = X[:, 0].astype(int)
        z = X[:, 1]
        l = X[:, 2]
        t = X[:, 3]
        opz = 1. + z
        self.update_parts(X)
        fac = l*opz/self.fourpi/self.DL_z(z)**2.0/self.g_AB/self.norms[b]
        return (fac * self.sum_mf).reshape((-1, 1))

    def gradients_X(self, dL_dm, X):
        grad = np.zeros_like(X)
        b = X[:, 0].astype(int)
        z = X[:, 1]
        l = X[:, 2]
        t = X[:, 3]
        opz = 1 + z
        self.update_parts(X)
        fac = opz/self.fourpi/self.DL_z(z)**2.0/self.g_AB/self.norms[b]
        grad[:, 2] = fac * self.sum_ell  # ell
        grad[:, 3] = l * fac * self.sum_t  # t
        # TODO: implement z gradient
        return dL_dm * grad

    def update_gradients(self, dL_dF, X):
        b = X[:, 0].astype(int)
        z = X[:, 1]
        l = X[:, 2]
        t = X[:, 3]
        opz = 1 + z
        self.update_parts(X)
        fac = 1. / self.fourpi / self.DL_z(z)**2.0 / self.g_AB / self.norms[b]
        self.alpha.gradient = np.dot(dL_dF.T, opz * l * fac * self.sum_alpha)
        ind = t > 0
        if ind.sum() > 0:
            self.beta.gradient = np.dot(dL_dF[ind].T,
                                        (opz * l * fac * self.sum_beta)[ind])


class Photoz_kernel(Kern):
    """
    Photoz kernel based on RBF kernel.
    """
    def __init__(self, fcoefs_amp, fcoefs_mu, fcoefs_sig,
                 lines_mu, lines_sig,
                 var_C, var_L, alpha_C, alpha_L, alpha_T,
                 g_AB=1.0, DL_z=None, name='photoz_kern',
                 redshiftGrid=None,
                 use_interpolators=True):
        """ Constructor."""
        # Call standard Kern constructor with 3 dimensions (t, b and z).
        super(Photoz_kernel, self).__init__(4, None, name)
        # If luminosity_distance function not provided, use approximation
        if DL_z is None:
            self.DL_z = approx_DL()
        else:
            self.DL_z = DL_z
        # Store arrays of coefficients.
        if redshiftGrid is None:
            self.redshiftGrid = np.linspace(0, 3, num=120)
        else:
            self.redshiftGrid = redshiftGrid
        self.nz = self.redshiftGrid.size
        self.use_interpolators = use_interpolators
        self.g_AB = g_AB
        self.fourpi = 4 * np.pi
        self.lines_mu = copy(np.array(lines_mu))
        self.lines_sig = copy(np.array(lines_sig))
        self.numLines = lines_mu.size
        assert fcoefs_amp.shape[0] == fcoefs_mu.shape[0] and\
            fcoefs_amp.shape[0] == fcoefs_sig.shape[0]
        self.fcoefs_amp = copy(np.array(fcoefs_amp))
        self.fcoefs_mu = copy(np.array(fcoefs_mu))
        self.fcoefs_sig = copy(np.array(fcoefs_sig))
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
        self.var_C.constrain_positive()
        self.var_L.constrain_positive()
        self.alpha_C.constrain_positive()
        self.alpha_L.constrain_positive()
        self.alpha_T.constrain_positive()
        self.link_parameters(self.var_C, self.var_L,
                             self.alpha_C, self.alpha_L, self.alpha_T)
        self.Thashd = 0
        self.BZhashd = 0
        self.Thash = 0
        self.T2hash = 0
        self.BZhash = 0
        self.BZ2hash = 0
        self.CLhash = 0
        # TODO: addd more realistic constraints?

    def set_alpha_C(self, alpha_C):
        """Set alpha_C"""
        self.update_model(False)
        index = self.alpha_C._parent_index_
        self.unlink_parameter(self.alpha_C)
        self.alpha_C = Param('alpha_C', float(alpha_C))
        self.alpha_C.constrain_positive()
        self.link_parameter(self.alpha_C, index=index)
        self.update_model(True)

    def set_alpha_L(self, alpha_L):
        """Set alpha_L"""
        self.update_model(False)
        index = self.alpha_L._parent_index_
        self.unlink_parameter(self.alpha_L)
        self.alpha_L = Param('alpha_L', float(alpha_L))
        self.alpha_L.constrain_positive()
        self.link_parameter(self.alpha_L, index=index)
        self.update_model(True)

    def set_var_C(self, var_C):
        """Set var_C"""
        self.update_model(False)
        index = self.var_C._parent_index_
        self.unlink_parameter(self.var_C)
        self.var_C = Param('var_C', float(var_C))
        self.var_C.constrain_positive()
        self.link_parameter(self.var_C, index=index)
        self.update_model(True)

    def set_var_L(self, var_L):
        """Set var_L"""
        self.update_model(False)
        index = self.var_L._parent_index_
        self.unlink_parameter(self.var_L)
        self.var_L = Param('var_L', float(var_L))
        self.var_L.constrain_positive()
        self.link_parameter(self.var_L, index=index)
        self.update_model(True)

    def set_alpha_T(self, alpha_T):
        """Set alpha_T"""
        self.update_model(False)
        index = self.alpha_T._parent_index_
        self.unlink_parameter(self.alpha_T)
        self.alpha_T = Param('alpha_T', float(alpha_T))
        self.alpha_T.constrain_positive()
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
        l1 = X[:, 2]
        self.update_kernelparts_diag(X)
        prefac = self.KTd * self.Zprefacd**2 * l1**2
        self.var_C.gradient = np.sum(dL_dKdiag * prefac * self.KCd)
        self.var_L.gradient = np.sum(dL_dKdiag * prefac * self.KLd)
        self.alpha_C.gradient = np.sum(dL_dKdiag * self.var_C *
                                       prefac * self.D_alpha_Cd)
        self.alpha_L.gradient = np.sum(dL_dKdiag * self.var_L *
                                       prefac * self.D_alpha_Ld)
        self.alpha_T.gradient = 0

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None:
            X2 = X
        l1 = X[:, 2]
        l2 = X2[:, 2]
        self.update_kernelparts(X, X2)
        prefac = self.KT * self.Zprefac**2 * l1[:, None] * l2[None, :]
        self.var_C.gradient = np.sum(dL_dK * prefac * self.KC)
        self.var_L.gradient = np.sum(dL_dK * prefac * self.KL)
        self.alpha_C.gradient\
            = np.sum(dL_dK * self.var_C * prefac * self.D_alpha_C)
        self.alpha_L.gradient\
            = np.sum(dL_dK * self.var_L * prefac * self.D_alpha_L)
        self.alpha_T.gradient\
            = np.sum(dL_dK * (X[:, 3:4]-X2[None, :, 3])**2 / self.alpha_T**3 *
                     prefac * (self.var_C * self.KC + self.var_L * self.KL))

    def Kdiag(self, X):
        l1 = X[:, 2]
        self.update_kernelparts_diag(X)
        return self.KTd * self.Zprefacd**2 * l1**2 *\
            (self.var_C * self.KCd + self.var_L * self.KLd)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        l1 = X[:, 2]
        l2 = X2[:, 2]
        self.update_kernelparts(X, X2)
        return self.KT * self.Zprefac**2 * l1[:, None] * l2[None, :] *\
            (self.var_C * self.KC + self.var_L * self.KL)

    def gradients_X(self, dL_dK, X, X2=None):
        if X2 is None:
            X2 = X
        self.update_kernelparts(X, X2)
        tmp = dL_dK * self.KT * self.Zprefac**2 * X[:, 2:3] * X2[None, :, 2] *\
            (self.var_C*self.KC + self.var_L*self.KL)
        grad = np.zeros(X.shape, dtype=np.float64)
        grad[:, 2] = np.sum(tmp / X[:, 2:3], axis=1)  # ell
        tempfull = - tmp * (X[:, 3:4] - X2[None, :, 3]) / self.alpha_T**2
        grad[:, 3] = np.sum(tempfull, axis=1)  # t
        # TODO: add kernel derivatives with respect to redshift
        return grad

    def gradients_X_diag(self, dL_dKdiag, X):
        # TODO: speed up diagonal gradients
        return self.gradients_X(dL_dKdiag, X)

    def cThash(self, X):
        return hash(X[:, 3].tostring()) + hash(self.alpha_T.tostring())

    def cCLhash(self):
        return hash(self.alpha_C.tostring()) + hash(self.alpha_L.tostring())

    def cBZhash(self, X):
        return hash(X[:, 0:2].tostring())

    def construct_interpolators(self):
        bands = np.arange(self.numBands).astype(int)
        fzgrid = 1 + self.redshiftGrid
        ts = (self.numBands, self.numBands, self.nz, self.nz)
        self.KC_grid, self.KL_grid = np.zeros(ts), np.zeros(ts)
        self.D_alpha_C_grid, self.D_alpha_L_grid, self.D_alpha_z_grid\
            = np.zeros(ts), np.zeros(ts), np.zeros(ts)
        for b1 in range(self.numBands):
            for b2 in range(self.numBands):
                b1_grid = np.repeat(b1, self.nz).astype(int)
                b2_grid = np.repeat(b2, self.nz).astype(int)
                kernelparts(self.nz, self.nz, self.numCoefs, self.numLines,
                            self.alpha_C, self.alpha_L,
                            self.fcoefs_amp, self.fcoefs_mu, self.fcoefs_sig,
                            self.lines_mu[:self.numLines],
                            self.lines_sig[:self.numLines],
                            self.norms,
                            b1_grid, fzgrid, b2_grid, fzgrid,
                            True,
                            self.KL_grid[b1, b2, :, :],
                            self.KC_grid[b1, b2, :, :],
                            self.D_alpha_C_grid[b1, b2, :, :],
                            self.D_alpha_L_grid[b1, b2, :, :],
                            self.D_alpha_z_grid[b1, b2, :, :])

        bands = np.arange(self.numBands).astype(int)
        fzgrid = 1 + self.redshiftGrid
        self.KL_diag_interp = np.empty(self.numBands, dtype=interp1d)
        self.KC_diag_interp = np.empty(self.numBands, dtype=interp1d)
        self.D_alpha_C_diag_interp = np.empty(self.numBands, dtype=interp1d)
        self.D_alpha_L_diag_interp = np.empty(self.numBands, dtype=interp1d)
        for b1 in range(self.numBands):
            ts = (self.nz, )
            KC_grid, KL_grid = np.zeros(ts), np.zeros(ts)
            D_alpha_C_grid, D_alpha_L_grid, D_alpha_z_grid\
                = np.zeros(ts), np.zeros(ts), np.zeros(ts)
            b1_grid = np.repeat(b1, self.nz).astype(int)
            kernelparts_diag(self.nz, self.numCoefs, self.numLines,
                             self.alpha_C, self.alpha_L,
                             self.fcoefs_amp, self.fcoefs_mu,
                             self.fcoefs_sig,
                             self.lines_mu[:self.numLines],
                             self.lines_sig[:self.numLines],
                             self.norms,
                             b1_grid, fzgrid,
                             True,
                             KL_grid,
                             KC_grid,
                             D_alpha_C_grid,
                             D_alpha_L_grid)
            self.KL_diag_interp[b1] = interp1d(fzgrid, KL_grid,
                                               kind=kind,
                                               assume_sorted=True)
            self.KC_diag_interp[b1] = interp1d(fzgrid, KC_grid,
                                               kind=kind,
                                               assume_sorted=True)
            self.D_alpha_C_diag_interp[b1] = interp1d(fzgrid, D_alpha_C_grid,
                                                      kind=kind,
                                                      assume_sorted=True)
            self.D_alpha_L_diag_interp[b1] = interp1d(fzgrid, D_alpha_L_grid,
                                                      kind=kind,
                                                      assume_sorted=True)

        self.CLhash = self.cCLhash()

    def update_kernelparts(self, X, X2=None):
        if X2 is None:
            X2 = X
        NO1, NO2 = X.shape[0], X2.shape[0]
        b1 = self.roundband(X[:, 0])
        b2 = self.roundband(X2[:, 0])
        fz1 = 1 + X[:, 1]
        fz2 = 1 + X2[:, 1]
        fzgrid = 1 + self.redshiftGrid

        if self.BZhash != self.cBZhash(X)\
            or self.BZ2hash != self.cBZhash(X2)\
                or self.CLhash != self.cCLhash():

            self.KL, self.KC, self.D_alpha_C, self.D_alpha_L, self.D_alpha_z =\
                np.zeros((NO1, NO2)), np.zeros((NO1, NO2)),\
                np.zeros((NO1, NO2)), np.zeros((NO1, NO2)),\
                np.zeros((NO1, NO2))

            if self.use_interpolators:

                if self.CLhash != self.cCLhash():
                    self.construct_interpolators()
                    self.CLhash = self.cCLhash()

                p1s = np.zeros(NO1, dtype=int)
                p2s = np.zeros(NO2, dtype=int)
                find_positions(NO1, self.nz, fz1, p1s, fzgrid)
                find_positions(NO2, self.nz, fz2, p2s, fzgrid)

                kernel_parts_interp(NO1, NO2,
                                    self.KC,
                                    b1, fz1, p1s,
                                    b2, fz2, p2s,
                                    fzgrid, self.KC_grid)
                kernel_parts_interp(NO1, NO2,
                                    self.D_alpha_C,
                                    b1, fz1, p1s,
                                    b2, fz2, p2s,
                                    fzgrid, self.D_alpha_C_grid)

                if self.numLines > 0:
                    kernel_parts_interp(NO1, NO2,
                                        self.KL,
                                        b1, fz1, p1s,
                                        b2, fz2, p2s,
                                        fzgrid, self.KL_grid)
                    kernel_parts_interp(NO1, NO2,
                                        self.D_alpha_L,
                                        b1, fz1, p1s,
                                        b2, fz2, p2s,
                                        fzgrid, self.D_alpha_L_grid)

            else:  # not use interpolators

                kernelparts(NO1, NO2, self.numCoefs, self.numLines,
                            self.alpha_C, self.alpha_L,
                            self.fcoefs_amp, self.fcoefs_mu, self.fcoefs_sig,
                            self.lines_mu[:self.numLines],
                            self.lines_sig[:self.numLines],
                            self.norms, b1, fz1, b2, fz2,
                            True, self.KL, self.KC,
                            self.D_alpha_C, self.D_alpha_L, self.D_alpha_z)

            self.Zprefac = (1+X[:, 1:2]) * (1+X2[None, :, 1]) /\
                (self.fourpi * self.g_AB * self.DL_z(X[:, 1:2]) *
                 self.DL_z(X2[None, :, 1]))
            self.BZhash = self.cBZhash(X)
            self.BZ2hash = self.cBZhash(X2)

        if self.Thash != self.cThash(X) or self.T2hash != self.cThash(X2):
            self.KT = np.exp(-0.5*pow((X[:, 3:4]-X2[None, :, 3]) /
                                      self.alpha_T, 2))
            self.Thash = self.cThash(X)
            self.T2hash = self.cThash(X2)

    def update_kernelparts_diag(self, X):
        NO1 = X.shape[0]
        b1 = X[:, 0].astype(int)
        fz1 = (1.+X[:, 1])
        if self.BZhashd != self.cBZhash(X):

            self.KLd, self.KCd = np.zeros((NO1,)), np.zeros((NO1,))
            self.D_alpha_Cd, self.D_alpha_Ld =\
                np.zeros((NO1,)), np.zeros((NO1,))

            if self.use_interpolators:

                if self.CLhash != self.cCLhash():
                    self.construct_interpolators()
                    self.CLhash = self.cCLhash()

                for i1 in range(self.numBands):
                    ind1 = np.where(b1 == i1)[0]
                    fz1 = 1 + X[ind1, 1]
                    is1 = np.argsort(fz1)
                    if ind1.size > 0:
                        self.KLd[ind1[is1]] =\
                            self.KL_diag_interp[i1](fz1[is1])
                        self.KCd[ind1[is1]] =\
                            self.KC_diag_interp[i1](fz1[is1])
                        self.D_alpha_Cd[ind1[is1]] =\
                            self.D_alpha_C_diag_interp[i1](fz1[is1])
                        self.D_alpha_Ld[ind1[is1]] =\
                            self.D_alpha_L_diag_interp[i1](fz1[is1])

            else:  # not use interpolators
                fz1 = 1 + X[:, 1]
                kernelparts_diag(self.nz, self.numCoefs, self.numLines,
                                 self.alpha_C, self.alpha_L,
                                 self.fcoefs_amp, self.fcoefs_mu,
                                 self.fcoefs_sig,
                                 self.lines_mu[:self.numLines],
                                 self.lines_sig[:self.numLines],
                                 self.norms, b1, fz1,
                                 True, self.KLd, self.KCd,
                                 self.D_alpha_Cd, self.D_alpha_Ld)

            self.Zprefacd = (1.+X[:, 1])**2 /\
                (self.fourpi * self.g_AB * self.DL_z(X[:, 1])**2)
            self.BZhashd = self.cBZhash(X)

        if self.Thashd != self.cThash(X):
            self.KTd = np.ones((X.shape[0],))
            self.Thashd = self.cThash(X)
