# -*- coding: utf-8 -*-

import numpy as np
from copy import copy
from scipy.special import erf
import scipy.linalg
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline

from delight.photoz_kernels_cy import kernelparts, kernelparts_diag,\
    kernel_parts_interp
from delight.utils_cy import find_positions
from delight.utils import approx_DL

kind = "linear"


class Photoz_linear_sed_basis():
    """
    Mean function of photoz GP, based on a library of templates.

    Args:
        f_mod_interp: grid of interpolators of size (num templates, num bands)
            called as ``f_mod_interp[it, ib](z)``
    """
    def __init__(self, f_mod_interp):
        """ Constructor."""
        # If luminosity_distance function not provided, use approximation
        self.f_mod_interp = f_mod_interp
        self.alpha = 0
        self.nt, self.nb = f_mod_interp.shape

    def f(self, X, which=None):
        """
        Compute mean function.

        Args:
            X: array of size (nobj, 3) containing the GP inputs.
                The column order is band, redshift, and luminosity.
            which (Optional): array of indices on which to compute the mean
                function. (default: all types in the SED basis).
        """
        b = X[:, 0].astype(int)
        z = X[:, 1]
        l = X[:, 2]
        opz = 1. + z

        if which is None:
            which = range(self.nt)
        hx = np.zeros((X.shape[0], self.nt))
        for it in which:
            for k in range(self.nb):
                sel = b == k
                hx[sel, it] = self.f_mod_interp[it, k](z[sel])

        return l[:, None] * hx


class Photoz_mean_function():
    """
    (Deprecated)
    Mean function of photoz GP, based on a power law model.
    """
    def __init__(self,
                 alpha,
                 fcoefs_amp, fcoefs_mu, fcoefs_sig,
                 g_AB=1.0, lambdaRef=4.5e3, DL_z=None, name='photoz_mf'):
        """ Constructor."""
        # If luminosity_distance function not provided, use approximation
        if DL_z is None:
            self.DL_z = approx_DL()
        else:
            self.DL_z = DL_z
        self.g_AB = g_AB
        assert lambdaRef > 1e2 and lambdaRef < 1e5
        self.lambdaRef = lambdaRef
        self.fourpi = 4 * np.pi
        self.sqrthalfpi = np.sqrt(np.pi/2)
        self.alpha = alpha
        assert fcoefs_amp.shape[0] == fcoefs_mu.shape[0] and\
            fcoefs_amp.shape[0] == fcoefs_sig.shape[0]
        self.fcoefs_amp = np.array(fcoefs_amp)
        self.fcoefs_mu = np.array(fcoefs_mu)
        self.fcoefs_sig = np.array(fcoefs_sig)
        self.numCoefs = fcoefs_amp.shape[1]
        self.norms = np.sqrt(2*np.pi)\
            * np.sum(self.fcoefs_amp * self.fcoefs_sig / self.fcoefs_mu, axis=1)
        self.fcoefs_amp *= self.fcoefs_mu

    def f(self, X):
        """
        Compute mean function.
        """
        b = X[:, 0].astype(int)
        z = X[:, 1]
        l = X[:, 2]
        opz = 1. + z
        lambdaRef = self.lambdaRef

        def IanddI(alpha, opz, mu, sig, lam):
            T1 = (alpha*sig**2 - mu*opz + lam*opz**2) / (np.sqrt(2)*sig*opz)
            T2 = alpha/2/opz**2*(alpha*sig**2 - 2*mu*opz + 2*lambdaRef*opz**2)
            erfT1 = erf(T1)
            expT2 = np.exp(T2)
            I = self.sqrthalfpi * sig / opz * erfT1 * expT2
            dIdalpha = 0
            return I, dIdalpha

        self.sum_mf = np.zeros_like(l)
        for i in range(self.numCoefs):
            amp, mu, sig = self.fcoefs_amp[b, i],\
                           self.fcoefs_mu[b, i],\
                           self.fcoefs_sig[b, i]
            I1, dIdalpha1 = IanddI(self.alpha, opz, mu, sig, 1e8)
            I2, dIdalpha2 = IanddI(self.alpha, opz, mu, sig, 0)
            self.sum_mf += amp * (I1 - I2)

        fac = l*opz**2/self.fourpi/self.DL_z(z)**2.0/self.g_AB/self.norms[b]
        return (fac * self.sum_mf).reshape((-1, 1))


class Photoz_kernel:
    """
    Photoz kernel based on RBF kernel in SED space.

    Args:
        fcoefs_amp: ``numpy.array`` of size (numBands, numCoefs)
            describint the amplitudes of the Gaussians approximating the
            photometric filters.
        fcoefs_mu: ``numpy.array`` of size (numBands, numCoefs)
            describint the positions of the Gaussians approximating the
            photometric filters.
        fcoefs_sig: ``numpy.array`` of size (numBands, numCoefs)
            describint the widths of the Gaussians approximating the
            photometric filters.
        lines_mu: ``numpy.array`` of SED line positions
        lines_sig: ``numpy.array`` of SED line widths
        var_C: GP variance for SED continuum correlations.
            Should be a ``float`, preferably between 1e-3 and 1e2.
        var_L: GP variance for SED line correlations.
            Should be a ``float`, preferably between 1e-3 and 1e2.
        alpha_C: GP lengthscale for smoothness of SED continuum correlations.
            Should be a ``float`, preferably between 1e1 and 1e4.
        alpha_L: GP lengthscale for smoothness of SED line correlations.
            Should be a ``float`, preferably between 1e1 and 1e4.
        lambdaRef (Optional): Pivot space for the SEDs
            (``float``, default: ``4.5e3``)
        g_AB (Optional): AB photometric normalization constant
            (``float``, default: ``1.0``)
        DL_z (Optional): function for computing the luminosity distance
            as a fct of redshift. Default: an analytic approximation.
        redshiftGrid (Optional): redshift grid (array) for computing the GP.
            (default: some fine grid.)
        use_interpolators (Optional): ``boolean`` indicating if the GP
            should be used for all predictions,
            or if an interpolation scheme should be used (default: ``True``)
    """
    def __init__(self,
                 fcoefs_amp, fcoefs_mu, fcoefs_sig,
                 lines_mu, lines_sig,
                 var_C,
                 var_L,
                 alpha_C,
                 alpha_L,
                 g_AB=1.0,
                 DL_z=None,
                 redshiftGrid=None,
                 use_interpolators=True):
        """ Constructor."""
        self.use_interpolators = use_interpolators
        if DL_z is None:
            self.DL_z = approx_DL()
        else:
            self.DL_z = DL_z
        self.g_AB = g_AB
        self.fourpi = 4 * np.pi
        self.lines_mu = np.array(lines_mu)
        self.lines_sig = np.array(lines_sig)
        self.numLines = self.lines_mu.size
        assert fcoefs_amp.shape[0] == fcoefs_mu.shape[0] and\
            fcoefs_amp.shape[0] == fcoefs_sig.shape[0]
        self.fcoefs_amp = fcoefs_amp
        self.fcoefs_mu = fcoefs_mu
        self.fcoefs_sig = fcoefs_sig
        self.numCoefs = fcoefs_amp.shape[1]
        self.numBands = fcoefs_amp.shape[0]
        self.norms = np.sqrt(2*np.pi)\
            * np.sum(self.fcoefs_amp * self.fcoefs_sig, axis=1)
        # Initialize parameters and link them.
        self.var_C = var_C
        self.var_L = var_L
        self.alpha_C = alpha_C
        self.alpha_L = alpha_L
        if redshiftGrid is None:
            self.redshiftGrid = np.linspace(0, 4, num=160)
        else:
            self.redshiftGrid = copy(redshiftGrid)
        self.nz = self.redshiftGrid.size
        self.construct_interpolators()

    def roundband(self, bfloat):
        """
        Convenient fct to cast the last dimension (band index) as integer.
        """
        # In GPy, numpy arrays are type ObsAr, so the values must be extracted.
        b = bfloat.astype(int)
        # Check bounds. This is ok because band indices should never change
        # unless there are tiny numerical errors withint GPy.
        b[b < 0] = 0
        b[b >= self.numBands] = self.numBands - 1
        return b

    def Kdiag(self, X):
        """
        Compute GP kernel on the diagonal only.
        """
        l1 = X[:, 2]
        self.update_kernelparts_diag(X)
        return self.KTd * self.Zprefacd**2 * l1**2 *\
            (self.var_C * self.KCd + self.var_L * self.KLd)

    def K(self, X, X2=None):
        """
        Compute GP kernel, auto or cross depending on whether X2 is set.
        """
        if X2 is None:
            X2 = X
        l1 = X[:, 2]
        l2 = X2[:, 2]
        self.update_kernelparts(X, X2)
        return self.Zprefac**2 * l1[:, None] * l2[None, :] *\
            (self.var_C * self.KC + self.var_L * self.KL)

    def update_kernelparts_diag(self, X):
        """
        Update the precomputed parts of the kernel, on the diagonal only.
        X is an array of size (nobj, 3) containing the GP inputs.
        The column order is band, redshift, and luminosity.
        """
        NO1 = X.shape[0]
        b1 = self.roundband(X[:, 0])
        fz1 = 1 + X[:, 1]
        self.KLd, self.KCd = np.zeros((NO1,)), np.zeros((NO1,))
        self.D_alpha_Cd, self.D_alpha_Ld =\
            np.zeros((NO1,)), np.zeros((NO1,))
        self.KTd = np.ones((NO1,))
        self.Zprefacd = (1.+X[:, 1])**2 /\
            (self.fourpi * self.g_AB * self.DL_z(X[:, 1])**2)

        if self.use_interpolators:

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
            kernelparts_diag(NO1, self.numCoefs, self.numLines,
                             self.alpha_C, self.alpha_L,
                             self.fcoefs_amp, self.fcoefs_mu,
                             self.fcoefs_sig,
                             self.lines_mu[:self.numLines],
                             self.lines_sig[:self.numLines],
                             self.norms, b1, fz1,
                             True, self.KLd, self.KCd,
                             self.D_alpha_Cd, self.D_alpha_Ld)

    def update_kernelparts(self, X, X2=None):
        """
        Update the precomputed parts of the kernel.
        X is an array of size (nobj, 3) containing the GP inputs.
        The column order is band, redshift, and luminosity.
        """
        if X2 is None:
            X2 = X
        NO1, NO2 = X.shape[0], X2.shape[0]
        b1 = self.roundband(X[:, 0])
        b2 = self.roundband(X2[:, 0])
        fz1 = 1 + X[:, 1]
        fz2 = 1 + X2[:, 1]
        fzgrid = 1 + self.redshiftGrid

        self.KL, self.KC, self.D_alpha_C, self.D_alpha_L, self.D_alpha_z =\
            np.zeros((NO1, NO2)), np.zeros((NO1, NO2)),\
            np.zeros((NO1, NO2)), np.zeros((NO1, NO2)),\
            np.zeros((NO1, NO2))

        if self.use_interpolators:

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

    def construct_interpolators(self):
        """
        Construct interpolation scheme for the kernel.
        This significantly speeds up calculations by computing and storing
        the kernel evaluated on a grid, for later interpolation.
        """
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
                                               assume_sorted=True,
                                               bounds_error=False,
                                               fill_value="extrapolate")
            self.KC_diag_interp[b1] = interp1d(fzgrid, KC_grid,
                                               kind=kind,
                                               assume_sorted=True,
                                               bounds_error=False,
                                               fill_value="extrapolate")
            self.D_alpha_C_diag_interp[b1] = interp1d(fzgrid, D_alpha_C_grid,
                                                      kind=kind,
                                                      assume_sorted=True,
                                                      bounds_error=False,
                                                      fill_value="extrapolate")
            self.D_alpha_L_diag_interp[b1] = interp1d(fzgrid, D_alpha_L_grid,
                                                      kind=kind,
                                                      assume_sorted=True,
                                                      bounds_error=False,
                                                      fill_value="extrapolate")


class Photoz_SN_kernel(Photoz_kernel):
    """
    Photoz kernel based on RBF kernel in SED space.

    Args:
        fcoefs_amp: ``numpy.array`` of size (numBands, numCoefs)
            describint the amplitudes of the Gaussians approximating the
            photometric filters.
        fcoefs_mu: ``numpy.array`` of size (numBands, numCoefs)
            describint the positions of the Gaussians approximating the
            photometric filters.
        fcoefs_sig: ``numpy.array`` of size (numBands, numCoefs)
            describint the widths of the Gaussians approximating the
            photometric filters.
        lines_mu: ``numpy.array`` of SED line positions
        lines_sig: ``numpy.array`` of SED line widths
        var_C: GP variance for SED continuum correlations.
            Should be a ``float`, preferably between 1e-3 and 1e2.
        var_L: GP variance for SED line correlations.
            Should be a ``float`, preferably between 1e-3 and 1e2.
        alpha_T: GP lengthscale for smoothness of time correlations.
            Should be a ``float`.
        alpha_C: GP lengthscale for smoothness of SED continuum correlations.
            Should be a ``float`, preferably between 1e1 and 1e4.
        alpha_L: GP lengthscale for smoothness of SED line correlations.
            Should be a ``float`, preferably between 1e1 and 1e4.
        lambdaRef (Optional): Pivot space for the SEDs
            (``float``, default: ``4.5e3``)
        g_AB (Optional): AB photometric normalization constant
            (``float``, default: ``1.0``)
        DL_z (Optional): function for computing the luminosity distance
            as a fct of redshift. Default: an analytic approximation.
        redshiftGrid (Optional): redshift grid (array) for computing the GP.
            (default: some fine grid.)
        use_interpolators (Optional): ``boolean`` indicating if the GP
            should be used for all predictions,
            or if an interpolation scheme should be used (default: ``True``)
    """
    def __init__(self,
                 fcoefs_amp, fcoefs_mu, fcoefs_sig,
                 lines_mu, lines_sig,
                 var_C,
                 var_L,
                 alpha_T,
                 alpha_C,
                 alpha_L,
                 g_AB=1.0,
                 DL_z=None,
                 redshiftGrid=None,
                 use_interpolators=True):
        """ Constructor."""
        self.alpha_T = alpha_T
        super().__init__(
            fcoefs_amp, fcoefs_mu, fcoefs_sig,
            lines_mu, lines_sig, var_C, var_L,
            alpha_C, alpha_L, g_AB, DL_z,
            redshiftGrid, use_interpolators)

    def Kdiag(self, X):
        """
        Compute GP kernel on the diagonal only.
        """
        return super().Kdiag(X[:, 0:3])

    def K(self, X, X2=None):
        """
        Compute GP kernel, auto or cross depending on whether X2 is set.
        """
        if X2 is None:
            X2 = X
        t1 = X[:, 3]
        t2 = X2[:, 3]
        kt = np.exp(-(X[:, None, 3] - X2[None, :, 3])**2/self.alpha_T)
        return kt * super().K(X[:, 0:3], X2[:, 0:3])
