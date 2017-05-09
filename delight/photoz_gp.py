# -*- coding: utf-8 -*-

import numpy as np
from copy import copy
import scipy.linalg
from scipy.optimize import minimize
from scipy.interpolate import interp1d

from delight.utils import approx_DL, scalefree_flux_likelihood, symmetrize
from delight.photoz_kernels import *

log_2_pi = np.log(2*np.pi)

__all__ = ["PhotozGP", "PhotozGP_SN"]


class PhotozGP_SN:
    """
    Photo-z Gaussian process, with physical kernel and mean function.

    Args:
        bandCoefAmplitudes: ``numpy.array`` of size (numBands, numCoefs)
            describint the amplitudes of the Gaussians approximating the
            photometric filters.
        bandCoefPositions: ``numpy.array`` of size (numBands, numCoefs)
            describint the positions of the Gaussians approximating the
            photometric filters.
        bandCoefWidths: ``numpy.array`` of size (numBands, numCoefs)
            describint the widths of the Gaussians approximating the
            photometric filters.
        lines_pos: ``numpy.array`` of SED line positions
        lines_width: ``numpy.array`` of SED line widths
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
        redshiftGridGP: redshift grid (array) for computing the GP.
        use_interpolators (Optional): ``boolean`` indicating if the GP
            should be used for all predictions,
            or if an interpolation scheme should be used (default: ``True``)
        lambdaRef (Optional): Pivot space for the SEDs
            (``float``, default: ``4.5e3``)
        g_AB (Optional): AB photometric normalization constant
            (``float``, default: ``1.0``)
    """
    def __init__(self,
                 bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
                 lines_pos, lines_width,
                 var_C, var_L, alpha_T, alpha_C, alpha_L,
                 redshiftGridGP,
                 use_interpolators=True,
                 lambdaRef=4.5e3,
                 g_AB=1.0):

        DL = approx_DL()
        self.bands = np.arange(bandCoefAmplitudes.shape[0])
        self.kernel = Photoz_SN_kernel(
            bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
            lines_pos, lines_width, var_C, var_L, alpha_T, alpha_C, alpha_L,
            g_AB=g_AB, DL_z=DL, redshiftGrid=redshiftGridGP,
            use_interpolators=use_interpolators)
        self.redshiftGridGP = redshiftGridGP

    def setData(self, X, Y, Yvar):
        """
        Set data content for the Gaussian process.

        Args:
            X: array of size (nobj, 4) containing the GP inputs.
                The column order is band, redshift, and luminosity.
            Y: array of size (nobj, 1) containing the GP outputs.
                Contains the photometric fluxes corresponding to the inputs.
            Yvar: array of size (nobj, 1) containing the GP outputs.
                Contains the flux variances corresponding to the inputs.
        """
        self.X = X
        self.Y = Y.reshape((-1, 1))
        self.Yvar = Yvar.reshape((-1, 1))
        self.KXX = self.kernel.K(self.X)
        self.A = self.KXX + np.diag(self.Yvar.flatten())
        sign, self.logdet = np.linalg.slogdet(self.A)
        self.logdet *= sign
        self.L = scipy.linalg.cholesky(self.A, lower=True)
        self.D = 1*self.Y
        self.beta = scipy.linalg.cho_solve((self.L, True), self.D)

    def margLike(self):
        """
        Returns marginalized likelihood of GP.
        """
        return 0.5 * np.sum(self.beta * self.D) +\
            0.5 * self.logdet + 0.5 * self.D.size * log_2_pi

    def predict(self, x_pred, diag=True):
        """
        Raw way to predict outputs with the GP.
        Args:
            x_pred: input array of size (nobj, 4).
                The column order is band, redshift, and luminosity.
            diag (Optional): return the predicted variance on the diagonal only
        """
        assert x_pred.shape[1] == 4
        KXXp = self.kernel.K(x_pred, self.X)
        v = scipy.linalg.cho_solve((self.L, True), KXXp.T)
        if diag:
            y_pred_cov = self.kernel.Kdiag(x_pred)
            for i in range(x_pred.shape[0]):
                y_pred_cov[i] -= KXXp[i, :].dot(v[:, i])
        else:
            KXpXp = self.kernel.K(x_pred)
            v = scipy.linalg.cho_solve((self.L, True), KXXp.T)
            y_pred_cov = KXpXp - KXXp.dot(v)
        y_pred = np.dot(KXXp, self.beta)
        return y_pred, y_pred_cov


class PhotozGP:
    """
    Photo-z Gaussian process, with physical kernel and mean function.

    Args:
        f_mod_interp: grid of interpolators of size (num templates, num bands)
            called as ``f_mod_interp[it, ib](z)``
        bandCoefAmplitudes: ``numpy.array`` of size (numBands, numCoefs)
            describint the amplitudes of the Gaussians approximating the
            photometric filters.
        bandCoefPositions: ``numpy.array`` of size (numBands, numCoefs)
            describint the positions of the Gaussians approximating the
            photometric filters.
        bandCoefWidths: ``numpy.array`` of size (numBands, numCoefs)
            describint the widths of the Gaussians approximating the
            photometric filters.
        lines_pos: ``numpy.array`` of SED line positions
        lines_width: ``numpy.array`` of SED line widths
        var_C: GP variance for SED continuum correlations.
            Should be a ``float`, preferably between 1e-3 and 1e2.
        var_L: GP variance for SED line correlations.
            Should be a ``float`, preferably between 1e-3 and 1e2.
        alpha_C: GP lengthscale for smoothness of SED continuum correlations.
            Should be a ``float`, preferably between 1e1 and 1e4.
        alpha_L: GP lengthscale for smoothness of SED line correlations.
            Should be a ``float`, preferably between 1e1 and 1e4.
        redshiftGridGP: redshift grid (array) for computing the GP.
        use_interpolators (Optional): ``boolean`` indicating if the GP
            should be used for all predictions,
            or if an interpolation scheme should be used (default: ``True``)
        lambdaRef (Optional): Pivot space for the SEDs
            (``float``, default: ``4.5e3``)
        g_AB (Optional): AB photometric normalization constant
            (``float``, default: ``1.0``)
    """
    def __init__(self,
                 f_mod_interp,
                 bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
                 lines_pos, lines_width,
                 var_C, var_L, alpha_C, alpha_L,
                 redshiftGridGP,
                 use_interpolators=True,
                 lambdaRef=4.5e3,
                 g_AB=1.0):

        DL = approx_DL()
        self.bands = np.arange(bandCoefAmplitudes.shape[0])
        if isinstance(f_mod_interp, int):
            self.mean_fct = None
            self.nt = f_mod_interp
        else:
            self.mean_fct = Photoz_linear_sed_basis(f_mod_interp)
            self.nt = f_mod_interp.shape[0]
        # self.mean_fct = Photoz_mean_function(
        #    alpha, bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
        #    g_AB=g_AB, lambdaRef=lambdaRef, DL_z=DL)
        self.kernel = Photoz_kernel(
            bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
            lines_pos, lines_width, var_C, var_L, alpha_C, alpha_L,
            g_AB=g_AB, DL_z=DL, redshiftGrid=redshiftGridGP,
            use_interpolators=use_interpolators)
        self.redshiftGridGP = redshiftGridGP

    def setData(self, X, Y, Yvar, bestType=None):
        """
        Set data content for the Gaussian process.

        Args:
            X: array of size (nobj, 3) containing the GP inputs.
                The column order is band, redshift, and luminosity.
            Y: array of size (nobj, 1) containing the GP outputs.
                Contains the photometric fluxes corresponding to the inputs.
            Yvar: array of size (nobj, 1) containing the GP outputs.
                Contains the flux variances corresponding to the inputs.
        """
        self.X = X
        self.Y = Y.reshape((-1, 1))
        self.Yvar = Yvar.reshape((-1, 1))
        if isinstance(self.mean_fct, Photoz_mean_function):
            mf = self.mean_fct.f(X)
        else:
            mf = None
        self.KXX = self.kernel.K(self.X)
        self.A = self.KXX + np.diag(self.Yvar.flatten())
        sign, self.logdet = np.linalg.slogdet(self.A)
        self.logdet *= sign
        self.L = scipy.linalg.cholesky(self.A, lower=True)
        self.D = 1*self.Y
        self.betas = np.zeros(self.nt)
        if self.mean_fct is not None:  # set mean fct to best fit template
            self.bestType = bestType
            self.betas[bestType] = 1.0
            which = np.where(self.betas > 0)[0]
            hx = self.mean_fct.f(self.X, which=which).T
            hx[~np.isfinite(hx)] = 0
            self.D -= np.dot(hx.T, self.betas)[:, None]
        self.beta = scipy.linalg.cho_solve((self.L, True), self.D)

    def getCore(self):
        """
        Returns core matrices, useful to re-use the GP elsewhere.
        The core matrices contain stuff that doesn't need to be recomputed.
        Returns array of size numTemplates+numBands+numBands*(numBands+1)//2.
        """
        B = self.D.size
        nt = self.betas.size
        halfL = self.L[np.tril_indices(B)]
        flatarray = np.zeros((nt + B + B*(B+1)//2, ))
        flatarray[0:nt] = self.betas
        flatarray[nt:nt+B*(B+1)//2] = halfL
        flatarray[nt+B*(B+1)//2:] = self.D.ravel()
        return flatarray

    def setCore(self, X, B, nt, flatarray):
        """
        Set the GP core matrices.
        The core matrices contain stuff that doesn't need to be recomputed.

        Args:
            flatarray: size numTemplates+numBands+numBands*(numBands+1)//2.
            X: the GP inputs, of size (nobj, 3).
            B: ``float`` the number of bands.
            nt: ``float`` the number of templates.

        """
        self.X = X
        self.betas = flatarray[0:nt]
        self.bestType = int(np.argmax(self.betas))
        self.D = flatarray[nt+B*(B+1)//2:].reshape((-1, 1))
        self.L = np.zeros((B, B))
        self.L[np.tril_indices(B)] = flatarray[nt:nt+B*(B+1)//2]
        self.beta = scipy.linalg.cho_solve((self.L, True), self.D)

    def margLike(self):
        """
        Returns marginalized likelihood of GP.
        """
        return 0.5 * np.sum(self.beta * self.D) +\
            0.5 * self.logdet + 0.5 * self.D.size * log_2_pi

    def predict(self, x_pred, diag=True):
        """
        Raw way to predict outputs with the GP.
        Args:
            x_pred: input array of size (nobj, 3).
                The column order is band, redshift, and luminosity.
            diag (Optional): return the predicted variance on the diagonal only
        """
        assert x_pred.shape[1] == 3
        KXXp = self.kernel.K(x_pred, self.X)
        v = scipy.linalg.cho_solve((self.L, True), KXXp.T)
        if diag:
            y_pred_cov = self.kernel.Kdiag(x_pred)
            for i in range(x_pred.shape[0]):
                y_pred_cov[i] -= KXXp[i, :].dot(v[:, i])
        else:
            KXpXp = self.kernel.K(x_pred)
            v = scipy.linalg.cho_solve((self.L, True), KXXp.T)
            y_pred_cov = KXpXp - KXXp.dot(v)
        if isinstance(self.mean_fct, Photoz_mean_function):
            mf = self.mean_fct.f(x_pred)
        elif isinstance(self.mean_fct, Photoz_linear_sed_basis):
            which = np.where(self.betas > 0)[0]
            hx_pred = self.mean_fct.f(x_pred, which=which).T
            mf = np.dot(hx_pred.T, self.betas)[:, None]
        else:
            mf = 0
        y_pred = np.dot(KXXp, self.beta) + mf
        return y_pred, y_pred_cov

    def predictAndInterpolate(self, redshiftGrid, ell=1.0, z=None):
        """
        Convenient way to get flux predictions on a redshift/band grid.
        First compute on the coarce GP grid and then interpolate on finer grid.
        ell should be set to reference luminosity used in the GP.
        z is an additional redshift to compute predictions at.

        Args:
            redshiftGrid: array to get predictions for.
                The bands are automatically set.
            ell (Optional): to change the luminosity scaling if necessary.
            z (Optional): add an additional point to the redshift Grid.
        """
        numBands = self.bands.size
        numZGP = self.redshiftGridGP.size
        redshiftGridGP_loc = 1 * self.redshiftGridGP
        if z is not None:
            zloc = np.abs(z - redshiftGridGP_loc).argmin()
            redshiftGridGP_loc[zloc] = z
        xv, yv = np.meshgrid(redshiftGridGP_loc, self.bands,
                             sparse=False, indexing='xy')
        X_pred = np.ones((numBands*numZGP, 3))
        X_pred[:, 0] = yv.flatten()
        X_pred[:, 1] = xv.flatten()
        X_pred[:, 2] = ell
        y_pred, y_pred_cov = self.predict(X_pred, diag=True)
        model_mean = np.zeros((redshiftGrid.size, numBands))
        model_var = np.zeros((redshiftGrid.size, numBands))
        for i in range(numBands):
            y_pred_bin = y_pred[i*numZGP:(i+1)*numZGP].ravel()
            y_var_bin = y_pred_cov[i*numZGP:(i+1)*numZGP].ravel()
            model_mean[:, i] = interp1d(redshiftGridGP_loc,
                                        y_pred_bin,
                                        assume_sorted=True,
                                        copy=False)(redshiftGrid)
            # np.interp(redshiftGrid, redshiftGridGP_loc, y_pred_bin)
            if np.any(y_var_bin <= 0):
                print(z, "band", i, "y_pred_bin",
                      y_pred_bin, "y_var_bin", y_var_bin)
            model_var[:, i] = interp1d(redshiftGridGP_loc,
                                       y_var_bin,
                                       assume_sorted=True,
                                       copy=False)(redshiftGrid)
            #  np.interp(redshiftGrid, redshiftGridGP_loc, y_var_bin)
        # model_covar = np.zeros((redshiftGrid.size, numBands, numBands))
        # for i in range(numBands):
        #    for j in range(numBands):
        #        y_covar_bin =
        # y_pred_fullcov[i*numZGP:(i+1)*numZGP, :][:, j*numZGP:(j+1)*numZGP]
        #        interp_spline =
        # RectBivariateSpline(redshiftGridGP_loc,
        # redshiftGridGP_loc, y_covar_bin)
        #        model_covar[:, i, j] =
        # interp_spline(redshiftGrid, redshiftGrid, grid=False)
        return model_mean, model_var

    def estimateAlphaEll(self):
        """
        (Deprecated)
        Estimate alpha by fitting colours with power law
        then estimate ell by fixing alpha by fitting fluxes with power law.
        """
        X_pred = 1*self.X

        def fun(alpha):
            self.mean_fct.alpha = alpha[0]
            y_pred = self.mean_fct.f(X_pred).ravel()
            y_pred *= np.mean(self.Y) / y_pred.mean()
            chi2 = scalefree_flux_likelihood(self.Y.ravel(),
                                             self.Yvar.ravel(),
                                             y_pred[None, None, :],
                                             returnChi2=True)
            return chi2

        x0 = [0.0]
        z = self.X[0, 1]
        res = minimize(fun, x0, method='L-BFGS-B', tol=1e-9,
                       bounds=[((1+2*z)*-2e-4, 4e-4)])
        if res.success is False or np.abs(res.x[0]) > 1e-2:
            raise Exception("Problem! Optimized alpha is ", res.x[0])
        self.mean_fct.alpha = res.x[0]

        def fun(ell):
            X_pred[:, 2] = ell
            y_pred = self.mean_fct.f(X_pred).ravel()
            chi2s = (self.Y.ravel() - y_pred)**2 / self.Yvar
            return np.sum(chi2s)

        ell = self.X[0, 2]
        x0 = [ell]
        res = minimize(fun, x0, method='L-BFGS-B', tol=1e-9,
                       bounds=[(1e-3*ell, 1e3*ell)])
        # bounds=[(1e-3*ell, 1e3*ell)])
        if res.x[0] < 0:
            raise Exception("Problem! Optimized ell is ", res.x[0])
        # print("alpha optimized:", self.mean_fct.alpha,
        #  "ell optimized:", res.x[0])
        self.X[:, 2] = res.x[0]
        self.setData(self.X, self.Y, self.Yvar)  # Need to recompute core

        return self.mean_fct.alpha, self.X[0, 2]

    def optimizeHyperparamaters(self, x0=None, verbose=False):
        """
        Optimize Hyperparamaters with marglike as objective.
        """
        assert self.kernel.use_interpolators is False
        if x0 is None:
            x0 = [1.0, 1e3]  # V_C, V_L, alpha_C
        res = minimize(self.updateHyperparamatersAndReturnMarglike, x0,
                       method='L-BFGS-B',
                       bounds=[(1e-12, 1e12), (1e2, 1e4)])
        V_C, alpha_C = res.x
        if verbose:
            print("Optimized parameters: ", res.x)
        self.kernel.var_C, self.kernel.var_L = 1*V_C, 1*V_C
        self.kernel.alpha_C, self.kernel.alpha_L = 1*alpha_C, 1*alpha_C

    def updateHyperparamatersAndReturnMarglike(self, pars):
        """
        For optimizing Hyperparamaters with marglike as objective using scipy.
        """
        V_C, alpha_C = pars
        self.kernel.var_C, self.kernel.var_L = 1*V_C, 1*V_C
        self.kernel.alpha_C, self.kernel.alpha_L = 1*alpha_C, 1*alpha_C
        self.KXX = self.kernel.K(self.X)
        self.A = self.KXX + np.diag(self.Yvar.flatten())
        sign, self.logdet = np.linalg.slogdet(self.A)
        self.logdet *= sign
        self.L = scipy.linalg.cholesky(self.A, lower=True)
        self.D = 1*self.Y
        self.betas = np.zeros(self.nt)
        if self.mean_fct is not None:  # set mean fct to best fit template
            # self.bestType = bestType
            # self.betas[bestType] = 1.0
            which = np.where(self.betas > 0)[0]
            hx = self.mean_fct.f(self.X, which=which).T
            self.D -= np.dot(hx.T, self.betas)[:, None]
        self.beta = scipy.linalg.cho_solve((self.L, True), self.D)
        return self.margLike()

    def optimizeAlpha_GP(self):
        """
        (Deprecated)
        Optimize alpha with marglike as objective.
        """
        x0 = 0.0  # [0.0, self.X[0, 2]]
        res = minimize(self.updateAlphaAndReturnMarglike, x0,
                       method='L-BFGS-B', tol=1e-6,
                       bounds=[(-3e-4, 3e-4)])
        # , (1e-3*self.X[0, 2], 1e3*self.X[0, 2])])
        self.mean_fct.alpha = res.x[0]
        # self.X[:, 2] = res.x[1]

    def updateAlphaAndReturnMarglike(self, alpha):
        """
        (Deprecated)
        For optimizing alpha with the marglike as objective using scipy.
        """
        self.mean_fct.alpha = alpha[0]
        self.D = self.Y - self.mean_fct.f(self.X)
        self.beta = scipy.linalg.cho_solve((self.L, True), self.D)
        return self.margLike()
