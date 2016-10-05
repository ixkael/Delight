
import numpy as np
from copy import copy
import scipy.linalg
from scipy.optimize import minimize

from delight.utils import approx_DL, scalefree_flux_likelihood, symmetrize
from delight.photoz_kernels import *

log_2_pi = np.log(2*np.pi)


class PhotozGP:
    """
    Photo-z Gaussian process, with physical kernel and mean function.
    """
    def __init__(self, f_mod_interp,
                 bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
                 lines_pos, lines_width, var_C, var_L, alpha_C, alpha_L,
                 redshiftGridGP, use_interpolators=True,
                 lambdaRef=4.5e3, g_AB=1.0):

        DL = approx_DL()
        self.bands = np.arange(bandCoefAmplitudes.shape[0])
        self.mean_fct = Photoz_linear_sed_basis(f_mod_interp)
        #self.mean_fct = Photoz_mean_function(
        #    alpha, bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
        #    g_AB=g_AB, lambdaRef=lambdaRef, DL_z=DL)
        self.kernel = Photoz_kernel(
            bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
            lines_pos, lines_width, var_C, var_L, alpha_C, alpha_L,
            g_AB=g_AB, DL_z=DL, redshiftGrid=redshiftGridGP,
            use_interpolators=use_interpolators)
        self.redshiftGridGP = redshiftGridGP

    def setData(self, X, Y, Yvar):
        """
        Set data content for the Gaussian process
        """
        self.X = X
        self.Y = Y.reshape((-1, 1))
        self.Yvar = Yvar.reshape((-1, 1))
        if isinstance(self.mean_fct, Photoz_mean_function):
            mf = self.mean_fct.f(X)
        elif isinstance(self.mean_fct, Photoz_linear_sed_basis):
            mf = 0
            if False:
                hx = self.mean_fct.f(self.X).T
                def fun(betas):
                    chi2 = (np.dot(hx.T, betas) - self.Y[:, 0])**2 / self.Yvar[:, 0]
                    return chi2.sum()
                ell = self.X[:, 2].mean()
                x0 = np.repeat(ell / self.mean_fct.nt, self.mean_fct.nt)
                res = minimize(fun, x0, method='L-BFGS-B',
                               bounds=[(0, 15*ell)] * self.mean_fct.nt)
                if res.success is False > 1e-2:
                    raise Exception("Problem!", res)
                self.betas = res.x
                mf = np.dot(hx.T, self.betas)[:, None]
        else:
            mf = 0
        self.betas = np.zeros(self.mean_fct.nt)
        self.D = self.Y - mf
        self.KXX = self.kernel.K(self.X)
        self.logdet = np.log(scipy.linalg.det(self.KXX))
        self.A = self.KXX + np.diag(self.Yvar.flatten())
        self.L = scipy.linalg.cholesky(self.A, lower=True)
        self.beta = scipy.linalg.cho_solve((self.L, True), self.D)

    def getCore(self):
        """
        Returns core matrices, useful to re-use the GP elsewhere.
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
        Set core matrices
        """
        self.X = X
        self.betas = flatarray[0:nt]
        self.D = flatarray[nt+B*(B+1)//2:].reshape((-1, 1))
        self.L = np.zeros((B, B))
        self.L[np.tril_indices(B)] = flatarray[nt:nt+B*(B+1)//2]
        self.beta = scipy.linalg.cho_solve((self.L, True), self.D)

    def margLike(self):
        """
        Returns marginalized likelihood of GP
        """
        return 0.5 * np.sum(self.beta * self.D) +\
            0.5 * self.logdet + 0.5 * self.D.size * log_2_pi

    def predict(self, x_pred):
        """
        Raw way to predict outputs with the GP
        """
        assert x_pred.shape[1] == 3
        KXXp = self.kernel.K(x_pred, self.X)
        KXpXp = self.kernel.K(x_pred)
        v = scipy.linalg.cho_solve((self.L, True), KXXp.T)
        if isinstance(self.mean_fct, Photoz_mean_function):
            mf = self.mean_fct.f(x_pred)
        elif isinstance(self.mean_fct, Photoz_linear_sed_basis):
            hx = self.mean_fct.f(self.X).T
            hx_pred = self.mean_fct.f(x_pred).T
            if False:
                mf = np.dot(hx_pred.T, self.betas)[:, None]
            else:
                R = hx_pred - np.dot(hx, v)
                hkh = np.dot(hx, scipy.linalg.cho_solve((self.L, True), hx.T))
                nt = self.mean_fct.nt
                b = np.repeat(1.0, nt)[:, None]
                bvar = (b/2.)**2
                vh = np.dot(hx, self.beta) + b / bvar
                hkherr = hkh + np.diag(1./bvar.flatten())
                betabar, _, _, _ = np.linalg.lstsq(hkherr, vh)
                self.betas = betabar.ravel()
                bis = scipy.linalg.solve(hkherr, R)
                mf = np.dot(R.T, betabar)
                KXpXp += np.dot(R.T, bis)
            #print("betabar", betabar.ravel() / np.max(betabar))
            #store: betabar
        else:
            mf = 0
        y_pred = np.dot(KXXp, self.beta) + mf
        y_pred_fullcov = KXpXp - KXXp.dot(v)
        return y_pred, y_pred_fullcov

    def predictAndInterpolate(self, redshiftGrid, ell=1.0, z=None):
        """
        Convenient way to get flux predictions on a redshift/band grid.
        First compute on the coarce GP grid and then interpolate on finer grid.
        ell should be set to reference luminosity used in the GP.
        z is an additional redshift to compute predictions at.
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
        y_pred, y_pred_fullcov = self.predict(X_pred)
        model_mean = np.zeros((redshiftGrid.size, numBands))
        model_var = np.zeros((redshiftGrid.size, numBands))
        for i in range(numBands):
            y_pred_bin = y_pred[i*numZGP:(i+1)*numZGP].ravel()
            y_var_bin = np.diag(y_pred_fullcov)[i*numZGP:(i+1)*numZGP]
            model_mean[:, i] = np.interp(redshiftGrid,
                                         redshiftGridGP_loc, y_pred_bin)
            model_var[:, i] = np.interp(redshiftGrid,
                                        redshiftGridGP_loc, y_var_bin)
        #model_covar = np.zeros((redshiftGrid.size, numBands, numBands))
        #for i in range(numBands):
        #    for j in range(numBands):
        #        y_covar_bin = y_pred_fullcov[i*numZGP:(i+1)*numZGP, :][:, j*numZGP:(j+1)*numZGP]
        #        interp_spline = RectBivariateSpline(redshiftGridGP_loc, redshiftGridGP_loc, y_covar_bin)
        #        model_covar[:, i, j] = interp_spline(redshiftGrid, redshiftGrid, grid=False)
        return model_mean, model_var

    def estimateAlphaEll(self):
        """
        Estimate alpha by fitting colours with power law
        then estimate ell by fixing alpha by fitting fluxes with power law.
        """
        X_pred = 1*self.X

        def fun(alpha):
            self.mean_fct.alpha = alpha[0]
            y_pred = self.mean_fct.f(X_pred).ravel()
            y_pred *= np.mean(self.Y) / y_pred.mean()
            chi2 = scalefree_flux_likelihood(self.Y.ravel(), self.Yvar.ravel(),
                                                y_pred[None, None, :], returnChi2=True)
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

    def drawSED(self, z, ell, wavs):
        """
        Draw SED from GP model (prior only, no likelihood)
        """
        opz = 1 + z
        meanfct = np.exp(self.mean_fct.alpha *
                         (wavs/opz - self.mean_fct.lambdaRef))
        dWav = wavs[:, None] - wavs[None, :]
        cov_C = self.kernel.var_C *\
            np.exp(-0.5*(dWav/opz/self.kernel.alpha_C)**2)
        cov_L = 0*cov_C
        for mu, sig in zip(self.kernel.lines_mu, self.kernel.lines_sig):
            cov_L += self.kernel.var_L *\
                np.exp(-0.5*(dWav/opz/self.kernel.alpha_L)**2) *\
                np.exp(-0.5*((wavs[:, None]/opz-mu)/sig)**2) *\
                np.exp(-0.5*((wavs[None, :]/opz-mu)/sig)**2)
        fac = opz * ell / 4 / np.pi / self.kernel.DL_z(z)**2
        sed = fac * np.random.multivariate_normal(meanfct, cov_C + cov_L)
        numB = self.mean_fct.fcoefs_amp.shape[0]
        filters = np.zeros((numB, wavs.size))
        for i in range(numB):
            for amp, mu, sig in zip(self.mean_fct.fcoefs_amp[i, :],
                                    self.mean_fct.fcoefs_mu[i, :],
                                    self.mean_fct.fcoefs_sig[i, :]):
                filters[i, :] += amp * np.exp(-0.5*((wavs-mu)/sig)**2)
        return sed, fac, cov_C + cov_L, filters

    def optimizeAlpha_GP(self):
        """
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
        For optimizing alpha with the marglike as objective using scipy.
        """
        self.mean_fct.alpha = alpha[0]
        self.D = self.Y - self.mean_fct.f(self.X)
        self.beta = scipy.linalg.cho_solve((self.L, True), self.D)
        return self.margLike()
