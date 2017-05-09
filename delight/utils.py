# -*- coding: utf-8 -*-

import numpy as np
from scipy.misc import derivative


class approx_DL():
    """
    Approximate luminosity_distance relation,
    agrees with astropy.FlatLambdaCDM(H0=70, Om0=0.3, Ob0=None) better than 1%
    """
    def __call__(self, z):
        return np.exp(30.5 * z**0.04 - 21.7)

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def symmetrize(a):
    """
    Symmmetrize matrix
    """
    return a + a.T - np.diag(a.diagonal())


def random_X_bzl(size, numBands=5, redshiftMax=3.0):
    """Create random (but reasonable) input space for photo-z GP """
    X = np.zeros((size, 3))
    X[:, 0] = np.random.randint(low=0, high=numBands-1, size=size)
    X[:, 1] = np.random.uniform(low=0.1, high=redshiftMax, size=size)
    X[:, 2] = np.random.uniform(low=0.5, high=10.0, size=size)
    return X


def random_filtercoefs(numBands, numCoefs):
    """Create random (but reasonable) coefficients describing
    numBands photometric filters as sum of gaussians"""
    fcoefs_amp\
        = np.random.uniform(low=0., high=1., size=numBands*numCoefs)\
        .reshape((numBands, numCoefs))
    fcoefs_mu\
        = np.random.uniform(low=3e3, high=1e4, size=numBands*numCoefs)\
        .reshape((numBands, numCoefs))
    fcoefs_sig\
        = np.random.uniform(low=30, high=500, size=numBands*numCoefs)\
        .reshape((numBands, numCoefs))
    return fcoefs_amp, fcoefs_mu, fcoefs_sig


def random_linecoefs(numLines):
    """Create random (but reasonable) coefficients describing lines in SEDs"""
    lines_mu = np.random.uniform(low=1e3, high=1e4, size=numLines)
    lines_sig = np.random.uniform(low=5, high=50, size=numLines)
    return lines_mu, lines_sig


def random_hyperparams():
    """Create random (but reasonable) hyperparameters for photo-z GP"""
    alpha_T, var_C, var_L = np.random.uniform(low=0.5, high=2.0, size=3)
    alpha_C, alpha_L = np.random.uniform(low=10.0, high=1000.0, size=2)
    return var_C, var_L, alpha_C, alpha_L, alpha_T


def flux_likelihood(f_obs, f_obs_var, f_mod, f_mod_var=None):
    nz, nt, nf = f_mod.shape
    df = f_mod - f_obs[None, :]  # nz, nf
    if f_mod_var is None:
        sigma = f_obs_var[None, None, :]
    else:
        sigma = f_mod_var + f_obs_var[None, None, :]
    den = np.sqrt((2*np.pi)**nf * np.prod(sigma, axis=2))
    return np.exp(-0.5*np.sum(df**2/sigma, axis=2)) / den


def scalefree_flux_likelihood_multiobj(
        f_obs,  # no, ..., nf
        f_obs_var,  # no, ..., nf
        f_mod,  # ..., nf
        normalized=True):

    assert len(f_obs.shape) == len(f_mod.shape)
    assert len(f_obs_var.shape) == len(f_mod.shape)
    assert len(f_mod.shape) >= 2
    nf = f_mod.shape[-1]
    assert f_obs.shape[-1] == nf
    assert f_obs_var.shape[-1] == nf
    # nz * nt * nf
    invvar = np.where(f_obs/f_obs_var < 1e-6, 0.0, f_obs_var**-1.0)
    FOT = np.sum(f_mod * f_obs * invvar, axis=-1)  # no * nt
    FTT = np.sum(f_mod**2 * invvar, axis=-1)  # no * nt
    FOO = np.sum(f_obs**2 * invvar, axis=-1)  # no * nt
    sigma_det = np.prod(f_obs_var, axis=-1)
    chi2 = FOO - FOT**2.0 / FTT  # no * nt
    denom = np.sqrt(FTT)
    ellML = FOT / FTT
    if normalized:
        denom *= np.sqrt(sigma_det * (2*np.pi)**nf)
    like = np.exp(-0.5*chi2) / denom  # no * nt
    return like, ellML


def dirichlet(alphas, rsize=1):
    """
    Draw samples from a Dirichlet distribution.
    """
    gammabs = np.array([np.random.gamma(alpha, size=rsize)
                        for alpha in alphas])
    fbs = gammabs / gammabs.sum(axis=0)
    return fbs.T


def approx_flux_likelihood(
        f_obs,  # nf
        f_obs_var,  # nf
        f_mod,  # nz, nt, nf
        ell_hat=0,  # 1 or nz, nt
        ell_var=0,  # 1 or nz, nt
        f_mod_covar=None,  # nz, nt, nf (, nf)
        marginalizeEll=True,
        normalized=True,
        returnChi2=False,
        returnEllML=False):
    """
    Approximate flux likelihood, with scaling of both the mean and variance.
    This approximates the true likelihood with an iterative scheme.
    """

    assert len(f_obs.shape) == 1
    assert len(f_obs_var.shape) == 1
    assert len(f_mod.shape) == 3
    nz, nt, nf = f_mod.shape
    if f_mod_covar is not None:
        assert len(f_mod_covar.shape) == 3
    if f_mod_covar is None or len(f_mod_covar.shape) == 3:
        f_obs_r = f_obs[None, None, :]
        ellML = 0
        niter = 1 if f_mod_covar is None else 2
        for i in range(niter):
            if f_mod_covar is not None:
                var = f_obs_var[None, None, :] + ellML**2 * f_mod_covar
            else:
                var = f_obs_var[None, None, :]
            invvar = 1/var  # nz * nt * nf
            # np.where(f_obs_r/var < 1e-6, 0.0, var**-1.0)  # nz * nt * nf
            FOT = np.sum(f_mod * f_obs_r * invvar, axis=2)
            FTT = np.sum(f_mod**2 * invvar, axis=2)
            FOO = np.sum(f_obs_r**2 * invvar, axis=2)
            if np.all(ell_var > 0):
                FOT += ell_hat / ell_var  # nz * nt
                FTT += 1. / ell_var  # nz * nt
                FOO += ell_hat**2 / ell_var  # nz * nt
            sigma_det = np.prod(var, axis=2)
            ellbk = 1*ellML
            ellML = (FOT / FTT)[:, :, None]
    if returnEllML:
        return ellML
    chi2 = FOO - FOT**2.0 / FTT  # nz * nt
    if returnChi2:
        return chi2
    denom = 1
    if normalized:
        denom = denom * np.sqrt(sigma_det * (2*np.pi)**nf)
        if np.all(ell_var > 0):
            denom = denom * np.sqrt(2*np.pi * ell_var)
    if marginalizeEll:
        denom = denom * np.sqrt(FTT)
        if np.all(ell_var > 0):
            denom = denom / np.sqrt(2*np.pi)
    like = np.exp(-0.5*chi2) / denom  # nz * nt
    return like


def scalefree_flux_likelihood(f_obs, f_obs_var,
                              f_mod, returnChi2=False):
    nz, nt, nf = f_mod.shape
    var = f_obs_var  # nz * nt * nf
    invvar = np.where(f_obs/var < 1e-6, 0.0, var**-1.0)  # nz * nt * nf
    FOT = np.sum(f_mod * f_obs * invvar, axis=2)  # nz * nt
    FTT = np.sum(f_mod**2 * invvar, axis=2)  # nz * nt
    FOO = np.dot(invvar, f_obs**2)  # nz * nt
    ellML = FOT / FTT
    chi2 = FOO - FOT**2.0 / FTT  # nz * nt
    like = np.exp(-0.5*chi2) / np.sqrt(FTT)  # nz * nt
    if returnChi2:
        return chi2 + FTT, ellML
    else:
        return like, ellML


def CIlevel(redshiftGrid, PDF, fraction, numlevels=200):
    """
    Computes confidence interval from PDF.
    """
    evidence = np.trapz(PDF, redshiftGrid)
    for level in np.linspace(0, PDF.max(), num=numlevels):
        ind = np.where(PDF <= level)
        resint = np.trapz(PDF[ind], redshiftGrid[ind])
        if resint >= fraction*evidence:
            return level


def kldiv(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions"""
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def computeMetrics(ztrue, redshiftGrid, PDF, confIntervals):
    """
    Compute various metrics on the PDF
    """
    zmean = np.average(redshiftGrid, weights=PDF)
    zmap = redshiftGrid[np.argmax(PDF)]
    zstdzmean = np.sqrt(np.average((redshiftGrid-zmean)**2, weights=PDF))
    zstdzmap = np.sqrt(np.average((redshiftGrid-zmap)**2, weights=PDF))
    pdfAtZ = np.interp(ztrue, redshiftGrid, PDF)
    cumPdfAtZ = np.interp(ztrue, redshiftGrid, PDF.cumsum())
    confidencelevels = [
        CIlevel(redshiftGrid, PDF, 1.0 - confI) for confI in confIntervals
    ]
    return [ztrue, zmean, zstdzmean, zmap, zstdzmap, pdfAtZ, cumPdfAtZ]\
        + confidencelevels


def gaussian2d(x1, x2, mu1, mu2, cov1, cov2, corr):
    dx = np.array([x1 - mu1, x2 - mu2])
    cov = np.array([[cov1, corr], [corr, cov2]])
    v = np.exp(-0.5*np.dot(dx, np.linalg.solve(cov, dx)))
    v /= (2*np.pi) * np.sqrt(np.linalg.det(cov))
    return v


def gaussian(x, mu, sig):
    return np.exp(-0.5*((x-mu)/sig)**2.0) / np.sqrt(2*np.pi) / sig


def lngaussian(x, mu, sig):
    return - 0.5*((x - mu)/sig)**2 - 0.5*np.log(2*np.pi) - np.log(sig)


def lngaussian_gradmu(x, mu, sig):
    return (x - mu) / sig**2


def derivative_test(x0, fun, fun_grad, relative_accuracy,
                    n=1, lim=0, order=9, dxfac=0.01,
                    verbose=False, superverbose=False):
    grads = fun_grad(x0)
    for i in range(x0.size):
        if verbose:
            print(i, end=" ")

        def f(v):
            x = 1*x0
            x[i] = v
            return fun(x)
        grads2 = derivative(f, x0[i], dx=dxfac*x0[i], order=order, n=n)
        if superverbose:
            print(i, 'analytical:', grads[i], 'numerical:', grads2)
        if np.abs(grads2) >= lim:
            np.testing.assert_allclose(grads2, grads[i],
                                       rtol=relative_accuracy)


def multiobj_flux_likelihood_margell(
        f_obs,  # nobj * nf
        f_obs_var,  # nobj * nf
        f_mod,  # nt * nz * nf
        ell_hat,  # nt * nz
        ell_var,  # nt * nz
        marginalizeEll=True,
        normalized=True):
    """
    TODO
    """
    assert len(f_obs.shape) == 2
    assert len(f_obs_var.shape) == 2
    assert len(f_mod.shape) == 3
    assert len(ell_hat.shape) == 2
    assert len(ell_var.shape) == 2
    nt, nz, nf = f_mod.shape
    FOT = np.sum(
        f_mod[None, :, :, :] *
        f_obs[:, None, None, :] / f_obs_var[:, None, None, :],
        axis=3) +\
        ell_hat[None, :, :] / ell_var[None, :, :]
    FTT = np.sum(
        f_mod[None, :, :, :]**2 / f_obs_var[:, None, None, :],
        axis=3) + 1 / ell_var[None, :, :]
    FOO = np.sum(
        f_obs[:, None, None, :]**2 / f_obs_var[:, None, None, :],
        axis=3) +\
        ell_hat[None, :, :]**2.0 / ell_var[None, :, :]
    sigma_det = np.prod(f_obs_var[:, None, None, :], axis=3)
    chi2 = FOO - FOT**2.0 / FTT  # nobj * nt * nz
    denom = 1.
    if normalized:
        denom = denom *\
            np.sqrt(sigma_det * (2*np.pi)**nf) *\
            np.sqrt(2*np.pi * ell_var[None, :, :])
    if marginalizeEll:
        denom = denom * np.sqrt(FTT) / np.sqrt(2*np.pi)
    like = np.exp(-0.5*chi2) / denom  # nobj * nt * nz
    return like
