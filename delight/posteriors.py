# -*- coding: utf-8 -*-

import numpy as np

#from scipy.misc import logsumexp
from scipy.special import logsumexp

def hypercube2simplex(zs):
    fac = np.concatenate((1 - zs, np.array([1])))
    zsb = np.concatenate((np.array([1]), zs))
    fs = np.cumprod(zsb) * fac
    return fs


def hypercube2simplex_jacobian(fs, zs):
    jaco = np.zeros((zs.size, fs.size))
    for j in range(fs.size):
        for i in range(zs.size):
            if i < j:
                jaco[i, j] = fs[j] / zs[i]
            if i == j:
                jaco[i, j] = fs[j] / (zs[j] - 1)
    return jaco


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


def trapz(x, y, axis=0):
    return 0.5 * np.sum((y[1:]+y[:-1])*(x[1:]-x[:-1]), axis=axis)


def object_evidences_marglnzell(
    f_obs,  # nobj * nf
    f_obs_var,  # nobj * nf
    f_mod,  # nt * nz * nf
    z_grid,
    mu_ell, mu_lnz, var_ell, var_lnz, rho  # nt
        ):
    numTypes, nz = f_mod.shape[0], f_mod.shape[1]
    lnz_grid_t = np.log(z_grid[None, :]) * np.ones((numTypes, 1))  # nt * nz
    mu_ell_prime = mu_ell[:, None] +\
        rho[:, None] * (lnz_grid_t - mu_lnz[:, None]) / var_lnz[:, None]
    # nt * nz
    var_ell_prime = (var_ell[:, None] - rho[:, None]**2 / var_lnz[:, None])\
        * np.ones((1, nz))  # nt * nz

    marglike = multiobj_flux_likelihood_margell(
            f_obs, f_obs_var,  # nobj * nf
            f_mod,  # nt * nz * nf
            mu_ell_prime, var_ell_prime,  # nobj * nt * nz
            marginalizeEll=True, normalized=True)  # nobj * nt * nz
    prior_lnz = gaussian(lnz_grid_t, mu_lnz[:, None], var_lnz[:, None]**0.5)
    # nt * nz

    # evidences_it = \
    # np.trapz(prior_lnz[None, :, :] * marglike, x=z_grid, axis=2)
    x = z_grid[None, None, :]
    y = prior_lnz[None, :, :] * marglike
    evidences_it = 0.5 * np.sum((y[:, :, 1:]+y[:, :, :-1]) *
                                (x[:, :, 1:]-x[:, :, :-1]), axis=2)
    # nobj * nt

    return evidences_it


def object_evidences_numerical(
    f_obs,  # nobj * nf
    f_obs_var,  # nobj * nf
    f_mod,  # nt * nz * nf
    z_grid, ell_grid,
    mu_ell, mu_lnz, var_ell, var_lnz, rho  # nt
        ):
    nobj = f_obs.shape[0]
    nt, nz = f_mod.shape[0], f_mod.shape[1]
    assert z_grid.size == nz
    nl = ell_grid.size
    lnz_grid_t = np.log(z_grid[None, :]) * np.ones((nt, 1))  # nt * nz

    prior_lnzell = np.zeros((nt, nz, nl))
    like_lnzell = np.zeros((nobj, nt, nz, nl))
    for it in range(nt):
        for iz, z in enumerate(z_grid):
            for il, el in enumerate(ell_grid):
                prior_lnzell[it, iz, il] =\
                    gaussian2d(np.log(z), el,
                               mu_lnz[it], mu_ell[it],
                               var_lnz[it], var_ell[it], rho[it])
                v = gaussian(f_obs[:, :], el*f_mod[it, iz, :],
                             f_obs_var[:, :]**0.5)
                like_lnzell[:, it, iz, il] = np.prod(v, axis=1)

    evidences_it = np.trapz(
        np.trapz(prior_lnzell[None, :, :, :] * like_lnzell[:, :, :, :],
                 x=ell_grid, axis=3), x=z_grid, axis=2)    # nobj * nt

    return evidences_it
