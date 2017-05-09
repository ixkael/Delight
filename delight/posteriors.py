# -*- coding: utf-8 -*-

import numpy as np

from delight.utils import multiobj_flux_likelihood_margell,\
    gaussian, gaussian2d

from scipy.misc import logsumexp
from scipy.misc import derivative


def lnposterior_onetype_lzd(t, elli_s, z_is, d_is,
                            fluxObs_ibs, fluxObsEr_ibs,
                            fluxModInterps_tbs, grad=False):
    na = 0  # alphas.size
    nobj, nb = fluxObs_ibs.shape
    nt, nb2 = fluxModInterps_tbs.shape
    assert nb == nb2
    lnpriors_its = 0.0  # lnprior_ts(z_is, d_is, ell_is, alphas)  # nobj * nt
    fluxMod_tbs = np.zeros((nobj, nb))  # nobj * nt * nb
    for b in range(nb):
        fluxMod_tbs[:, b] = elli_s * fluxModInterps_tbs[t, b](z_is,
                                                              d_is, grid=False)
    lnlikes_itbs = lngaussian(fluxObs_ibs[:, :],
                              fluxMod_tbs[:, :],
                              fluxObsEr_ibs[:, :])  # nobj * nb
    lnlikes_its = np.sum(lnlikes_itbs, axis=1)  # nobj
    lnprob_is = lnpriors_its + lnlikes_its  # nobj
    if grad is False:
        return np.sum(lnprob_is, axis=0)  # nobj
    lnlikes_itbs_gradfluxes = lngaussian_gradmu(fluxObs_ibs[:, :],
                                                fluxMod_tbs[:, :],
                                                fluxObsEr_ibs[:, :])
    fluxMod_tbs_grad_zis = np.zeros((nobj, nb))  # nobj * nt * nb
    fluxMod_tbs_grad_ellis = np.zeros((nobj, nb))  # nobj * nt * nb
    fluxMod_tbs_grad_dis = np.zeros((nobj, nb))  # nobj * nt * nb
    for b in range(nb):
        fluxMod_tbs_grad_ellis[:, b] =\
            fluxModInterps_tbs[t, b](z_is, d_is, grid=False)
        fluxMod_tbs_grad_zis[:, b] = elli_s *\
            fluxModInterps_tbs[t, b](z_is, d_is, dx=1, dy=0, grid=False)
        fluxMod_tbs_grad_dis[:, b] = elli_s *\
            fluxModInterps_tbs[t, b](z_is, d_is, dx=0, dy=1, grid=False)
    lnpost_grad_its = np.zeros((na + nobj*3, ))
    lnprobfac = np.exp(lnlikes_its + lnpriors_its - lnprob_is)
    lnprobfac_grad = np.sum(lnlikes_itbs_gradfluxes *
                            fluxMod_tbs_grad_ellis, axis=1)
    lnpost_grad_its[na:na+nobj] = lnprobfac * lnprobfac_grad
    lnprobfac_grad = np.sum(lnlikes_itbs_gradfluxes *
                            fluxMod_tbs_grad_zis, axis=1)
    lnpost_grad_its[na+nobj:na+2*nobj] = lnprobfac * lnprobfac_grad  # nobj
    lnprobfac_grad = np.sum(lnlikes_itbs_gradfluxes *
                            fluxMod_tbs_grad_dis, axis=1)
    lnpost_grad_its[na+2*nobj:na+3*nobj] = lnprobfac * lnprobfac_grad  # nobj
    return np.sum(lnprob_is, axis=0), lnpost_grad_its


def lnposterior_lzd(elli_s, z_is, d_is,
                    fluxObs_ibs, fluxObsEr_ibs,
                    fluxModInterps_tbs, grad=False):
    na = 0  # alphas.size
    nobj, nb = fluxObs_ibs.shape
    nt, nb2 = fluxModInterps_tbs.shape
    if len(z_is.shape) > 1 and len(elli_s.shape) > 1 and len(d_is.shape) > 1:
        multitypes = True
        assert z_is.shape[1] == nt
        assert elli_s.shape[1] == nt
        assert d_is.shape[1] == nt
    else:
        multitypes = False
    assert nb == nb2
    lnpriors_its = 0.0  # lnprior_ts(z_is, d_is, ell_is, alphas)  # nobj * nt
    fluxMod_tbs = np.zeros((nobj, nt, nb))  # nobj * nt * nb
    for t in range(nt):
        for b in range(nb):
            if multitypes:
                fluxMod_tbs[:, t, b] = elli_s[:, t] *\
                    fluxModInterps_tbs[t, b](z_is[:, t], d_is[:, t],
                                             grid=False)
            else:
                fluxMod_tbs[:, t, b] = elli_s *\
                    fluxModInterps_tbs[t, b](z_is, d_is, grid=False)
    lnlikes_itbs = lngaussian(fluxObs_ibs[:, None, :],
                              fluxMod_tbs[:, :, :],
                              fluxObsEr_ibs[:, None, :])  # nobj * nt * nb
    lnlikes_its = np.sum(lnlikes_itbs, axis=2)  # nobj * nt
    lnprob_is = logsumexp(lnpriors_its + lnlikes_its, axis=1)  # nobj
    if grad is False:
        return np.sum(lnprob_is, axis=0)  # scalar
    lnlikes_itbs_gradfluxes = lngaussian_gradmu(fluxObs_ibs[:, None, :],
                                                fluxMod_tbs[:, :, :],
                                                fluxObsEr_ibs[:, None, :])
    fluxMod_tbs_grad_zis = np.zeros((nobj, nt, nb))  # nobj * nt * nb
    fluxMod_tbs_grad_ellis = np.zeros((nobj, nt, nb))  # nobj * nt * nb
    fluxMod_tbs_grad_dis = np.zeros((nobj, nt, nb))  # nobj * nt * nb
    for t in range(nt):
        for b in range(nb):
            if multitypes:
                fluxMod_tbs_grad_ellis[:, t, b] =\
                    fluxModInterps_tbs[t, b](z_is[:, t], d_is[:, t],
                                             grid=False)
                fluxMod_tbs_grad_zis[:, t, b] = elli_s[:, t] *\
                    fluxModInterps_tbs[t, b](z_is[:, t], d_is[:, t],
                                             dx=1, grid=False)
                fluxMod_tbs_grad_dis[:, t, b] = elli_s[:, t] *\
                    fluxModInterps_tbs[t, b](z_is[:, t], d_is[:, t],
                                             dy=1, grid=False)
            else:
                fluxMod_tbs_grad_ellis[:, t, b] =\
                    fluxModInterps_tbs[t, b](z_is, d_is, grid=False)
                fluxMod_tbs_grad_zis[:, t, b] = elli_s *\
                    fluxModInterps_tbs[t, b](z_is, d_is, dx=1, dy=0,
                                             grid=False)
                fluxMod_tbs_grad_dis[:, t, b] = elli_s *\
                    fluxModInterps_tbs[t, b](z_is, d_is, dx=0, dy=1,
                                             grid=False)
    lnpost_grad_its = np.zeros((na + nobj*3, nt))
    lnprobfac = np.exp(lnlikes_its + lnpriors_its - lnprob_is[:, None])
    lnprobfac_grad = np.sum(lnlikes_itbs_gradfluxes *
                            fluxMod_tbs_grad_ellis, axis=2)
    lnpost_grad_its[na:na+nobj, :] = lnprobfac * lnprobfac_grad
    lnprobfac_grad = np.sum(lnlikes_itbs_gradfluxes *
                            fluxMod_tbs_grad_zis, axis=2)
    lnpost_grad_its[na+nobj:na+2*nobj, :] = lnprobfac * lnprobfac_grad
    # nobj * nt
    lnprobfac_grad = np.sum(lnlikes_itbs_gradfluxes *
                            fluxMod_tbs_grad_dis, axis=2)
    lnpost_grad_its[na+2*nobj:na+3*nobj, :] = lnprobfac * lnprobfac_grad
    # nobj * nt
    if multitypes:
        return np.sum(lnprob_is, axis=0), lnpost_grad_its
    else:
        return np.sum(lnprob_is, axis=0), np.sum(lnpost_grad_its, axis=1)


def lnposterior_ell_redshift_dust(elli_s, z_is,
                                  fluxObs_ibs, fluxObsEr_ibs,
                                  fluxModInterps_tbs, grad=False):
    na = 0  # alphas.size
    nobj, nb = fluxObs_ibs.shape
    nt, nb2 = fluxModInterps_tbs.shape
    lnpriors_its = 0.0  # lnprior_ts(z_is, d_is, ell_is, alphas)  # nobj * nt
    fluxMod_tbs = np.zeros((nobj, nt, nb))  # nobj * nt * nb
    for t in range(nt):
        for b in range(nb):
            if multitypes:
                fluxMod_tbs[:, t, b] = elli_s[:, t] *\
                    fluxModInterps_tbs[t, b](z_is[:, t], d_is[:, t],
                                             grid=False)
            else:
                fluxMod_tbs[:, t, b] = elli_s *\
                    fluxModInterps_tbs[t, b](z_is, d_is, grid=False)
    lnlikes_itbs = lngaussian(fluxObs_ibs[:, None, :],
                              fluxMod_tbs[:, :, :],
                              fluxObsEr_ibs[:, None, :])  # nobj * nt * nb
    lnlikes_its = np.sum(lnlikes_itbs, axis=2)  # nobj * nt
    lnprob_is = logsumexp(lnpriors_its + lnlikes_its, axis=1)  # nobj
    if grad is False:
        return np.sum(lnprob_is, axis=0)  # scalar
    lnlikes_itbs_gradfluxes = lngaussian_gradmu(fluxObs_ibs[:, None, :],
                                                fluxMod_tbs[:, :, :],
                                                fluxObsEr_ibs[:, None, :])
    fluxMod_tbs_grad_zis = np.zeros((nobj, nt, nb))  # nobj * nt * nb
    fluxMod_tbs_grad_ellis = np.zeros((nobj, nt, nb))  # nobj * nt * nb
    for t in range(nt):
        for b in range(nb):
            fluxMod_tbs_grad_ellis[:, t, b] =\
                fluxModInterps_tbs[t, b](z_is)
            fluxMod_tbs_grad_zis[:, t, b] = elli_s *\
                fluxModInterps_tbs[t, b](z_is, nu=1)
    lnpost_grad_its = np.zeros((na + nobj*3, nt))
    lnprobfac = np.exp(lnlikes_its + lnpriors_its - lnprob_is[:, None])
    lnprobfac_grad = np.sum(lnlikes_itbs_gradfluxes *
                            fluxMod_tbs_grad_ellis, axis=2)
    lnpost_grad_its[na:na+nobj, :] = lnprobfac * lnprobfac_grad
    lnprobfac_grad = np.sum(lnlikes_itbs_gradfluxes *
                            fluxMod_tbs_grad_zis, axis=2)
    lnpost_grad_its[na+nobj:na+2*nobj, :] = lnprobfac * lnprobfac_grad
    # nobj * nt
    return np.sum(lnprob_is, axis=0), np.sum(lnpost_grad_its, axis=1)


def object_evidences_marglnzell(
    f_obs,  # nobj * nf
    f_obs_var,  # nobj * nf
    f_mod,  # nt * nz * nf
    z_grid,
    mu_ell, mu_lnz, var_ell, var_lnz, rho  # nt x 1
        ):
    numTypes, nz = f_mod.shape[0], f_mod.shape[1]
    lnz_grid_t = np.log(z_grid[None, :]) * np.ones((numTypes, 1))  # nt * nz
    mu_ell_prime = mu_ell + rho * (lnz_grid_t - mu_lnz) / var_lnz  # nt * nz
    var_ell_prime = (var_ell - rho**2 / var_lnz) * np.ones((1, nz))  # nt * nz

    marglike = multiobj_flux_likelihood_margell(
            f_obs, f_obs_var,  # nobj * nf
            f_mod,  # nt * nz * nf
            mu_ell_prime, var_ell_prime,  # nobj * nt * nz
            marginalizeEll=True, normalized=True)  # nobj * nt * nz
    prior_lnz = gaussian(lnz_grid_t, mu_lnz, var_lnz**0.5)  # nt * nz
    print(prior_lnz.shape)
    print(marglike.shape)

    evidences_it = np.trapz(prior_lnz[None, :, :] * marglike, x=z_grid, axis=2)
    # nobj * nt

    return evidences_it


def object_evidences_numerical(
    f_obs,  # nobj * nf
    f_obs_var,  # nobj * nf
    f_mod,  # nt * nz * nf
    z_grid, ell_grid,
    mu_ell, mu_lnz, var_ell, var_lnz, rho  # nt x 1
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
                               mu_lnz[it, 0], mu_ell[it, 0],
                               var_lnz[it, 0], var_ell[it, 0], rho[it, 0])
                v = gaussian(f_obs[:, :], el*f_mod[it, iz, :],
                             f_obs_var[:, :]**0.5)
                like_lnzell[:, it, iz, il] = np.prod(v, axis=1)

    evidences_it = np.trapz(
        np.trapz(prior_lnzell[None, :, :, :] * like_lnzell[:, :, :, :],
                 x=ell_grid, axis=3), x=z_grid, axis=2)    # nobj * nt

    return evidences_it
