# -*- coding: utf-8 -*-

from delight.utils import *
from astropy.cosmology import FlatLambdaCDM
from delight.utils import approx_flux_likelihood
from delight.utils_cy import approx_flux_likelihood_cy
from time import time

relative_accuracy = 0.05


def test_approx_DL():
    for z in np.linspace(0.01, 4, num=10):
        z = 2.
        v1 = approx_DL()(z)
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=None)
        v2 = cosmo.luminosity_distance(z).value
        assert abs(v1/v2 - 1) < 0.01


def test_random_X():
    size = 10
    X = random_X_bzl(size, numBands=5, redshiftMax=3.0)
    assert X.shape == (size, 3)


def test_full_fluxlikelihood():

    nz, nt, nf = 100, 100, 5

    for i in range(3):
        f_obs = np.random.uniform(low=1, high=2, size=nf)
        f_obs_var = np.random.uniform(low=.1, high=.2, size=nf)
        f_mod = np.random.uniform(low=1, high=2, size=nz*nt*nf)\
            .reshape((nz, nt, nf))
        f_mod_covar = np.random.uniform(low=.1, high=.2, size=nz*nt*nf)\
            .reshape((nz, nt, nf))
        ell_hat, ell_var = 1, 0.01

        t1 = time()
        res1 = approx_flux_likelihood(
            f_obs, f_obs_var, f_mod, f_mod_covar=f_mod_covar,
            ell_hat=ell_hat, ell_var=ell_var)
        t2 = time()
        res2 = np.zeros_like(res1)
        approx_flux_likelihood_cy(
            res2, nz, nt, nf,
            f_obs, f_obs_var, f_mod, f_mod_covar,
            ell_hat, ell_var)
        t3 = time()
        print(t2-t1, t3-t2)
        assert np.allclose(res1, res2, rtol=relative_accuracy)


def test_flux_likelihood_approxscalemarg():

    nz, nt, nf = 3, 2, 5
    fluxes = np.random.uniform(low=1, high=2, size=nf)
    fluxesVar = np.random.uniform(low=.1, high=.2, size=nf)
    model_mean = np.random.uniform(low=1, high=2, size=nz*nt*nf)\
        .reshape((nz, nt, nf))
    model_var = np.random.uniform(low=.1, high=.2, size=nz*nt*nf)\
        .reshape((nz, nt, nf))
    model_covar = np.zeros((nz, nt, nf, nf))
    for i in range(nz):
        for j in range(nt):
            model_covar[i, j, :, :] = np.diag(model_var[i, j, :])

    ell, ell_var = 1, 1e6
    like_grid1 = approx_flux_likelihood(
        fluxes, fluxesVar,
        model_mean,
        f_mod_covar=0*model_var,
        ell_hat=ell,
        ell_var=ell_var,
        normalized=False
    )
    like_grid2 = scalefree_flux_likelihood(
        fluxes, fluxesVar,
        model_mean
    )
    relative_accuracy = 1e-2
    np.allclose(like_grid1, like_grid2, rtol=relative_accuracy)
