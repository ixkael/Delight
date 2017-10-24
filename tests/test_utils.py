# -*- coding: utf-8 -*-

from scipy.interpolate import interp2d
from delight.utils import *
from astropy.cosmology import FlatLambdaCDM
from delight.utils import approx_flux_likelihood
from delight.posteriors import gaussian, gaussian2d
from delight.utils_cy import approx_flux_likelihood_cy
from delight.utils_cy import find_positions, bilininterp_precomputedbins
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
        ell_hat, ell_var = np.ones((nz, )), 0.01*np.ones((nz, ))

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
        np.allclose(res1, res2, rtol=relative_accuracy)


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

    ell, ell_var = 0, 0
    like_grid1 = approx_flux_likelihood(
        fluxes, fluxesVar,
        model_mean,
        f_mod_covar=0*model_var,
        ell_hat=ell,
        ell_var=ell_var,
        normalized=False, marginalizeEll=True, renormalize=False
    )
    like_grid2, ells = scalefree_flux_likelihood(
        fluxes, fluxesVar,
        model_mean
    )
    relative_accuracy = 1e-2
    np.allclose(like_grid1, like_grid2, rtol=relative_accuracy)


def test_interp():

    numBands, nobj = 3, 10
    nz1, nz2 = 40, 50
    grid1, grid2 = np.logspace(0., 1., nz1), np.linspace(1., 10., nz2)

    v1s, v2s = np.random.uniform(1, 10, nobj), np.random.uniform(1, 10, nobj)
    p1s = np.zeros((nobj, ), dtype=int)
    find_positions(nobj, nz1, v1s, p1s, grid1)
    p2s = np.zeros((nobj, ), dtype=int)
    find_positions(nobj, nz2, v2s, p2s, grid2)

    Kgrid = np.zeros((numBands, nz1, nz2))
    for b in range(numBands):
        Kgrid[b, :, :] = (grid1[:, None] * grid2[None, :])**(b+1.)

    Kinterp = np.zeros((numBands, nobj))
    bilininterp_precomputedbins(numBands, nobj, Kinterp, v1s, v2s, p1s, p2s,
                                grid1, grid2, Kgrid)
    Kinterp2 = np.zeros((numBands, nobj))

    for b in range(numBands):
        interp = interp2d(grid2, grid1, Kgrid[b, :, :])
        for o in range(nobj):
            Kinterp2[b, o] = interp(v1s[o], v2s[o])

    np.allclose(Kinterp, Kinterp2, rtol=relative_accuracy)


def test_correlatedgaussianfactorization():

    mu_ell, mu_lnz, var_ell, var_lnz, rho = np.random.uniform(0, 1, 5)
    rho *= np.sqrt(var_ell*var_lnz)

    for i in range(10):
        lnz, ell = np.random.uniform(-1, 2, 2)
        mu_ell_prime = mu_ell + rho * (lnz - mu_lnz) / var_lnz
        var_ell_prime = var_ell - rho**2 / var_lnz
        val1 = gaussian(mu_ell_prime, ell, var_ell_prime**0.5)
        val1 *= gaussian(mu_lnz, lnz, var_lnz**0.5)
        val2 = gaussian2d(ell, lnz, mu_ell, mu_lnz, var_ell, var_lnz, rho)
        assert np.abs(val1/val2) - 1 < 1e-12

        rho = 0
        val2 = gaussian2d(ell, lnz, mu_ell, mu_lnz, var_ell, var_lnz, rho)
        val3 = gaussian(ell, mu_ell, var_ell**0.5) *\
            gaussian(lnz, mu_lnz, var_lnz**0.5)
        assert np.abs(val2/val3) - 1 < 1e-12
