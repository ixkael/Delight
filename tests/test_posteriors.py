# -*- coding: utf-8 -*-

import numpy as np
from delight.priors import *
from delight.utils import *
from delight.posteriors import *


def test_multiobj_flux_likelihood_margell():

    nt = 1
    mu_ell = np.random.uniform(0, 1, nt)
    mu_lnz = np.random.uniform(-1, 1, nt)
    var_ell = np.random.uniform(0, 1, nt)
    var_lnz = np.random.uniform(0, 1, nt)
    rho = np.random.uniform(0, 1, nt)
    rho *= np.sqrt(var_lnz*var_ell)

    nz = 5
    nl = 500
    z_grid = np.linspace(1e-1, 1e1, nz)
    ell_grid = np.linspace(-5, 5, nl)

    nf = 2
    f_mod = np.random.randn(nt, nz, nf)
    nobj = 5
    f_obs = np.random.randn(nobj, nf)
    f_obs_var = np.abs(np.random.randn(nobj, nf))

    vals1 = object_evidences_marglnzell(
        f_obs,  # nobj * nf
        f_obs_var,  # nobj * nf
        f_mod,  # nt * nz * nf
        z_grid,
        mu_ell, mu_lnz, var_ell, var_lnz, rho)

    vals2 = object_evidences_numerical(
        f_obs,  # nobj * nf
        f_obs_var,  # nobj * nf
        f_mod,  # nt * nz * nf
        z_grid, ell_grid,
        mu_ell, mu_lnz, var_ell, var_lnz, rho)

    np.testing.assert_allclose(vals1, vals2, rtol=1e-4)
