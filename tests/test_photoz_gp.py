
import pytest
import numpy as np

from delight.priors import Rayleigh, Schechter, Kumaraswamy
from delight.photoz_gp import PhotozGP
from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel
from delight.utils import random_X_bztl,\
    random_filtercoefs, random_linecoefs, random_hyperparams

NREPEAT = 4
size = 10
numBands = 5
numLines = 0
numCoefs = 1


@pytest.fixture(params=[False])
def create_p_z_t(request):
    if request.param is False:
        return None
    else:
        alpha0, alpha1 = np.random.uniform(0., 2., size=2)
        return Rayleigh(alpha0, alpha1)


@pytest.fixture(params=[False])
def create_p_ell_t(request):
    if request.param is False:
        return None
    else:
        ellStar, alpha0, alpha1 = np.random.uniform(0., 2., size=2)
        return Schechter(ellStar, alpha0, alpha1)


@pytest.fixture(params=[False])
def create_p_t(request):
    if request.param is False:
        return None
    else:
        alpha0, alpha1 = np.random.uniform(0., 2., size=2)
        return Kumaraswamy(alpha0, alpha1)


@pytest.fixture()
def create_gp(create_p_ell_t, create_p_z_t, create_p_t):

    X = random_X_bztl(size, numBands=numBands)
    bands, redshifts, types, luminosities = np.split(X, 4, axis=1)

    fcoefs_amp, fcoefs_mu, fcoefs_sig \
        = random_filtercoefs(numBands, numCoefs)
    lines_mu, lines_sig = random_linecoefs(numLines)
    var_T, alpha_C, alpha_L, alpha_T = random_hyperparams()

    kern = Photoz_kernel(fcoefs_amp, fcoefs_mu, fcoefs_sig,
                         lines_mu, lines_sig, var_T,
                         alpha_C, alpha_L, alpha_T)

    mean_function = Photoz_mean_function()

    noisy_fluxes = np.random.uniform(low=0., high=1., size=size)
    flux_variances = np.random.uniform(low=0., high=1., size=size)

    prior_ell_t = create_p_ell_t
    assert(prior_ell_t is None or isinstance(prior_ell_t, Schechter))
    prior_z_t = create_p_z_t
    assert(prior_z_t is None or isinstance(prior_z_t, Rayleigh))
    prior_t = create_p_t
    assert(prior_t is None or isinstance(prior_t, Kumaraswamy))

    gp = PhotozGP(
        bands, redshifts, luminosities, types,
        noisy_fluxes, flux_variances,
        kern, mean_function,
        prior_z_t=prior_z_t,
        prior_ell_t=prior_ell_t,
        prior_t=prior_t,
        X_inducing=None,
        fix_inducing_to_mean_prediction=True
        )

    return gp


def test_no_inducing(create_gp):
    gp = create_gp
    assert(isinstance(gp, PhotozGP))


def test_gradients(create_gp):
    pass
