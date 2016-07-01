
import pytest
import numpy as np

import GPy

from delight.priors import Rayleigh, Schechter, Kumaraswamy
from delight.photoz_gp import PhotozGP
from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel
from delight.utils import random_X_bzlt,\
    random_filtercoefs, random_linecoefs, random_hyperparams

from scipy.misc import derivative
from copy import deepcopy

NREPEAT = 1
nObj = 5
nObjUnfixed = 4
numBands = 2
numLines = 4
numCoefs = 4
relative_accuracy = 0.1
size = numBands * nObj
bandsUsed = range(numBands)


@pytest.fixture(params=[False, True])
def create_p_z_t(request):
    """Create valid p(z|t) prior with reasonable parameters"""
    if request.param is False:
        return None
    else:
        alpha0, alpha1 = np.random.uniform(0.2, 2., size=2)
        return Rayleigh(alpha0, alpha1)


@pytest.fixture(params=[False, True])
def create_p_ell_t(request):
    """Create valid p(ell|t) prior with reasonable parameters"""
    if request.param is False:
        return None
    else:
        ellStar = np.random.uniform(0., 10., size=1)
        alpha0, alpha1 = np.random.uniform(-0.4, -0.3, size=2)
        return Schechter(ellStar, alpha0, alpha1)


@pytest.fixture(params=[False, True])
def create_p_t(request):
    """Create valid p(t) prior with reasonable parameters"""
    if request.param is False:
        return None
    else:
        alpha0, alpha1 = np.random.uniform(0.2, 2., size=2)
        return Kumaraswamy(alpha0, alpha1)


@pytest.fixture()
def create_gp(create_p_ell_t, create_p_z_t, create_p_t):
    """Create valid GP with reasonable parameters, kernel, mean fct"""

    X = random_X_bzlt(nObj)
    alpha = np.random.uniform(low=1e-3, high=2e-3, size=1)
    beta = np.random.uniform(low=0, high=1, size=1)
    bands, redshifts, luminosities, types = np.split(X, 4, axis=1)

    fcoefs_amp, fcoefs_mu, fcoefs_sig \
        = random_filtercoefs(numBands, numCoefs)
    lines_mu, lines_sig = random_linecoefs(numLines)
    var_C, var_L, alpha_C, alpha_L, alpha_T = random_hyperparams()

    noisy_fluxes = np.random.uniform(low=0., high=1., size=size)\
        .reshape((nObj, numBands))
    flux_variances = np.random.uniform(low=0., high=1., size=size)\
        .reshape((nObj, numBands))

    prior_ell_t = create_p_ell_t
    assert(prior_ell_t is None or isinstance(prior_ell_t, Schechter))
    prior_z_t = create_p_z_t
    assert(prior_z_t is None or isinstance(prior_z_t, Rayleigh))
    prior_t = create_p_t
    assert(prior_t is None or isinstance(prior_t, Kumaraswamy))

    unfixed_indices = np.random.choice(np.arange(nObj),
                                       nObjUnfixed,
                                       replace=False)

    gp = PhotozGP(
        redshifts, luminosities, types, unfixed_indices,
        noisy_fluxes, flux_variances, bandsUsed,
        fcoefs_amp, fcoefs_mu, fcoefs_sig,
        lines_mu, lines_sig,
        alpha, beta, var_C, var_L,
        alpha_C, alpha_L, alpha_T,
        prior_z_t=prior_z_t,
        prior_ell_t=prior_ell_t,
        prior_t=prior_t,
        X_inducing=None,
        fix_inducing_to_mean_prediction=True
        )

    return gp


def test_gradients(create_gp):
    """Test all gradients of the full likelihood function"""
    gp = create_gp
    assert(isinstance(gp, PhotozGP))

    v1 = gp.mean_function.alpha.gradient

    def f_alpha(v):
        gp2 = deepcopy(gp)
        gp2.mean_function.set_alpha(v)
        return gp2._log_marginal_likelihood
    v2 = derivative(f_alpha, gp.mean_function.alpha.values,
                    dx=0.01*gp.mean_function.alpha.values)
    assert abs(v1/v2-1) < relative_accuracy

    v1 = gp.mean_function.beta.gradient

    def f_beta(v):
        gp2 = deepcopy(gp)
        gp2.mean_function.set_beta(v)
        return gp2._log_marginal_likelihood
    v2 = derivative(f_beta, gp.mean_function.beta.values,
                    dx=0.01*gp.mean_function.beta.values)
    assert abs(v1/v2-1) < relative_accuracy

    v1 = gp.kern.alpha_L.gradient

    def f_alpha_L(v):
        gp2 = deepcopy(gp)
        gp2.kern.set_alpha_L(v)
        return gp2._log_marginal_likelihood
    v2 = derivative(f_alpha_L, gp.kern.alpha_L.values,
                    dx=0.01*gp.kern.alpha_L.values)
    if np.abs(v1) > 1e-10 and np.abs(v2) > 1e-10:
        assert abs(v1/v2-1) < relative_accuracy

    v1 = gp.kern.alpha_C.gradient

    def f_alpha_C(v):
        gp2 = deepcopy(gp)
        gp2.kern.set_alpha_C(v)
        return gp2._log_marginal_likelihood
    v2 = derivative(f_alpha_C, gp.kern.alpha_C.values,
                    dx=0.01*gp.kern.alpha_C.values)
    assert abs(v1/v2-1) < relative_accuracy

    v1 = gp.kern.var_C.gradient

    def f_var_C(v):
        gp2 = deepcopy(gp)
        gp2.kern.set_var_C(v)
        return gp2._log_marginal_likelihood
    v2 = derivative(f_var_C, gp.kern.var_C.values,
                    dx=0.01*gp.kern.var_C.values)
    assert abs(v1/v2-1) < relative_accuracy

    v1 = gp.kern.var_L.gradient

    def f_var_L(v):
        gp2 = deepcopy(gp)
        gp2.kern.set_var_L(v)
        return gp2._log_marginal_likelihood
    v2 = derivative(f_var_L, gp.kern.var_L.values,
                    dx=0.01*gp.kern.var_L.values)
    if np.abs(v1) > 1e-7 and np.abs(v2) > 1e-7:
        assert abs(v1/v2-1) < relative_accuracy

    v1 = gp.kern.alpha_T.gradient

    def f_alpha_T(v):
        gp2 = deepcopy(gp)
        gp2.kern.set_alpha_T(v)
        return gp2._log_marginal_likelihood
    v2 = derivative(f_alpha_T, gp.kern.alpha_T.values,
                    dx=0.01*gp.kern.alpha_T.values)
    if np.abs(v1) > 1e-10 and np.abs(v2) > 1e-10:
        assert abs(v1/v2-1) < relative_accuracy

    if gp.prior_z_t is not None:

        v1 = gp.prior_z_t.alpha0.gradient

        def f_z_alpha0(v):
            gp2 = deepcopy(gp)
            gp2.prior_z_t.set_alpha0(v)
            return gp2._log_marginal_likelihood
        v2 = derivative(f_z_alpha0, gp.prior_z_t.alpha0.values,
                        dx=0.01*gp.prior_z_t.alpha0.values)
        assert abs(v1/v2-1) < relative_accuracy

        v1 = gp.prior_z_t.alpha1.gradient

        def f_z_alpha1(v):
            gp2 = deepcopy(gp)
            gp2.prior_z_t.set_alpha1(v)
            return gp2._log_marginal_likelihood
        v2 = derivative(f_z_alpha1, gp.prior_z_t.alpha1.values,
                        dx=0.01*gp.prior_z_t.alpha1.values)
        assert abs(v1/v2-1) < relative_accuracy

    if gp.prior_ell_t is not None:

        v1 = gp.prior_ell_t.alpha0.gradient

        def f_L_alpha0(v):
            gp2 = deepcopy(gp)
            gp2.prior_ell_t.set_alpha0(v)
            return gp2._log_marginal_likelihood
        v2 = derivative(f_L_alpha0, gp.prior_ell_t.alpha0.values,
                        dx=0.01*gp.prior_ell_t.alpha0.values)
        assert abs(v1/v2-1) < relative_accuracy

        v1 = gp.prior_ell_t.alpha1.gradient

        def f_L_alpha1(v):
            gp2 = deepcopy(gp)
            gp2.prior_ell_t.set_alpha1(v)
            return gp2._log_marginal_likelihood
        v2 = derivative(f_L_alpha1, gp.prior_ell_t.alpha1.values,
                        dx=0.01*gp.prior_ell_t.alpha1.values)
        assert abs(v1/v2-1) < relative_accuracy

    if gp.prior_t is not None:

        v1 = gp.prior_t.alpha0.gradient

        def f_T_alpha0(v):
            gp2 = deepcopy(gp)
            gp2.prior_t.set_alpha0(v)
            return gp2._log_marginal_likelihood
        v2 = derivative(f_T_alpha0, gp.prior_t.alpha0.values,
                        dx=0.01*gp.prior_t.alpha0.values)
        assert abs(v1/v2-1) < relative_accuracy

        v1 = gp.prior_t.alpha1.gradient

        def f_T_alpha1(v):
            gp2 = deepcopy(gp)
            gp2.prior_t.set_alpha1(v)
            return gp2._log_marginal_likelihood
        v2 = derivative(f_T_alpha1, gp.prior_t.alpha1.values,
                        dx=0.01*gp.prior_t.alpha1.values)
        assert abs(v1/v2-1) < relative_accuracy

    for dim in range(nObjUnfixed):

        v1 = gp.unfixed_types.gradient[dim]

        def f_t(t):
            gp2 = deepcopy(gp)
            gp2.update_model(False)
            gp2.unfixed_types.values[dim] = t
            gp2.update_model(True)
            return gp2._log_marginal_likelihood
        v2 = derivative(f_t, gp.unfixed_types.values[dim],
                        dx=0.01*gp.unfixed_types.values[dim])
        assert abs(v1/v2-1) < relative_accuracy

        v1 = gp.unfixed_luminosities.gradient[dim]

        def f_ell(t):
            gp2 = deepcopy(gp)
            gp2.update_model(False)
            gp2.unfixed_luminosities.values[dim] = t
            gp2.update_model(True)
            return gp2._log_marginal_likelihood
        v2 = derivative(f_ell, gp.unfixed_luminosities.values[dim],
                        dx=0.01*gp.unfixed_luminosities.values[dim])
        assert abs(v1/v2-1)

        #  TODO: add tests for redshift gradients


def test_optimize(create_gp):
    """Test optimizer"""
    gp = create_gp
    assert(isinstance(gp, PhotozGP))

    #  TODO: let types and redshifts free
    gp.unfixed_types.fix()
    gp.unfixed_redshifts.fix()

    gp.optimize()


def test_hmc(create_gp):
    """Test HMC: initialize the sampler and draw a few points"""
    gp = create_gp
    assert(isinstance(gp, PhotozGP))

    #  TODO: let types and redshifts free
    gp.unfixed_types.fix()
    gp.unfixed_redshifts.fix()
    print gp.parameter_names_flat()
    print gp.optimizer_array

    hmc = GPy.inference.mcmc.HMC(gp, stepsize=1e-2)
    s = hmc.sample(num_samples=3, hmc_iters=2)

# TODO: test covariance is semi positive definite
