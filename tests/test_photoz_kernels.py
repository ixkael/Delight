# -*- coding: utf-8 -*-
"""Test routines from photoz_kernels.py"""

import numpy as np
from scipy.misc import derivative
from copy import deepcopy as copy

from delight.utils import random_X_bzlt,\
    random_filtercoefs, random_linecoefs, random_hyperparams

from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel

size = 6
NREPEAT = 4
numBands = 5
numLines = 10
numCoefs = 10
relative_accuracy = 0.05
# TODO: add tests for diagonal gradients of kernel?
# TODO: add formal/numerical test for kernel w.r.t. mean fct
# TODO: add tests for with and without caching! fixture for random X


def test_kernel():

    for i in range(NREPEAT):
        X = random_X_bzlt(size, numBands=numBands)

        fcoefs_amp, fcoefs_mu, fcoefs_sig \
            = random_filtercoefs(numBands, numCoefs)
        lines_mu, lines_sig = random_linecoefs(numLines)
        var_C, var_L, alpha_C, alpha_L, alpha_T = random_hyperparams()
        print 'Failed with params:', var_C, var_L, alpha_C, alpha_L, alpha_T

        gp = Photoz_kernel(fcoefs_amp, fcoefs_mu, fcoefs_sig,
                           lines_mu, lines_sig, var_C, var_L,
                           alpha_C, alpha_L, alpha_T)

        v1 = np.diag(gp.K(X))
        v2 = gp.Kdiag(X)

        np.testing.assert_almost_equal(v1, v2)


def test_kernel_gradients():
    """
    Numerically test the gradients of the kernel with respect to
    hyperparameters using random inputs and hyperparameters.
    """

    for i in range(NREPEAT):
        X = random_X_bzlt(size, numBands=numBands)
        X2 = random_X_bzlt(size, numBands=numBands)

        fcoefs_amp, fcoefs_mu, fcoefs_sig \
            = random_filtercoefs(numBands, numCoefs)
        lines_mu, lines_sig = random_linecoefs(numLines)
        var_C, var_L, alpha_C, alpha_L, alpha_T = random_hyperparams()
        print 'Failed with params:', var_C, var_L, alpha_C, alpha_L, alpha_T

        gp = Photoz_kernel(fcoefs_amp, fcoefs_mu, fcoefs_sig,
                           lines_mu, lines_sig, var_C, var_L,
                           alpha_C, alpha_L, alpha_T)
        dL_dK = 1
        gp.update_gradients_full(dL_dK, X, X2)

        v1 = gp.alpha_T.gradient

        def f_alpha_T(alpha_T):
            gp2 = copy(gp)
            gp2.set_alpha_T(alpha_T)
            return np.sum(gp2.K(X, X2))

        v2 = derivative(f_alpha_T, alpha_T, dx=0.01*alpha_T)
        if np.abs(v1) > 1e-13 or np.abs(v2) > 1e-13:
            assert abs(v1/v2-1) < relative_accuracy

        v1 = gp.alpha_L.gradient

        def f_alpha_L(alpha_L):
            gp2 = copy(gp)
            gp2.set_alpha_L(alpha_L)
            return np.sum(gp2.K(X, X2))

        v2 = derivative(f_alpha_L, alpha_L, dx=0.01*alpha_L)
        if np.abs(v1) > 1e-13 or np.abs(v2) > 1e-13:
            assert abs(v1/v2-1) < relative_accuracy

        v1 = gp.alpha_C.gradient

        def f_alpha_C(alpha_C):
            gp2 = copy(gp)
            gp2.set_alpha_C(alpha_C)
            return np.sum(gp2.K(X, X2))

        v2 = derivative(f_alpha_C, alpha_C, dx=0.01*alpha_C)
        if np.abs(v1) > 1e-13 or np.abs(v2) > 1e-13:
            assert abs(v1/v2-1) < relative_accuracy

        v1 = gp.var_C.gradient

        def f_var_C(var_C):
            gp = Photoz_kernel(fcoefs_amp, fcoefs_mu, fcoefs_sig, lines_mu,
                               lines_sig, var_C, var_L,
                               alpha_C, alpha_L, alpha_T)
            return np.sum(gp.K(X, X2))

        v2 = derivative(f_var_C, var_C, dx=0.01*var_C)
        if np.abs(v1) > 1e-13 or np.abs(v2) > 1e-13:
            assert abs(v1/v2-1) < relative_accuracy

        v1 = gp.var_L.gradient

        def f_var_L(var_L):
            gp = Photoz_kernel(fcoefs_amp, fcoefs_mu, fcoefs_sig, lines_mu,
                               lines_sig, var_C, var_L,
                               alpha_C, alpha_L, alpha_T)
            return np.sum(gp.K(X, X2))

        v2 = derivative(f_var_L, var_L, dx=0.01*var_L)
        if np.abs(v1) > 1e-13 or np.abs(v2) > 1e-13:
            assert abs(v1/v2-1) < relative_accuracy


def test_kernel_Xgradients():
    """
    Numerically test the gradients of the kernel with respect to the inputs
     using random inputs and hyperparameters.
    """
    for i in range(NREPEAT):
        X = random_X_bzlt(size, numBands=numBands)
        X2 = random_X_bzlt(size, numBands=numBands)

        fcoefs_amp, fcoefs_mu, fcoefs_sig \
            = random_filtercoefs(numBands, numCoefs)
        lines_mu, lines_sig = random_linecoefs(numLines)
        var_C, var_L, alpha_C, alpha_L, alpha_T = random_hyperparams()
        print 'Failed with params:', var_C, var_L, alpha_C, alpha_L, alpha_T

        gp = Photoz_kernel(fcoefs_amp, fcoefs_mu, fcoefs_sig,
                           lines_mu, lines_sig, var_C, var_L,
                           alpha_C, alpha_L, alpha_T)
        dL_dK = 1.0
        grad_XX = gp.gradients_X(dL_dK, X, X)
        grad_X_diag = gp.gradients_X_diag(dL_dK, X)
        np.testing.assert_almost_equal(grad_X_diag, grad_XX)

        v1mat = gp.gradients_X(dL_dK, X, X2)
        v2mat = np.zeros_like(v1mat)
        for dim in [2, 3]:  # 0 for b, 1 for z, 2 for l, 3 for t. Only 3 works.
            print 'Dimension', dim
            for k1 in range(size):
                def f_x(x1):
                    gp = Photoz_kernel(fcoefs_amp, fcoefs_mu, fcoefs_sig,
                                       lines_mu, lines_sig, var_C, var_L,
                                       alpha_C, alpha_L, alpha_T)
                    Xb = 1*X
                    Xb[k1, dim] = x1
                    return np.sum(gp.K(Xb, X2))

                v2mat[k1, dim] \
                    = derivative(f_x, X[k1, dim], dx=0.01*X[k1, dim])
                assert abs(v1mat[k1, dim]/v2mat[k1, dim]-1)\
                    < relative_accuracy


def test_meanfunction_gradients_X():
    """
    Numerically test the gradients of the mean function
    with respect to the inputs using random inputs and hyperparameters.
    """
    fcoefs_amp, fcoefs_mu, fcoefs_sig \
        = random_filtercoefs(numBands, numCoefs)
    for i in range(NREPEAT):
        alpha = np.random.uniform(low=0, high=2e-3, size=1)
        beta = np.random.uniform(low=1, high=3, size=1)
        X = random_X_bzlt(size)
        mf = Photoz_mean_function(alpha, beta,
                                  fcoefs_amp, fcoefs_mu, fcoefs_sig)
        dL_dm = np.ones((size, 1))
        dL_dK = np.diag(dL_dm)
        mf.update_gradients(dL_dm, X)

        v1 = mf.alpha.gradient

        def f_alpha(alpha):
            mf2 = copy(mf)
            mf2.set_alpha(alpha)
            return np.sum(mf2.f(X))

        v2 = derivative(f_alpha, alpha, dx=0.01*alpha)
        assert abs(v1/v2-1) < relative_accuracy

        v1 = mf.beta.gradient

        def f_beta(beta):
            mf2 = copy(mf)
            mf2.set_beta(beta)
            return np.sum(mf2.f(X))

        v2 = derivative(f_beta, beta, dx=0.01*beta)
        assert abs(v1/v2-1) < relative_accuracy

        v1mat = mf.gradients_X(dL_dK, X)
        v2mat = np.zeros_like(v1mat)
        for dim in [2, 3]:  # Order: b z l t.
            print dim
            for k1 in range(size):
                def f_x(x1):
                    Xb = 1*X
                    Xb[k1, dim] = x1
                    return np.sum(mf.f(Xb))

                v2mat[k1, dim] \
                    = derivative(f_x, X[k1, dim], dx=0.01*X[k1, dim])
                assert abs(v1mat[k1, dim]/v2mat[k1, dim]-1) < relative_accuracy


def test_meanfunction():
    """
    Other tests of the mean function
    """
    fcoefs_amp, fcoefs_mu, fcoefs_sig \
        = random_filtercoefs(numBands, numCoefs)
    for i in range(NREPEAT):
        alpha = np.random.uniform(low=0, high=2e-3, size=1)
        beta = np.random.uniform(low=1., high=3., size=1)
        X = random_X_bzlt(size)
        bands, redshifts, luminosities, types = np.split(X, 4, axis=1)
        bands = bands.astype(int)
        mf = Photoz_mean_function(alpha, beta,
                                  fcoefs_amp, fcoefs_mu, fcoefs_sig)
        assert mf.f(X).shape == (size, 1)

        f_mod = np.zeros((size, ))
        oneplusz = 1 + redshifts
        norms = np.sqrt(2*np.pi) * np.sum(fcoefs_amp * fcoefs_sig, axis=1)
        for i in range(numCoefs):
            amp, mu, sig = fcoefs_amp[bands, i],\
                           fcoefs_mu[bands, i],\
                           fcoefs_sig[bands, i]
            for k in range(size):
                ell = luminosities[k]
                lambdaMin = mu[k] - 4*sig[k]
                lambdaMax = mu[k] + 4*sig[k]
                xf = np.linspace(lambdaMin, lambdaMax, num=200)
                yf = amp[k] * np.exp(-0.5*((xf-mu[k])/sig[k])**2)
                sed = ell*np.exp(-alpha*types[k]**beta*(xf/oneplusz[k]-4.5e3))
                fac = oneplusz[k] / mf.DL_z(redshifts[k])**2 / (4*np.pi)
                f_mod[k] += np.trapz(sed*yf, x=xf) / norms[bands[k]] * fac

        f_mod2 = mf.f(X).ravel()
        np.allclose(f_mod, f_mod2, rtol=relative_accuracy)
