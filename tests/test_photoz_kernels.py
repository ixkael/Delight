# -*- coding: utf-8 -*-
"""Test routines from photoz_kernels.py"""

import numpy as np
from scipy.misc import derivative

from delight.utils import random_X_bztl,\
    random_filtercoefs, random_linecoefs, random_hyperparams

from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel


NREPEAT = 4
relative_accuracy = 0.05


def test_kernel_gradients():
    """
    Numerically test the gradients of the kernel with respect to
    hyperparameters using random inputs and hyperparameters.
    """
    size = 4
    numBands = 5
    numLines = 3
    numCoefs = 5

    for i in range(NREPEAT):
        X = random_X_bztl(size, numBands=numBands)
        X2 = random_X_bztl(size, numBands=numBands)

        fcoefs_amp, fcoefs_mu, fcoefs_sig \
            = random_filtercoefs(numBands, numCoefs)
        lines_mu, lines_sig = random_linecoefs(numLines)
        var_T, alpha_C, alpha_L, alpha_T = random_hyperparams()
        print 'Failed with parameters:', var_T, alpha_C, alpha_L, alpha_T

        gp = Photoz_kernel(fcoefs_amp, fcoefs_mu, fcoefs_sig,
                           lines_mu, lines_sig, var_T,
                           alpha_C, alpha_L, alpha_T)
        dL_dK = 1.0
        gp.update_gradients_full(dL_dK, X, X2)

        v1 = gp.alpha_T.gradient

        def f_alpha_T(alpha_T):
            gp = Photoz_kernel(fcoefs_amp, fcoefs_mu, fcoefs_sig, lines_mu,
                               lines_sig, var_T, alpha_C, alpha_L, alpha_T)
            return np.sum(gp.K(X, X2))

        v2 = derivative(f_alpha_T, alpha_T, dx=0.01*alpha_T, order=5)
        if np.abs(v1) > 1e-13 or np.abs(v2) > 1e-13:
            assert abs(v1/v2-1) < relative_accuracy

        v1 = gp.alpha_L.gradient

        def f_alpha_L(alpha_L):
            gp = Photoz_kernel(fcoefs_amp, fcoefs_mu, fcoefs_sig, lines_mu,
                               lines_sig, var_T, alpha_C, alpha_L, alpha_T)
            return np.sum(gp.K(X, X2))

        v2 = derivative(f_alpha_L, alpha_L, dx=0.01*alpha_L, order=5)
        if np.abs(v1) > 1e-13 or np.abs(v2) > 1e-13:
            assert abs(v1/v2-1) < relative_accuracy

        v1 = gp.alpha_C.gradient

        def f_alpha_C(alpha_C):
            gp = Photoz_kernel(fcoefs_amp, fcoefs_mu, fcoefs_sig, lines_mu,
                               lines_sig, var_T, alpha_C, alpha_L, alpha_T)
            return np.sum(gp.K(X, X2))

        v2 = derivative(f_alpha_C, alpha_C, dx=0.01*alpha_C, order=5)
        if np.abs(v1) > 1e-13 or np.abs(v2) > 1e-13:
            assert abs(v1/v2-1) < relative_accuracy


def test_kernel_gradients_X():
    """
    Numerically test the gradients of the kernel with respect to the inputs
     using random inputs and hyperparameters.
    """
    size = 3
    numBands = 5
    numLines = 0
    numCoefs = 1

    for i in range(NREPEAT):
        X = random_X_bztl(size, numBands=numBands)
        X2 = random_X_bztl(size, numBands=numBands)

        fcoefs_amp, fcoefs_mu, fcoefs_sig \
            = random_filtercoefs(numBands, numCoefs)
        lines_mu, lines_sig = random_linecoefs(numLines)
        var_T, alpha_C, alpha_L, alpha_T = random_hyperparams()
        print 'Failed with parameters:', var_T, alpha_C, alpha_L, alpha_T

        gp = Photoz_kernel(fcoefs_amp, fcoefs_mu, fcoefs_sig,
                           lines_mu, lines_sig, var_T,
                           alpha_C, alpha_L, alpha_T)
        dL_dK = 1.0
        grad_XX = gp.gradients_X(dL_dK, X, X)
        grad_X_diag = gp.gradients_X_diag(dL_dK, X)

        np.testing.assert_almost_equal(grad_X_diag, grad_XX)

        v1mat = gp.gradients_X(dL_dK, X, X2)
        v2mat = np.zeros_like(v1mat)
        for dim in [3]:  # 0 for b, 1 for z, 2 for l, 3 for t. Only 0 works.
            for k1 in range(size):
                def f_x(x1):
                    gp = Photoz_kernel(fcoefs_amp, fcoefs_mu, fcoefs_sig,
                                       lines_mu, lines_sig, var_T,
                                       alpha_C, alpha_L, alpha_T)
                    Xb = 1*X
                    Xb[k1, dim] = x1
                    return np.sum(gp.K(Xb, X2))

                v2mat[k1, dim] \
                    = derivative(f_x, X[k1, dim], dx=0.01*X[k1, dim], order=5)

        np.testing.assert_almost_equal(v1mat, v2mat)


def test_meanfunction_gradients_X():
    """
    Numerically test the gradients of the mean function
    with respect to the inputs using random inputs and hyperparameters.
    """
    size = 5

    for i in range(NREPEAT):
        X = random_X_bztl(size)
        mf = Photoz_mean_function()
        dL_dK = 1.0
        v1mat = mf.gradients_X(dL_dK, X)
        v2mat = np.zeros_like(v1mat)
        for dim in [1, 2]:  # Order: b z l t.
            for k1 in range(size):
                def f_x(x1):
                    Xb = 1*X
                    Xb[k1, dim] = x1
                    return np.sum(mf.f(Xb))

                v2mat[k1, dim] \
                    = derivative(f_x, X[k1, dim], dx=0.01*X[k1, dim], order=5)

        np.testing.assert_almost_equal(v1mat, v2mat)


def test_meanfunction():
    """
    Other tests of the mean function
    """
    size = 5

    for i in range(NREPEAT):
        X = random_X_bztl(size)
        mf = Photoz_mean_function()
        print 'mf.f(X).shape', mf.f(X).shape
        assert mf.f(X).shape == (size, 1)
