# -*- coding: utf-8 -*-
"""Test routines from photoz_kernels.py"""

import numpy as np
from delight.utils import *
from delight.photoz_kernels_cy import kernelparts, kernelparts_diag
from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel

size = 5
NREPEAT = 2
numBands = 2
numLines = 3
numCoefs = 5
relative_accuracy = 0.1


def test_kernel():

    for i in range(NREPEAT):
        X = random_X_bzl(size, numBands=numBands)

        fcoefs_amp, fcoefs_mu, fcoefs_sig \
            = random_filtercoefs(numBands, numCoefs)
        lines_mu, lines_sig = random_linecoefs(numLines)
        var_C, var_L, alpha_C, alpha_L, alpha_T = random_hyperparams()
        print('Failed with params:', var_C, var_L, alpha_C, alpha_L, alpha_T)

        gp = Photoz_kernel(fcoefs_amp, fcoefs_mu, fcoefs_sig,
                           lines_mu, lines_sig, var_C, var_L,
                           alpha_C, alpha_L, alpha_T,
                           use_interpolators=True)


def test_meanfunction():
    """
    Other tests of the mean function
    """
    fcoefs_amp, fcoefs_mu, fcoefs_sig \
        = random_filtercoefs(numBands, numCoefs)
    for i in range(NREPEAT):
        X = random_X_bzl(size, numBands=numBands)
        bands, redshifts, luminosities = np.split(X, 3, axis=1)
        bands = bands.astype(int)
        mf = Photoz_mean_function(0.0, fcoefs_amp, fcoefs_mu, fcoefs_sig)
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
                xfz = xf/oneplusz[k]
                sed = ell * np.exp(-mf.alpha*(xfz-4.5e3))
                fac = oneplusz[k] / mf.DL_z(redshifts[k])**2 / (4*np.pi)
                f_mod[k] += mu[k] * np.trapz(sed*yf, x=xf) \
                    / norms[bands[k]] * fac

        f_mod2 = mf.f(X).ravel()
        assert np.allclose(f_mod, f_mod2, rtol=relative_accuracy)


def test_interpolation():

    for i in range(NREPEAT):

        fcoefs_amp, fcoefs_mu, fcoefs_sig \
            = random_filtercoefs(numBands, numCoefs)
        lines_mu, lines_sig = random_linecoefs(numLines)
        var_C, var_L, alpha_C, alpha_L, alpha_T = random_hyperparams()
        norms = np.sqrt(2*np.pi) * np.sum(fcoefs_amp * fcoefs_sig, axis=1)
        print('Failed with params:', var_C, var_L, alpha_C, alpha_L, alpha_T)

        kern = Photoz_kernel(fcoefs_amp, fcoefs_mu, fcoefs_sig,
                             lines_mu, lines_sig, var_C, var_L,
                             alpha_C, alpha_L, alpha_T)

        for j in range(numBands):

            X = np.vstack((np.repeat(j, kern.nz),
                           kern.redshiftGrid,
                           np.repeat(1, kern.nz),
                           np.repeat(0, kern.nz))).T
            assert X.shape[0] == kern.nz
            assert X.shape[1] == 4

            Kfull = kern.K(X)
            Kdiag = kern.Kdiag(X)
            assert np.allclose(np.diag(Kfull), Kdiag, rtol=relative_accuracy)

            b1 = kern.roundband(X[:, 0])
            fz1 = (1. + X[:, 1])

            kern.construct_interpolators()
            kern.update_kernelparts(X)

            ts = (kern.nz, kern.nz)
            KC, KL = np.zeros(ts), np.zeros(ts)
            D_alpha_C, D_alpha_L, D_alpha_z\
                = np.zeros(ts), np.zeros(ts), np.zeros(ts)
            kernelparts(kern.nz, kern.nz, numCoefs, numLines,
                        alpha_C, alpha_L,
                        fcoefs_amp, fcoefs_mu, fcoefs_sig,
                        lines_mu, lines_sig,
                        norms, b1, fz1, b1, fz1,
                        True, KL, KC,
                        D_alpha_C, D_alpha_L, D_alpha_z)

            assert np.allclose(KL, kern.KL, rtol=relative_accuracy)
            assert np.allclose(KC, kern.KC, rtol=relative_accuracy)
            assert np.allclose(D_alpha_C, kern.D_alpha_C,
                               rtol=relative_accuracy)
            assert np.allclose(D_alpha_L, kern.D_alpha_L,
                               rtol=relative_accuracy)
