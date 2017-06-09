# -*- coding: utf-8 -*-

import numpy as np
from delight.utils import *
from delight.photoz_kernels_cy import \
    kernelparts, kernelparts_diag, kernel_parts_interp
from delight.utils_cy import find_positions

size = 50
nz = 150
numBands = 2
numLines = 5
numCoefs = 10
relative_accuracy = 0.1


def test_diagonalOfKernels():
    """
    Test that diagonal of kernels and derivatives are correct across functions.
    """
    X = random_X_bzl(size, numBands=numBands)
    X2 = X

    fcoefs_amp, fcoefs_mu, fcoefs_sig = random_filtercoefs(numBands, numCoefs)
    lines_mu, lines_sig = random_linecoefs(numLines)
    var_C, var_L, alpha_C, alpha_L, alpha_T = random_hyperparams()
    norms = np.sqrt(2*np.pi) * np.sum(fcoefs_amp * fcoefs_sig, axis=1)

    NO1, NO2 = X.shape[0], X2.shape[0]
    b1 = X[:, 0].astype(int)
    b2 = X2[:, 0].astype(int)
    fz1 = 1 + X[:, 1]
    fz2 = 1 + X2[:, 1]
    KC, KL \
        = np.zeros((NO1, NO2)), np.zeros((NO1, NO2))
    D_alpha_C, D_alpha_L, D_alpha_z \
        = np.zeros((NO1, NO2)), np.zeros((NO1, NO2)), np.zeros((NO1, NO2))
    kernelparts(NO1, NO2, numCoefs, numLines,
                alpha_C, alpha_L,
                fcoefs_amp, fcoefs_mu, fcoefs_sig,
                lines_mu[:numLines], lines_sig[:numLines], norms,
                b1, fz1, b2, fz2, True,
                KL, KC,
                D_alpha_C, D_alpha_L, D_alpha_z)

    KC_diag, KL_diag\
        = np.zeros((NO1,)), np.zeros((NO1,))
    D_alpha_C_diag, D_alpha_L_diag = np.zeros((NO1,)), np.zeros((NO1,))
    kernelparts_diag(NO1, numCoefs, numLines,
                     alpha_C, alpha_L,
                     fcoefs_amp, fcoefs_mu, fcoefs_sig,
                     lines_mu[:numLines], lines_sig[:numLines], norms,
                     b1, fz1, True, KL_diag, KC_diag,
                     D_alpha_C_diag, D_alpha_L_diag)

    np.testing.assert_almost_equal(KL_diag, np.diag(KL))
    np.testing.assert_almost_equal(KC_diag, np.diag(KC))
    np.testing.assert_almost_equal(D_alpha_C_diag, np.diag(D_alpha_C))
    np.testing.assert_almost_equal(D_alpha_L_diag, np.diag(D_alpha_L))


def test_find_positions():

    a = np.array([0., 1., 2., 3., 4.])
    b = np.array([0.5, 2.5, 3.0, 3.1, 4.0])
    pos = np.zeros(b.size, dtype=np.long)
    find_positions(b.size, a.size, b, pos, a)
    np.testing.assert_almost_equal(pos, [0, 2, 2, 3, 3])


def test_kernel_parts_interp():

    fcoefs_amp, fcoefs_mu, fcoefs_sig = random_filtercoefs(numBands, numCoefs)
    lines_mu, lines_sig = random_linecoefs(numLines)
    var_C, var_L, alpha_C, alpha_L, alpha_T = random_hyperparams()
    norms = np.sqrt(2*np.pi) * np.sum(fcoefs_amp * fcoefs_sig, axis=1)

    zgrid = np.linspace(0, 3, num=nz)
    opzgrid = 1 + zgrid

    KC_grid, KL_grid =\
        np.zeros((numBands, numBands, nz, nz)),\
        np.zeros((numBands, numBands, nz, nz))
    D_alpha_C_grid, D_alpha_L_grid, D_alpha_z_grid =\
        np.zeros((numBands, numBands, nz, nz)),\
        np.zeros((numBands, numBands, nz, nz)),\
        np.zeros((numBands, numBands, nz, nz))
    for ib1 in range(numBands):
        for ib2 in range(numBands):
            b1 = np.repeat(ib1, nz)
            b2 = np.repeat(ib2, nz)
            fz1 = 1 + zgrid
            fz2 = 1 + zgrid
            kernelparts(nz, nz, numCoefs, numLines,
                        alpha_C, alpha_L,
                        fcoefs_amp, fcoefs_mu, fcoefs_sig,
                        lines_mu[:numLines], lines_sig[:numLines], norms,
                        b1, fz1, b2, fz2, True,
                        KL_grid[ib1, ib2, :, :], KC_grid[ib1, ib2, :, :],
                        D_alpha_C_grid[ib1, ib2, :, :],
                        D_alpha_L_grid[ib1, ib2, :, :],
                        D_alpha_z_grid[ib1, ib2, :, :])

    Xrand = random_X_bzl(size, numBands=numBands)
    X2rand = random_X_bzl(size, numBands=numBands)
    NO1, NO2 = Xrand.shape[0], X2rand.shape[0]
    b1 = Xrand[:, 0].astype(int)
    b2 = X2rand[:, 0].astype(int)
    fz1 = 1 + Xrand[:, 1]
    fz2 = 1 + X2rand[:, 1]

    KC_rand, KL_rand =\
        np.zeros((NO1, NO2)),\
        np.zeros((NO1, NO2))
    D_alpha_C_rand, D_alpha_L_rand, D_alpha_z_rand =\
        np.zeros((NO1, NO2)),\
        np.zeros((NO1, NO2)),\
        np.zeros((NO1, NO2))
    kernelparts(NO1, NO2, numCoefs, numLines,
                alpha_C, alpha_L,
                fcoefs_amp, fcoefs_mu, fcoefs_sig,
                lines_mu[:numLines], lines_sig[:numLines], norms,
                b1, fz1, b2, fz2, True,
                KL_rand, KC_rand,
                D_alpha_C_rand, D_alpha_L_rand, D_alpha_z_rand)

    p1s = np.zeros(size, dtype=int)
    p2s = np.zeros(size, dtype=int)
    find_positions(size, nz, fz1, p1s, opzgrid)
    find_positions(size, nz, fz2, p2s, opzgrid)

    KC_interp, KL_interp =\
        np.zeros((NO1, NO2)),\
        np.zeros((NO1, NO2))
    KC_diag_interp, KL_diag_interp =\
        np.zeros((NO1, )),\
        np.zeros((NO1, ))
    D_alpha_C_interp, D_alpha_L_interp, D_alpha_z_interp =\
        np.zeros((NO1, NO2)),\
        np.zeros((NO1, NO2)),\
        np.zeros((NO1, NO2))

    kernel_parts_interp(size, size,
                        KC_interp,
                        b1, fz1, p1s,
                        b2, fz2, p2s,
                        opzgrid, KC_grid)
    print(np.abs(KC_interp/KC_rand - 1))
    assert np.mean(np.abs(KC_interp/KC_rand - 1)) < relative_accuracy
    assert np.max(np.abs(KC_interp/KC_rand - 1)) < relative_accuracy

    kernel_parts_interp(size, size,
                        D_alpha_C_interp,
                        b1, fz1, p1s,
                        b2, fz2, p2s,
                        opzgrid, D_alpha_C_grid)
    print(np.abs(D_alpha_C_interp/D_alpha_C_rand - 1))
    assert np.mean(np.abs(D_alpha_C_interp/D_alpha_C_rand - 1))\
        < relative_accuracy
    assert np.max(np.abs(D_alpha_C_interp/D_alpha_C_rand - 1))\
        < relative_accuracy
