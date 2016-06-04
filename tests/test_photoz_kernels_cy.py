
import numpy as np
from delight.utils import *
from delight.photoz_kernels_cy import kernelparts, kernelparts_diag

def test_diagonalOfKernels():
    """
    Test that diagonal of kernels and derivatives are correct across functions.
    """
    size = 5
    numBands = 5
    numLines = 4
    numCoefs = 5
    X = random_X(size, numBands=numBands)
    X2 = X

    fcoefs_amp, fcoefs_mu, fcoefs_sig = random_filtercoefs(numBands, numCoefs)
    lines_mu, lines_sig = random_linecoefs(numLines)
    var_T, alpha_C, alpha_L, alpha_T = random_hyperparams()
    norms = np.sqrt(2*np.pi) * np.sum(fcoefs_amp * fcoefs_sig, axis=1)

    NO1, NO2 = X.shape[0], X2.shape[0]
    t1 = X[:,0]
    t2 = X2[:,0]
    b1 = X[:,1].astype(int)
    b2 = X2[:,1].astype(int)
    fz1 = 1 + X[:,2]
    fz2 = 1 + X2[:,2]
    norm1, norm2 = np.zeros((NO1,)), np.zeros((NO2,))
    KT, KC, KL = np.zeros((NO1, NO2)), np.zeros((NO1, NO2)), np.zeros((NO1, NO2))
    D_alpha_C, D_alpha_L, D_alpha_z = np.zeros((NO1, NO2)), np.zeros((NO1, NO2)), np.zeros((NO1, NO2))
    kernelparts(NO1, NO2, numCoefs, numLines,
        alpha_C, alpha_L, alpha_T,
        fcoefs_amp, fcoefs_mu, fcoefs_sig, lines_mu[:numLines], lines_sig[:numLines],
        t1, b1, fz1, t2, b2, fz2, True, norm1, norm2, KL, KC, KT, D_alpha_C, D_alpha_L, D_alpha_z)

    KT_diag, KC_diag, KL_diag = np.zeros((NO1,)), np.zeros((NO1,)), np.zeros((NO1,))
    D_alpha_C_diag, D_alpha_L_diag = np.zeros((NO1,)), np.zeros((NO1,))
    kernelparts_diag(NO1, numCoefs, numLines,
        alpha_C, alpha_L, alpha_T,
        fcoefs_amp, fcoefs_mu, fcoefs_sig, lines_mu[:numLines], lines_sig[:numLines],
        t1, b1, fz1, True, norm1, KL_diag, KC_diag, KT_diag, D_alpha_C_diag, D_alpha_L_diag)

    np.testing.assert_almost_equal(KT_diag, np.diag(KT))
    np.testing.assert_almost_equal(KL_diag, np.diag(KL))
    np.testing.assert_almost_equal(KC_diag, np.diag(KC))
    np.testing.assert_almost_equal(D_alpha_C_diag, np.diag(D_alpha_C))
    np.testing.assert_almost_equal(D_alpha_L_diag, np.diag(D_alpha_L))
