
import numpy as np
from .photoz_kernels_cy import kernelparts
from .photoz_kernels_cy import kernelparts_diag

def test_diagonalOfKernels():
    X = np.zeros
    X2 = X
    NO1, NO2 = X.shape[0], X2.shape[0]
    t1 = X[:,0]
    t2 = X2[:,0]
    b1 = self.roundband(X[:,1])
    b2 = self.roundband(X2[:,1])
    fz1 = 1 + X[:,2]
    fz2 = 1 + X2[:,2]
    norm1, norm2 = np.zeros((NO1,)), np.zeros((NO2,))
    KT, KC, KL = np.zeros((NO1, NO2)), np.zeros((NO1, NO2)), np.zeros((NO1, NO2))
    D_alpha_C, D_alpha_L = np.zeros((NO1, NO2)), np.zeros((NO1, NO2))
    kernelparts_fast(NO1, NO2, self.numCoefs, self.numLines,
        self.alpha_C, self.alpha_L, self.alpha_T,
        self.fcoefs_amp, self.fcoefs_mu, self.fcoefs_sig, self.lines_mu[:self.numLines], self.lines_sig[:self.numLines],
        t1, b1, fz1, t2, b2, fz2, True, norm1, norm2, KL, KC, KT, D_alpha_C, D_alpha_L)
    self.slope_T.gradient = np.sum(dL_dK * t1[:,None] * t2[None,:] * KT * (KC + KL))
    self.var_T.gradient = np.sum(dL_dK * KT * (KC + KL))
    Tpart = (self.var_T + self.slope_T * t1[:,None] * t2[None,:]) * KT
    self.alpha_C.gradient = np.sum(dL_dK * Tpart * D_alpha_C)
    self.alpha_L.gradient = np.sum(dL_dK * Tpart * D_alpha_L)
    self.alpha_T.gradient = np.sum(dL_dK * (t1[:,None]-t2[None,:])**2 / self.alpha_T**3 * Tpart * (KC + KL))

    KT_diag, KC_diag, KL_diag = np.zeros((NO1,)), np.zeros((NO1,)), np.zeros((NO1,))
    D_alpha_C_diag, D_alpha_L_diag = np.zeros((NO1,)), np.zeros((NO1,))
    kernelparts_diag_fast(NO1, self.numCoefs, self.numLines,
        self.alpha_C, self.alpha_L, self.alpha_T,
        self.fcoefs_amp, self.fcoefs_mu, self.fcoefs_sig, self.lines_mu[:self.numLines], self.lines_sig[:self.numLines],
        t1, b1, fz1, False, norm1, KL_diag, KC_diag, KT_diag, D_alpha_C_diag, D_alpha_L_diag)

    np.testing.assert_array_equal(KL_diag, np.diag(KL))
    np.testing.assert_array_equal(KC_diag, np.diag(KC))
    np.testing.assert_array_equal(KT_diag, np.diag(KT))
    np.testing.assert_array_equal(D_alpha_C_diag, np.diag(D_alpha_C))
    np.testing.assert_array_equal(D_alpha_L_diag, np.diag(D_alpha_L))
