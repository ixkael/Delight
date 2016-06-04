

import numpy as np
from delight.utils import *
from delight.photoz_kernels import Photoz
from scipy.misc import derivative

def test_gradients():
    """
    Numerically test the gradients of the kernel with random inputs/parameters.
    """
    size = 4
    numBands = 5
    numLines = 3
    numCoefs = 5
    for i in range(10):
        X = random_X(size, numBands=numBands)
        X2 = random_X(size, numBands=numBands)

        fcoefs_amp, fcoefs_mu, fcoefs_sig = random_filtercoefs(numBands, numCoefs)
        lines_mu, lines_sig = random_linecoefs(numLines)
        var_T, alpha_C, alpha_L, alpha_T = random_hyperparams()
        print var_T, alpha_C, alpha_L, alpha_T

        gp = Photoz(fcoefs_amp, fcoefs_mu, fcoefs_sig, lines_mu, lines_sig,
            var_T, alpha_C, alpha_L, alpha_T)
        dL_dK = 1.0
        gp.update_gradients_full(dL_dK, X, X2)

        v1 = gp.alpha_T.gradient
        def f_alpha_T(alpha_T):
            gp = Photoz(fcoefs_amp, fcoefs_mu, fcoefs_sig, lines_mu, lines_sig,
                var_T, alpha_C, alpha_L, alpha_T)
            return np.sum(gp.K(X,X2))
        v2 = derivative(f_alpha_T, alpha_T, dx=0.01*alpha_T, n=1, args=(), order=5)
        if np.abs(v1) > 1e-15 or np.abs(v2) > 1e-15:
            assert abs(v1/v2-1) < 0.01

        v1 = gp.alpha_L.gradient
        def f_alpha_L(alpha_L):
            gp = Photoz(fcoefs_amp, fcoefs_mu, fcoefs_sig, lines_mu, lines_sig,
                var_T, alpha_C, alpha_L, alpha_T)
            return np.sum(gp.K(X,X2))
        v2 = derivative(f_alpha_L, alpha_L, dx=0.01*alpha_L, n=1, args=(), order=5)
        if np.abs(v1) > 1e-15 or np.abs(v2) > 1e-15:
            assert abs(v1/v2-1) < 0.01

        v1 = gp.alpha_C.gradient
        def f_alpha_C(alpha_C):
            gp = Photoz(fcoefs_amp, fcoefs_mu, fcoefs_sig, lines_mu, lines_sig,
                var_T, alpha_C, alpha_L, alpha_T)
            return np.sum(gp.K(X,X2))
        v2 = derivative(f_alpha_C, alpha_C, dx=0.01*alpha_C, n=1, args=(), order=5)
        if np.abs(v1) > 1e-15 or np.abs(v2) > 1e-15:
            assert abs(v1/v2-1) < 0.01
