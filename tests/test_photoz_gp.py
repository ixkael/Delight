# -*- coding: utf-8 -*-

import pytest
import numpy as np
from scipy.interpolate import interp1d

from delight.photoz_gp import PhotozGP
from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel
from delight.utils import *

NREPEAT = 1
nObj = 5
nObjGP = 3
nObjUnfixed = 4
numBands = 2
numLines = 3
numCoefs = 8
numTemplates = 4
numZ = 10
relative_accuracy = 0.50
size = numBands * nObj
bandsUsed = range(numBands)

X = random_X_bzl(nObj, numBands=numBands)
bands, redshifts, luminosities = np.split(X, 3, axis=1)
fcoefs_amp, fcoefs_mu, fcoefs_sig \
    = random_filtercoefs(numBands, numCoefs)
lines_mu, lines_sig = random_linecoefs(numLines)
var_C, var_L, alpha_C, alpha_L, alpha_T = random_hyperparams()
Y = np.random.uniform(low=0.5, high=1., size=nObj)[:, None]
Yvar = np.random.uniform(low=0.05, high=0.1, size=nObj)[:, None]


@pytest.fixture(params=[True, False])
def use_interpolators(request):
    return request.param


def test_create_gp(use_interpolators):
    """Create valid GP with reasonable parameters, kernel, mean fct"""

    redshiftGrid = np.logspace(-2, np.log10(4), num=numZ)
    f_mod_interp = np.zeros((numTemplates, numBands), dtype=object)
    for it in range(numTemplates):
        for jf in range(numBands):
            data = np.random.randn(numZ)
            f_mod_interp[it, jf] = interp1d(redshiftGrid, data,
                                            kind='linear', bounds_error=False,
                                            fill_value='extrapolate')
    gp = PhotozGP(
        f_mod_interp,
        fcoefs_amp, fcoefs_mu, fcoefs_sig,
        lines_mu, lines_sig,
        var_C, var_L, alpha_C, alpha_L,
        redshiftGrid,
        use_interpolators=use_interpolators
        )
    gp.setData(X, Y, Yvar, np.random.randint(numTemplates))

    if use_interpolators is False:
        gp.optimizeHyperparamaters()

    return gp
