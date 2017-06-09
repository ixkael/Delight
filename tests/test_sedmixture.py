# -*- coding: utf-8 -*-

import numpy as np
from delight.sedmixture import *
from scipy.misc import derivative

relative_accuracy = 0.01


def test_PhotometricFilter():
    def f(x):
        return np.exp(-0.5*((x-3e3)/1e2)**2)
    x = np.linspace(2e3, 4e3, 1000)
    y = f(x)
    aFilter = PhotometricFilter('I', x, y)
    xb = np.random.uniform(low=2e3, high=4e3, size=10)
    res1 = f(xb)
    res2 = aFilter(xb)
    assert np.allclose(res2, res1, rtol=relative_accuracy)


def test_PhotometricFluxPolynomialInterpolation():

    def f(x):
        return np.exp(-0.5*((x-3e3)/1e2)**2)
    x = np.linspace(2e3, 4e3, 1000)
    y = f(x)
    bandName = 'I'
    photometricBands = [PhotometricFilter(bandName, x, y)]
    x = np.linspace(2e1, 4e5, 1000)
    y = f(x)
    aTemplate = SpectralTemplate_z(x, y, photometricBands,
                                   redshiftGrid=np.linspace(1e-2, 1.0, 10))

    redshifts = np.random.uniform(1e-2, 1.0, 10)

    f1 = aTemplate.photometricFlux(redshifts, bandName)
    f2 = aTemplate.photometricFlux_bis(redshifts, bandName)

    f1 = aTemplate.photometricFlux_gradz(redshifts, bandName)
    f2 = aTemplate.photometricFlux_gradz_bis(redshifts, bandName)
