# -*- coding: utf-8 -*-

import numpy as np
from delight.priors import *
from scipy.misc import derivative

NREPEAT = 4
relative_accuracy = 0.05
size = 1


def test_rayleigh():
    """
    Numerically test derivatives of Rayleigh distribution
    one by one for a few random parameters and values
    """
    for i in range(NREPEAT):
        z = np.random.uniform(low=0., high=3.0, size=size)
        alpha = np.random.uniform(low=0.2, high=2.0, size=1)
        dist = Rayleigh(alpha)

        v1 = dist.lnpdf(z)
        v2 = -np.log(dist.pdf(z))
        assert np.all(abs(v1/v2-1) < relative_accuracy)
