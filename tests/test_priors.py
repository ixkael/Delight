# -*- coding: utf-8 -*-

import numpy as np
from delight.priors import *
from scipy.misc import derivative
from delight.utils import derivative_test


class SimpleChildModel(Model):
    def __init__(self):
        self.children = []
        self.params = OrderedDict({'a': 2.0})


class SimpleParentModel(Model):
    def __init__(self):
        self.children = [SimpleChildModel()]
        self.params = OrderedDict({'b': 3.0, 'e': 2.0})


class SimpleGrandParentModel(Model):
    def __init__(self):
        self.children = [SimpleParentModel()]
        self.params = OrderedDict({'c': 4.0})


def test_SimpleModel():
    """Test the model hierarchy, setters and getters"""
    mod = SimpleGrandParentModel()
    theta = [0] * mod.numparams()
    mod.set(theta)
    assert mod.get() == theta


def test_RayleighRedshiftDistr():
    mod = RayleighRedshiftDistr()
    alpha = 2.0
    mod.set([alpha])
    assert mod.get() == [alpha]
    z = 2.0
    res = z * np.exp(-0.5 * z**2 / alpha**2) / alpha**2
    assert mod(z) == res


def test_powerLawLuminosityFct():
    mod = powerLawLuminosityFct()
    theta = np.array([-1.2])
    mod.set(theta)
    assert mod.get() == theta
    z = 4.0
    res = 1 / np.exp(1.0)
    assert mod(z, mod.ellStar) == res
    ell = 1.1*mod.ellStar

    def prob(alpha):
        mod.set(alpha)
        return mod(z, ell)

    def prob_grad(alpha):
        mod.set(alpha)
        return np.array([mod.jac(z, ell)])

    relative_accuracy = 0.01
    derivative_test(theta, prob, prob_grad, relative_accuracy)


def test_MultiTypePopulationPrior():
    numTypes, nz, nl = 3, 50, 50
    mod = MultiTypePopulationPrior(numTypes)
    ntot = numTypes * 1 - 1 + 1
    assert mod.numparams() == ntot
    theta = [0]*ntot
    mod.set(theta)
    assert mod.get() == theta

    mod = MultiTypePopulationPrior(numTypes, maglim=24)
    print(mod.get())
    theta = np.array(mod.get())
    redshifts = np.linspace(1e-2, 2, nz)
    luminosities = np.linspace(1e7, 1e9, nl)
    z_grid, l_grid = np.meshgrid(redshifts, luminosities)
    z_grid, l_grid = z_grid.ravel(), l_grid.ravel()
    grid = mod.grid(redshifts, luminosities)
    assert grid.shape[0] == numTypes
    assert grid.shape[1] == nz
    assert grid.shape[2] == nl

    grid2 = 0*grid
    for i in range(numTypes):
        zz, ll = np.meshgrid(redshifts, luminosities, indexing='ij')
        types = np.repeat(i, zz.ravel().size)
        grid2[i, :, :] = mod(types, zz.ravel(), ll.ravel()).reshape(zz.shape)
    assert np.allclose(grid, grid2)

    absMags = -2.5*np.log(luminosities)
    types2, redshifts2, luminosities2 = mod.draw(100, redshifts, luminosities)
    assert np.all(types2 >= 0)
    assert np.all(redshifts2 >= 0)
    assert np.all(luminosities2 >= 0)
    from copy import deepcopy

    for it in range(numTypes):
        for i in range(10):
            def prob(x):
                mod2 = deepcopy(mod)
                mod2.set(x)
                return mod2.gridflat(redshifts, luminosities)[it, i]

            def prob_grad(x):
                mod2 = deepcopy(mod)
                mod2.set(x)
                return mod2.gridflat_grad(redshifts, luminosities)[:, it, i]

            relative_accuracy = 0.01
            derivative_test(theta, prob, prob_grad, relative_accuracy,
                            dxfac=1e-2, order=15, lim=1e-4, superverbose=True)

    def prob(x):
        mod2 = deepcopy(mod)
        mod2.set(x)
        return np.sum(mod.gridflat(redshifts, luminosities))

    def prob_grad(x):
        mod2 = deepcopy(mod)
        mod2.set(x)
        return np.sum(mod.gridflat_grad(redshifts, luminosities), axis=(1, 2))

    relative_accuracy = 0.01
    print(prob_grad(theta))
    derivative_test(theta, prob, prob_grad, relative_accuracy,
                    dxfac=1e-1, order=15, lim=1e6, superverbose=True)
    # assert 0
