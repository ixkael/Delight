# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import gamma, gammaln, polygamma
from collections import OrderedDict


# TODO: ADD PARAM RANGES!!!

class Model:
    def __init__(self):
        self.children = []
        self.params = OrderedDict({})

    def set(self, theta):
        assert self.numparams() == len(theta)
        for i, (key, value) in enumerate(self.params.items()):
            self.params[key] = 1*theta[i]
        off = len(self.params)
        for c in self.children:
            n = c.numparams()
            c.set(theta[off:off+n])

    def get(self):
        res = [self.params[key] for key, value in self.params.items()]
        for c in self.children:
            res += c.get()
        return res

    def numparams(self):
        return int(len(self.params) +
                   np.sum([c.numparams() for c in self.children]))


class RayleighRedshiftDistr(Model):
    """
    Rayleigh distribution
        p(z|t) = z * exp(-0.5 * z^2 / alpha(t)^2) / alpha(t)^2
    """
    def __init__(self):
        self.children = []
        alpha = 0.5
        self.params = OrderedDict({'alpha': alpha})

    def __call__(self, z):
        alpha2 = self.params['alpha']**2.0
        return z * np.exp(-0.5 * z**2 / alpha2) / alpha2


class powerLawLuminosityFct(Model):
    """
    Power law luminosity function
    """
    def __init__(self):
        self.children = []
        alpha, phiStar, Mstar = -1.2, 0.01, -20.
        self.params = OrderedDict({'alpha': alpha,
                                   'phiStar': phiStar,
                                   'Mstar': Mstar})

    def __call__(self, z, M):
        edM = 10.**(0.4 * (self.params['Mstar'] - M))
        return np.log(10.) * 0.4 * self.params['phiStar'] *\
            edM**(self.params['alpha'] + 1) / np.exp(edM)


def gaussian(mu, sig):
    return np.exp(-0.5*(mu/sig)**2.0) / np.sqrt(2*np.pi) / sig


class doubleSchechterLuminosityFct(Model):
    """
    Double Schechter luminosity function
    """
    def __init__(self):
        self.children = []
        phiStar1 = 1.56e-02
        alpha1 = -1.66e-01
        phiStar2 = 6.71e-03
        alpha2 = -1.523
        MStar = -2.001e+01
        phiStar3 = 3.08e-05
        Mbright = -2.185e+01
        sigmaBright = 4.84e-01
        P1 = -1.79574
        P2 = -0.266409
        Q = -3.16
        self.params = OrderedDict({
            'P1': P1,
            'P2': P2,
            'Mstar': Mstar,
            'Q': Q,
            'phi1star': phi1star,
            'alpha1': alpha1,
            'phi2star': phi2star,
            'alpha2': alpha2,
            'phi3star': phi3star,
            'Mbright': Mbright,
            'sigmaBright': sigmaBright})

    def __call__(self, z, M):
        opz = 1/(1. + z) - 1/1.1
        ln10d25 = np.log(10) / 2.5
        Qterm = self.params['Q']*(1/(1+z) - 1/1.1)
        dmag = M - self.params['Mstar'] + Qterm
        dmag2 = M - self.params['Mbright'] + Qterm
        t1 = ln10d25 * self.params['phiStar1'] *\
            10**(0.4*(self.params['alpha1']+1)*dmag)
        t2 = ln10d25 * self.params['phiStar2'] *\
            10**(0.4*(self.params['alpha2']+1)*dmag)
        t3 = self.phiStar3 * gaussian(dmag2, self.params['sigmaBright'])
        return 10**(self.P1 + self.P2*z) *\
            ((t1 + t2) * np.exp(-10.**(0.4*dmag)) + t3)


class MultiTypePopulationPrior(Model):
    """
    p(lum, z, t) = p(lum | z, t) * p(z, t) * p(t)
    """
    def __init__(self, numTypes):
        self.numTypes = numTypes
        self.params = OrderedDict(dict(('pt'+str(i+1), 1./numTypes)
                                       for i in range(numTypes)))  # p(t)
        self.lumFct = powerLawLuminosityFct()  # p(lum | z)
        # p(z, t)
        self.nofzs = [RayleighRedshiftDistr() for i in range(numTypes)]
        self.children = [self.lumFct] + self.nofzs

    def coefs():
        alphas = np.zeros((len(self.params), ))
        for i, (key, value) in enumerate(self.params.items()):
            alphas[i] = value
        return alphas

    def __call__(self, redshifts, luminosities):
        absMags = -2.5*np.log10(luminosities)
        lumprior = self.lumFct(redshifts[None, :, None],
                               absMags[None, None, :])
        res = np.zeros((self.numTypes, redshifts.size, luminosities.size))
        for i, (key, alphavalue) in enumerate(self.params.items()):
            res[i, :, :] = alphavalue * lumprior *\
                self.nofzs[i](redshifts[:, None])
        return res  # numtypes * numz * numL
