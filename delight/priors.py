# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import gamma, gammaln, polygamma, gammainc
from collections import OrderedDict
import astropy.cosmology.core
from scipy.misc import derivative
from delight.utils import approx_DL


class Model:
    def __init__(self):
        self.children = []
        self.params = OrderedDict({})
        self.paramranges = OrderedDict({})

    def set(self, theta):
        assert self.numparams() == len(theta)
        for i, (key, value) in enumerate(self.params.items()):
            # print('setting', key, 'to', theta[i])
            self.params[key] = 1*theta[i]
        off = len(self.params)
        for c in self.children:
            n = c.numparams()
            c.set(theta[off:off+n])
            off += n

    def get(self):
        res = [self.params[key] for key, value in self.params.items()]
        # [print('getting', key, ':', self.params[key])
        #  for key, value in self.params.items()]
        for c in self.children:
            res += c.get()
        return res

    def get_ranges(self):
        res = [self.paramranges[key]
               for key, value in self.paramranges.items()]
        for c in self.children:
            res += c.get_ranges()
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
        self.paramranges = OrderedDict({'alpha': [0., 2.0]})

    def __call__(self, z):
        alpha2 = self.params['alpha']**2.0
        return z * np.exp(-0.5 * z**2 / alpha2) / alpha2


class ComovingVolumeEfect(Model):
    def __init__(self):
        self.cosmo = astropy.cosmology.core.FlatLambdaCDM(70, 0.3)
        self.children = []
        self.params = OrderedDict({})
        self.paramranges = OrderedDict({})
        self.zgrid = np.logspace(-5, 1, 100)
        self.comovol = self.cosmo.comoving_volume(self.zgrid).value

    def __call__(self, z):
        return np.interp(z, self.zgrid, self.comovol)


class powerLawLuminosityFct(Model):
    """
    Power law luminosity function
    """
    def __init__(self):
        self.children = []
        alpha, self.phiStar, self.ellStar = -1.2, 0.01, 10.**8.
        self.params = OrderedDict({'alpha': alpha})  # 'ellStar': ellStar})
        self.paramranges = OrderedDict({'alpha': [-1.5, -1.1]})
        # 'ellStar': [10.**8, 10.**10]})

    def __call__(self, z, ell):
        edl = ell/self.ellStar
        alpha = self.params['alpha']
        return edl**(alpha+1) * np.exp(-edl)

    def jac(self, z, ell):
        edl = ell/self.ellStar
        alpha = self.params['alpha']
        return edl**(alpha+1) * np.exp(-edl) * np.log(ell/self.ellStar)


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
    def __init__(self, numTypes, maglim=None):
        self.numTypes = numTypes
        self.params = OrderedDict({})
        self.paramranges = OrderedDict({})
        for i in range(numTypes-1):
            self.params['pt'+str(i+1)] = 0.5
            self.paramranges['pt'+str(i+1)] = [0.2, 0.9]
        if maglim is not None:
            self.maglim = maglim
            self.DL = approx_DL()
        else:
            self.maglim = None
        self.lumFct = powerLawLuminosityFct()  # p(lum | z)
        # p(z, t)
        self.nofz = ComovingVolumeEfect()
        self.children = [self.lumFct] + [self.nofz]

    def hypercube2simplex(self, zs):
        fac = np.concatenate((1 - zs, np.array([1])))
        zsb = np.concatenate((np.array([1]), zs))
        fs = np.cumprod(zsb) * fac
        return fs

    def hypercube2simplex_jacobian(self, fs, zs):
        jaco = np.zeros((zs.size, fs.size))
        for j in range(fs.size):
            for i in range(zs.size):
                if i < j:
                    jaco[i, j] = fs[j] / zs[i]
                if i == j:
                    jaco[i, j] = fs[j] / (zs[j] - 1)
        return jaco

    def coefs(self):
        zs = np.array([self.params['pt'+str(i+1)]
                       for i in range(self.numTypes-1)])
        return self.hypercube2simplex(zs)

    def detprob(self, redshifts, luminosities):
        fluxes = luminosities * (1 + redshifts) /\
            (4 * np.pi * self.DL(redshifts)**2. * 1e10)
        mags = - 2.5*np.log10(fluxes)
        magp = self.maglim - 0.4
        dets = np.exp(-0.5*((mags-magp)/0.4)**2)
        dets[mags <= magp] = 1.
        return dets  # numz * numL

    def gridflat(self, redshifts, luminosities, detprob=None):
        res = self.coefs()[:, None] * self.nofz(redshifts[None, :]) *\
            self.lumFct(redshifts[None, :], luminosities[None, :])
        if self.maglim is not None and detprob is None:
            res *= self.detprob(redshifts[None, :],
                                luminosities[None, :])
        if self.maglim is not None and detprob is not None:
            res *= detprob[None, :]
        return res  # numtypes * numz * numL

    def gridflat_grad(self, redshifts, luminosities, detprob=None):
        zs = np.array([self.params['pt'+str(i+1)]
                       for i in range(self.numTypes-1)])
        fs = self.hypercube2simplex(zs)
        fs2zs_grad = self.hypercube2simplex_jacobian(fs, zs)
        grid = self.gridflat(redshifts, luminosities, detprob=detprob)
        grads = np.zeros((self.numparams(),
                          grid.shape[0], grid.shape[1]))
        grads[0:self.numTypes-1, :, :] = fs2zs_grad[:, :, None] *\
            grid[None, :, :] / fs[None, :, None]
        grads[self.numTypes-1, :, :] = fs[:, None] *\
            self.nofz(redshifts[None, :]) *\
            self.lumFct.jac(redshifts[None, :], luminosities[None, :])
        if self.maglim is not None and detprob is None:
            grads[self.numTypes-1, :, :] *= \
                self.detprob(redshifts[None, :], luminosities[None, :])
        if self.maglim is not None and detprob is not None:
            grads[self.numTypes-1, :, :] *= detprob[None, :]
        return grads  # numpars * numtypes * numzL

    def grid(self, redshifts, luminosities, detprob=None):
        res = self.coefs()[:, None, None] *\
            self.nofz(redshifts[None, :, None]) *\
            self.lumFct(redshifts[None, :, None], luminosities[None, None, :])
        if self.maglim is not None and detprob is None:
            res *= self.detprob(redshifts[None, :, None],
                                luminosities[None, None, :])
        if self.maglim is not None and detprob is not None:
            res *= detprob[None, :, :]
        return res  # numtypes * numz * numL

    def __call__(self, types, redshifts, luminosities):
        lumprior = self.lumFct(redshifts, luminosities)
        nobj = types.size
        res = np.zeros((nobj, ))
        alphavalues = self.coefs()
        for i in range(self.numTypes):
            ind = types == i
            res[ind] = alphavalues[i] * lumprior[ind] *\
                self.nofz(redshifts[ind])
        if self.maglim is not None:
            res *= self.detprob(redshifts, luminosities)
        return res  # nobj

    def draw(self, nobj, redshiftGrid, luminosityGrid):
        grid = self.grid(redshiftGrid, luminosityGrid)
        cumgrid = np.concatenate(([0], np.cumsum(grid.flatten())))
        vals = np.random.uniform(low=0, high=cumgrid[-1], size=nobj)
        types = np.repeat(-1, nobj)
        redshifts = np.repeat(-1.0, nobj)
        luminosities = np.repeat(-1.0, nobj)
        for i in range(cumgrid.size - 1):
            ind = np.logical_and(vals > cumgrid[i], vals <= cumgrid[i+1])
            if ind.sum() > 0:
                off = 1
                while(cumgrid[i-off] == cumgrid[i]):
                    off += 1
                if off > 1:
                    locs = np.random.randint(low=i-off+1, high=i,
                                             size=np.sum(ind))
                else:
                    locs = np.repeat(i, np.sum(ind))
                indices = np.vstack(np.unravel_index(locs, grid.shape)).T
                types[ind] = indices[:, 0]
                redshifts[ind] = redshiftGrid[indices[:, 1]]
                luminosities[ind] = luminosityGrid[indices[:, 2]]
        return types, redshifts, luminosities
