
import numpy as np
from paramz.domains import _REAL, _POSITIVE
from GPy.core.parameterization.priors import Prior
import weakref
from scipy.special import gamma, gammaln, polygamma


class Schechter(Prior):
    """
    Schechter luminosity function (normalized)
        p(l|t) = (l/l*)^alpha(t) * exp(l/l*) / l* / Gamma(alpha(t)+1)
        with alpha(t) = alpha0 + alpha1 * t
    """
    domain = _REAL
    _instances = []

    def __new__(cls, ellStar=1.0, alpha0=-0.5, alpha1=0.1):  # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().ellStar == ellStar and instance().alpha0 == alpha0 and instance().alpha1 == alpha1:
                    return instance()
        newfunc = super(Prior, cls).__new__
        if newfunc is object.__new__:
            o = newfunc(cls)
        else:
            o = newfunc(cls, mu, sigma)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, ellStar, alpha0, alpha1):
        self.ellStar = float(ellStar)
        self.lnEllStar = np.log(self.ellStar)
        self.alpha0 = float(alpha0)
        self.alpha1 = float(alpha1)

    def __str__(self):
        return "Schechter({:.2g}, {:.2g}, {:.2g})".format(self.ellStar, self.alpha0, self.alpha1)

    def lnpdf(self, ell, t):
        """Lnprob"""
        alpha = self.alpha0 + self.alpha1 * t
        return - gammaln(1+alpha) - (alpha+1) * self.lnEllStar + alpha * np.log(ell) + ell/self.ellStar

    def lnpdf_grad_ell(self, ell, t):
        """Derivative of lnprob with respect to ell"""
        return 1/self.ellStar + (self.alpha0 + self.alpha1 * t) / ell

    def lnpdf_grad_t(self, ell, t):
        """Derivative of lnprob with respect to t"""
        return self.alpha1 * (np.log(ell) - self.lnEllStar - polygamma(0, 1 + self.alpha0 + self.alpha1 * t))

    def lnpdf_grad_alpha0(self, ell, t):
        """Derivative of lnprob with respect to alpha0"""
        return np.log(ell) - self.lnEllStar - polygamma(0, 1 + self.alpha0 + self.alpha1 * t)

    def lnpdf_grad_alpha1(self, ell, t):
        """Derivative of lnprob with respect to alpha1"""
        return t * (np.log(ell) - self.lnEllStar - polygamma(0, 1 + self.alpha0 + self.alpha1 * t))



class Kumaraswamy(Prior):
    """
    Kumaraswamy distribution
        p(t) = alpha0 * alpha1 * t^(alpha0-1) * (1-t^alpha0)^(alpha1-1)
    """
    domain = _REAL
    _instances = []

    def __new__(cls, alpha0=1.0, alpha1=1.0):  # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().alpha0 == alpha0 and instance().alpha1 == alpha1:
                    return instance()
        newfunc = super(Prior, cls).__new__
        if newfunc is object.__new__:
            o = newfunc(cls)
        else:
            o = newfunc(cls, mu, sigma)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, alpha0, alpha1):
        self.alpha0 = float(alpha0)
        self.alpha1 = float(alpha1)
        self.logalpha0 = np.log(alpha0)
        self.logalpha1 = np.log(alpha1)

    def __str__(self):
        return "Kumaraswamy({:.2g}, {:.2g})".format(self.alpha0, self.alpha1)

    def lnpdf(self, t):
        """Lnprob"""
        return self.logalpha0 + self.logalpha1 + (self.alpha0 - 1) * np.log(t) + (self.alpha1 - 1) * np.log(1 - t**self.alpha0)

    def lnpdf_grad_t(self, t):
        """Derivative of lnprob with respect to t"""
        return (self.alpha0 - 1) / t - (self.alpha0 * (self.alpha1- 1 ) * t**(self.alpha0 - 1)) / (1 - t**self.alpha0)

    def lnpdf_grad_alpha0(self, t):
        """Derivative of lnprob with respect to alpha0"""
        return 1/self.alpha0 + np.log(t) - (self.alpha1 - 1) * t**self.alpha0 * np.log(t) / (1 - t**self.alpha0)

    def lnpdf_grad_alpha1(self, t):
        """Derivative of lnprob with respect to alpha1"""
        return 1/self.alpha1 + np.log(1 - t**self.alpha0)



class Rayleigh(Prior):
    """
    Rayleigh distribution
        p(z|t) = exp(-0.5 * z^2 / alpha(t)^2) / alpha(t)^2
        with alpha(t) = alpha0 + alpha1 * t
    """
    domain = _REAL
    _instances = []

    def __new__(cls, alpha0=1.0, alpha1=1.0):  # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().alpha0 == alpha0 and instance().alpha1 == alpha1:
                    return instance()
        newfunc = super(Prior, cls).__new__
        if newfunc is object.__new__:
            o = newfunc(cls)
        else:
            o = newfunc(cls, mu, sigma)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, alpha0, alpha1):
        self.alpha0 = float(alpha0)
        self.alpha1 = float(alpha1)

    def __str__(self):
        return "Rayleigh({:.2g}, {:.2g})".format(self.alpha0, self.alpha1)

    def lnpdf(self, z, t):
        """Lnprob"""
        alpha2 = (self.alpha0 + self.alpha1 * t)**2.0
        return - alpha2 + np.log(z) - z**2 / 2 / alpha2

    def lnpdf_grad_z(self, z, t):
        """Derivative of lnprob with respect to z"""
        alpha2 = (self.alpha0 + self.alpha1 * t)**2
        return 1.0 / z - z / alpha2

    def lnpdf_grad_t(self, z, t):
        """Derivative of lnprob with respect to t"""
        alpha = (self.alpha0 + self.alpha1 * t)
        return -2 * self.alpha1 * alpha + self.alpha1 * z**2 / alpha**3

    def lnpdf_grad_alpha0(self, z, t):
        """Derivative of lnprob with respect to alpha0"""
        alpha = (self.alpha0 + self.alpha1 * t)
        return -2 * alpha + z**2 / alpha**3

    def lnpdf_grad_alpha1(self, z, t):
        """Derivative of lnprob with respect to alpha1"""
        alpha = (self.alpha0 + self.alpha1 * t)
        return -2 * t * alpha + t * z**2 / alpha**3
