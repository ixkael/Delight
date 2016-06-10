
import numpy as np
from paramz.domains import _REAL
from GPy.core.parameterization.priors import Prior, Parameterized, Param
import weakref
from scipy.special import gamma, gammaln, polygamma


class Schechter(Parameterized):
    """
    Schechter luminosity function (normalized)
        p(l|t) = (l/l*)^alpha(t) * exp(l/l*) / l* / Gamma(alpha(t)+1)
        with alpha(t) = alpha0 + alpha1 * t
    """
    domain = _REAL

    def __init__(self, ellStar, alpha0, alpha1, name='Schechter'):
        super(Schechter, self).__init__(name=name)
        self.ellStar = Param('ellStar', float(ellStar))
        self.alpha0 = Param('alpha0', float(alpha0))
        self.alpha1 = Param('alpha1', float(alpha1))
        self.link_parameter(self.ellStar)
        self.link_parameter(self.alpha0)
        self.link_parameter(self.alpha1)
        self.lnEllStar = np.log(ellStar)

    def set_ellStar(self, ellStar):
        """Set ellStar"""
        self.update_model(False)
        index = self.ellStar._parent_index_
        self.unlink_parameter(self.ellStar)
        self.ellStar = Param('ellStar', float(ellStar))
        self.link_parameter(self.ellStar, index=index)
        self.lnEllStar = np.log(ellStar)
        self.update_model(True)

    def set_alpha0(self, alpha0):
        """Set alpha0"""
        self.update_model(False)
        index = self.alpha0._parent_index_
        self.unlink_parameter(self.alpha0)
        self.alpha0 = Param('alpha0', float(alpha0))
        self.link_parameter(self.alpha0, index=index)
        self.logalpha0 = np.log(alpha0)
        self.update_model(True)

    def set_alpha1(self, alpha1):
        """Set alpha1"""
        self.update_model(False)
        index = self.alpha1._parent_index_
        self.unlink_parameter(self.alpha1)
        self.alpha1 = Param('alpha1', float(alpha1))
        self.link_parameter(self.alpha1, index=index)
        self.logalpha0 = np.log(alpha1)
        self.update_model(True)

    def __str__(self):
        return "Schechter({}, {}, {})".format(self.ellStar.values[0],
                                              self.alpha0.values[0],
                                              self.alpha1.values[0])

    def pdf(self, ell, t):
        """prob"""
        alpha = self.alpha0 + self.alpha1 * t
        return (ell / self.ellStar)**alpha * np.exp(ell / self.ellStar)\
            / self.ellStar / gamma(1+alpha)

    def lnpdf(self, ell, t):
        """minus Lnprob"""
        alpha = self.alpha0 + self.alpha1 * t
        return gammaln(1+alpha) + (alpha+1)\
            * self.lnEllStar - alpha * np.log(ell) - ell/self.ellStar

    def lnpdf_grad_ell(self, ell, t):
        """Derivative of lnprob with respect to ell"""
        return - 1/self.ellStar - (self.alpha0 + self.alpha1 * t) / ell

    def lnpdf_grad_t(self, ell, t):
        """Derivative of lnprob with respect to t"""
        return - self.alpha1*(np.log(ell) - self.lnEllStar -
                              polygamma(0, 1 + self.alpha0 + self.alpha1 * t))

    def lnpdf_grad_alpha0(self, ell, t):
        """Derivative of lnprob with respect to alpha0"""
        return - np.log(ell) + self.lnEllStar\
            + polygamma(0, 1 + self.alpha0 + self.alpha1 * t)

    def lnpdf_grad_alpha1(self, ell, t):
        """Derivative of lnprob with respect to alpha1"""
        return - t * (np.log(ell) - self.lnEllStar -
                      polygamma(0, 1 + self.alpha0 + self.alpha1 * t))

    def lnpdf_grad_ellStar(self, ell, t):
        """Derivative of lnprob with respect to alpha1"""
        return 1/self.ellStar *\
            (1 + self.alpha0 + self.alpha1 * t + ell/self.ellStar)

    def update_gradients(self, dL,  ell, t):
        """Update gradient structures"""
        self.ellStar.gradient = np.sum(dL * self.lnpdf_grad_ellStar(ell, t))
        self.alpha0.gradient = np.sum(dL * self.lnpdf_grad_alpha0(ell, t))
        self.alpha1.gradient = np.sum(dL * self.lnpdf_grad_alpha1(ell, t))

    def grad_ell(self, ell, t):
        """Derivative of prob with respect to ell"""
        return - self.pdf(ell, t) * self.lnpdf_grad_ell(ell, t)

    def grad_t(self, ell, t):
        """Derivative of prob with respect to t"""
        return - self.pdf(ell, t) * self.lnpdf_grad_t(ell, t)


class Kumaraswamy(Parameterized):
    """
    Kumaraswamy distribution
        p(t) = alpha0 * alpha1 * t^(alpha0-1) * (1-t^alpha0)^(alpha1-1)
    """
    domain = _REAL

    def __init__(self, alpha0, alpha1, name='Kumaraswamy'):
        super(Kumaraswamy, self).__init__(name=name)
        self.alpha0 = Param('alpha0', float(alpha0))
        self.alpha1 = Param('alpha1', float(alpha1))
        self.link_parameter(self.alpha0)
        self.link_parameter(self.alpha1)
        self.logalpha0 = np.log(alpha0)
        self.logalpha1 = np.log(alpha1)

    def set_alpha0(self, alpha0):
        """Set alpha0"""
        self.update_model(False)
        index = self.alpha0._parent_index_
        self.unlink_parameter(self.alpha0)
        self.alpha0 = Param('alpha0', float(alpha0))
        self.link_parameter(self.alpha0, index=index)
        self.logalpha0 = np.log(alpha0)
        self.update_model(True)

    def set_alpha1(self, alpha1):
        """Set alpha1"""
        self.update_model(False)
        index = self.alpha1._parent_index_
        self.unlink_parameter(self.alpha1)
        self.alpha1 = Param('alpha1', float(alpha1))
        self.link_parameter(self.alpha1, index=index)
        self.logalpha0 = np.log(alpha1)
        self.update_model(True)

    def __str__(self):
        print self.alpha0.values, self.alpha1.values
        return "Kumaraswamy({}, {})".format(self.alpha0.values[0],
                                            self.alpha1.values[0])

    def pdf(self, t):
        """Prob"""
        return self.alpha0 * self.alpha1 * t**(self.alpha0 - 1) *\
            (1 - t**self.alpha0)**(self.alpha1 - 1)

    def lnpdf(self, t):
        """Minus Lnprob"""
        return - self.logalpha0 - self.logalpha1\
            - (self.alpha0 - 1) * np.log(t) - (self.alpha1 - 1) *\
            np.log(1 - t**self.alpha0)

    def lnpdf_grad_t(self, t):
        """Derivative of lnprob with respect to t"""
        return - (self.alpha0 - 1) / t\
            + (self.alpha0 * (self.alpha1 - 1) *
               t**(self.alpha0 - 1)) / (1 - t**self.alpha0)

    def lnpdf_grad_alpha0(self, t):
        """Derivative of lnprob with respect to alpha0"""
        return - 1/self.alpha0 - np.log(t) + (self.alpha1 - 1) *\
            t**self.alpha0 * np.log(t) / (1 - t**self.alpha0)

    def lnpdf_grad_alpha1(self, t):
        """Derivative of lnprob with respect to alpha1"""
        return - 1/self.alpha1 - np.log(1 - t**self.alpha0)

    def update_gradients(self, dL, t):
        """Update gradient structures"""
        self.alpha0.gradient = np.sum(dL * self.lnpdf_grad_alpha0(t))
        self.alpha1.gradient = np.sum(dL * self.lnpdf_grad_alpha1(t))

    def grad_t(self, t):
        """Derivative of prob with respect to t"""
        return - self.pdf(t) * self.lnpdf_grad_t(t)


class Rayleigh(Parameterized):
    """
    Rayleigh distribution
        p(z|t) = z * exp(-0.5 * z^2 / alpha(t)^2) / alpha(t)^2
        with alpha(t) = alpha0 + alpha1 * t
    """
    domain = _REAL

    def __init__(self, alpha0, alpha1, name='Rayleigh'):
        super(Rayleigh, self).__init__(name)
        self.alpha0 = Param('alpha0', float(alpha0))
        self.alpha1 = Param('alpha1', float(alpha1))
        self.link_parameter(self.alpha0)
        self.link_parameter(self.alpha1)

    def set_alpha0(self, alpha0):
        """Set alpha0"""
        self.update_model(False)
        index = self.alpha0._parent_index_
        self.unlink_parameter(self.alpha0)
        self.alpha0 = Param('alpha0', float(alpha0))
        self.link_parameter(self.alpha0, index=index)
        self.logalpha0 = np.log(alpha0)
        self.update_model(True)

    def set_alpha1(self, alpha1):
        """Set alpha1"""
        self.update_model(False)
        index = self.alpha1._parent_index_
        self.unlink_parameter(self.alpha1)
        self.alpha1 = Param('alpha1', float(alpha1))
        self.link_parameter(self.alpha1, index=index)
        self.logalpha0 = np.log(alpha1)
        self.update_model(True)

    def __str__(self):
        return "Rayleigh({}, {})".format(self.alpha0.values[0],
                                         self.alpha1.values[0])

    def pdf(self, z, t):
        """Lnprob"""
        alpha2 = (self.alpha0 + self.alpha1 * t)**2.0
        return z * np.exp(-0.5 * z**2 / alpha2) / alpha2

    def lnpdf(self, z, t):
        """Minus Lnprob"""
        alpha2 = (self.alpha0 + self.alpha1 * t)**2.0
        return np.log(alpha2) - np.log(z) + 0.5 * z**2 / alpha2

    def lnpdf_grad_z(self, z, t):
        """Derivative of lnprob with respect to z"""
        alpha2 = (self.alpha0 + self.alpha1 * t)**2
        return - 1.0 / z + z / alpha2

    def lnpdf_grad_t(self, z, t):
        """Derivative of lnprob with respect to t"""
        alpha = (self.alpha0 + self.alpha1 * t)
        return 2 * self.alpha1 / alpha - self.alpha1 * z**2 / alpha**3

    def lnpdf_grad_alpha0(self, z, t):
        """Derivative of lnprob with respect to alpha0"""
        alpha = (self.alpha0 + self.alpha1 * t)
        return 2 / alpha - z**2 / alpha**3

    def lnpdf_grad_alpha1(self, z, t):
        """Derivative of lnprob with respect to alpha1"""
        alpha = (self.alpha0 + self.alpha1 * t)
        return 2 * t / alpha - t * z**2 / alpha**3

    def update_gradients(self, dL, z, t):
        """Update gradient structures"""
        self.alpha0.gradient = np.sum(dL * self.lnpdf_grad_alpha0(z, t))
        self.alpha1.gradient = np.sum(dL * self.lnpdf_grad_alpha1(z, t))

    def grad_z(self, z, t):
        """Derivative of prob with respect to z"""
        return - self.pdf(z, t) * self.lnpdf_grad_z(z, t)

    def grad_t(self, z, t):
        """Derivative of prob with respect to t"""
        return - self.pdf(z, t) * self.lnpdf_grad_t(z, t)
