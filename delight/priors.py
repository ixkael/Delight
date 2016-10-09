
import numpy as np
from scipy.special import gamma, gammaln, polygamma


class Rayleigh:
    """
    Rayleigh distribution
        p(z|t) = z * exp(-0.5 * z^2 / alpha(t)^2) / alpha(t)^2
    """
    def __init__(self, alpha):
        self.alpha = float(alpha)

    def pdf(self, z):
        """Lnprob"""
        alpha2 = self.alpha**2.0
        return z * np.exp(-0.5 * z**2 / alpha2) / alpha2

    def lnpdf(self, z):
        """Minus Lnprob"""
        alpha2 = self.alpha**2.0
        return np.log(alpha2) - np.log(z) + 0.5 * z**2 / alpha2
