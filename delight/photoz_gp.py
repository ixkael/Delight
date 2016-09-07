
import numpy as np
from copy import copy
import scipy.linalg
from scipy.optimize import minimize

from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel

log_2_pi = np.log(2*np.pi)


class PhotozGP:
    """
    Photo-z Gaussian process, with   physical kernel and mean function.
    Default: all parameters are variable except bands and likelihood/noise.
    """
    def __init__(self, mean_fct, kernel,
                 X=None, Y=None, Yvar=None, L=None, beta=None):

        self.mean_fct = mean_fct
        self.kernel = kernel
        if X is None:
            raise Exception("Problem! X should be specified")
        self.X = X

        if L is None and beta is None:
            if Y is None or Yvar is None:
                raise Exception("Problem! Y and Yvar should be specified")
            self.Y = Y.reshape((-1, 1))
            self.Yvar = Yvar.reshape((-1, 1))
            self.compute()

        if Y is None and Yvar is None:
            if L is None or beta is None:
                raise Exception("Problem! L and beta should be specified")
            self.L = L
            self.beta = beta.reshape((-1, 1))

    def compute(self):
        self.D = 1*self.Y
        if self.mean_fct is not None:
            self.D -= self.mean_fct.f(self.X)
        self.KXX = self.kernel.K(self.X)
        self.A = self.KXX + np.diag(self.Yvar.flatten())
        self.L = scipy.linalg.cholesky(self.A, lower=True)
        self.beta = scipy.linalg.cho_solve((self.L, True), self.D)
        logdet = np.log(scipy.linalg.det(self.KXX))
        self.marglike =\
            0.5 * np.sum(self.beta * self.D) +\
            0.5 * logdet + 0.5 * self.Y.size * log_2_pi

    def predict(self, x_pred):
        assert x_pred.shape[1] == 3
        KXXp = self.kernel.K(x_pred, self.X)
        KXpXp = self.kernel.K(x_pred)
        y_pred = np.dot(KXXp, self.beta)
        if self.mean_fct is not None:
            y_pred += self.mean_fct.f(x_pred)
        v = scipy.linalg.cho_solve((self.L, True), KXXp.T)
        y_pred_fullcov = KXpXp - KXXp.dot(v)
        return y_pred, y_pred_fullcov

    def updateAlphaAndReturnMarglike(self, alpha):
        self.mean_fct.alpha = alpha[0]
        #self.X[:, 2] = alpha[1]
        self.compute()
        return self.marglike

    def optimize_alpha(self):
        x0 = 0.0# [0.0, self.X[0, 2]]
        res = minimize(self.updateAlphaAndReturnMarglike, x0,
                       method='L-BFGS-B', tol=1e-6, bounds=[(-3e-4, 3e-4)])
        self.mean_fct.alpha = res.x[0]
        #self.X[:, 2] = res.x[1]
