
import numpy as np
from copy import deepcopy as copy


class HMC:
    def __init__(self, model, M=None, stepsize=1e-1):
        self.model = model
        self.stepsize = stepsize
        self.p = np.empty_like(model.optimizer_array.copy())
        self.derived_param_names = model.derived_param_names
        if M is None:
            self.M = np.eye(self.p.size)
        else:
            self.M = M
        self.Minv = np.linalg.inv(self.M)

    def sample(self, num_samples=1000, hmc_iters=20):
        params = []
        derived_params = []
        for i in range(num_samples):
            self.p[:] = np.random.multivariate_normal(
                np.zeros(self.p.size), self.M)
            H_old = self._computeH()
            theta_old = self.model.optimizer_array.copy()
            self._update(hmc_iters)
            H_new = self._computeH()
            if H_old > H_new:
                k = 1.
            else:
                k = np.exp(H_old-H_new)
            if np.random.rand() < k:
                params.append(copy(self.model.unfixed_param_array))
                derived_params.append(copy(self.model.derived_params))
            else:
                self.model.optimizer_array = theta_old
        return params, derived_params

    def _update(self, hmc_iters):
        for i in range(hmc_iters):
            self.p[:] += -self.stepsize/2. *\
                self.model._transform_gradients(
                    self.model.objective_function_gradients())
            self.model.optimizer_array = self.model.optimizer_array\
                + self.stepsize*np.dot(self.Minv, self.p)
            self.p[:] += -self.stepsize/2. *\
                self.model._transform_gradients(
                    self.model.objective_function_gradients())

    def _computeH(self,):
        return self.model.objective_function()\
            + self.p.size * np.log(2*np.pi) / 2.\
            + np.log(np.linalg.det(self.M)) / 2.\
            + np.dot(self.p, np.dot(self.Minv, self.p[:, None])) / 2.
