# -*- coding: utf-8 -*-

import numpy as np


def hmc_sampler(x0, lnprob, lnprobgrad, step_size,
                num_steps, inv_mass_matrix_diag=None, bounds=None, **kwargs):
    if bounds is None:
        bounds = np.zeros((x0.size, 2))
        bounds[:, 0] = 0.001
        bounds[:, 1] = 0.999
    if inv_mass_matrix_diag is None:
        inv_mass_matrix_diag = np.repeat(1, x0.size)
        inv_mass_matrix_diag_sqrt = np.repeat(1, x0.size)
    else:
        assert inv_mass_matrix_diag.size == x0.size
        inv_mass_matrix_diag_sqrt = inv_mass_matrix_diag**0.5
    v0 = np.random.randn(x0.size) / inv_mass_matrix_diag_sqrt
    v = v0 - 0.5 * step_size * lnprobgrad(x0, **kwargs)
    x = x0 + step_size * v * inv_mass_matrix_diag
    ind_upper = x > bounds[:, 1]
    x[ind_upper] = 2*bounds[ind_upper, 1] - x[ind_upper]
    v[ind_upper] = - v[ind_upper]
    ind_lower = x < bounds[:, 0]
    x[ind_lower] = 2*bounds[ind_lower, 0] - x[ind_lower]
    v[ind_lower] = - v[ind_lower]
    ind_upper = x > bounds[:, 1]
    ind_lower = x < bounds[:, 0]
    ind_bad = np.logical_or(ind_lower, ind_upper)
    if ind_bad.sum() > 0:
        print('Error: could not confine samples without bounds!')
        print('Number of problematic parameters:',
              ind_bad.sum(), 'out of', ind_bad.size)
        return x0

    for i in range(num_steps):
        v = v - step_size * lnprobgrad(x, **kwargs)
        x = x + step_size * v * inv_mass_matrix_diag
        ind_upper = x > bounds[:, 1]
        x[ind_upper] = 2*bounds[ind_upper, 1] - x[ind_upper]
        v[ind_upper] = - v[ind_upper]
        ind_lower = x < bounds[:, 0]
        x[ind_lower] = 2*bounds[ind_lower, 0] - x[ind_lower]
        v[ind_lower] = - v[ind_lower]
        ind_upper = x > bounds[:, 1]
        ind_lower = x < bounds[:, 0]
        ind_bad = np.logical_or(ind_lower, ind_upper)
        if ind_bad.sum() > 0:
            print('Error: could not confine samples without bounds!')
            print('Number of problematic parameters:',
                  ind_bad.sum(), 'out of', ind_bad.size)
            return x0

    v = v - 0.5 * step_size * lnprobgrad(x, **kwargs)
    orig = lnprob(x0, **kwargs)
    current = lnprob(x, **kwargs)
    if inv_mass_matrix_diag is None:
        orig += 0.5 * np.dot(v0.T, v0)
        current += 0.5 * np.dot(v.T, v)
    else:
        orig += 0.5 * np.sum(inv_mass_matrix_diag * v0**2.)
        current += 0.5 * np.sum(inv_mass_matrix_diag * v**2.)

    p_accept = min(1.0, np.exp(orig - current))
    if(np.any(~np.isfinite(x))):
        print('Error: some parameters are infinite!',
              np.sum(~np.isfinite(x)), 'out of', x.size)
        print('HMC steps and stepsize:', num_steps, step_size)
        return x0
    if p_accept > np.random.uniform():
        return x
    else:
        if p_accept < 0.01:
            print('Error: acceptance rate is very small! It is', p_accept)
            print('HMC steps and stepsize:', num_steps, step_size)
        return x0
