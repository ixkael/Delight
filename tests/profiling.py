# -*- coding: utf-8 -*-
"""Test routines from photoz_kernels.py"""

import time
import numpy as np
from scipy.misc import derivative
from delight.photoz_kernels_cy import kernelparts, kernelparts_diag

import GPy

from delight.photoz_gp import PhotozGP
from delight.utils import random_X_bzlt,\
    random_filtercoefs, random_linecoefs, random_hyperparams
from delight.hmc import HMC

from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel


NREPEAT = 4
nObj = 100
nInducing = 5
numBands = 4
size = numBands * nObj
redshiftGrid = np.linspace(0, 3, num=60)
use_interpolators = True

numLines = 3
numCoefs = 10
bandsUsed = range(numBands)
np.set_printoptions(suppress=True, precision=3)


relerr1 = np.zeros((size, size))
relerr2 = np.zeros((size, size))
relerr3 = np.zeros((size, size))
relerr4 = np.zeros((size, size))
t_constr = 0
t_interp = 0
t_raw = 0
for i in range(NREPEAT):
    X = random_X_bzlt(size, numBands=numBands)
    X2 = random_X_bzlt(size, numBands=numBands)
    fcoefs_amp, fcoefs_mu, fcoefs_sig \
        = random_filtercoefs(numBands, numCoefs)
    lines_mu, lines_sig = random_linecoefs(numLines)
    var_C, var_L, alpha_C, alpha_L, alpha_T = random_hyperparams()
    norms = np.sqrt(2*np.pi) * np.sum(fcoefs_amp * fcoefs_sig, axis=1)
    kern = Photoz_kernel(fcoefs_amp, fcoefs_mu, fcoefs_sig,
                         lines_mu, lines_sig, var_C, var_L,
                         alpha_C, alpha_L, alpha_T)
    b1 = X[:, 0].astype(int)
    b2 = X2[:, 0].astype(int)
    fz1 = (1. + X[:, 1])
    fz2 = (1. + X2[:, 1])

    t1 = time.time()
    kern.construct_interpolators()
    t2 = time.time()
    t_constr += (t2 - t1)

    t1 = time.time()
    kern.update_kernelparts(X, X2)
    t2 = time.time()
    t_interp += (t2 - t1)

    t1 = time.time()
    assert X.shape[0] == size
    ts = (size, size)
    KC, KL = np.zeros(ts), np.zeros(ts)
    D_alpha_C, D_alpha_L, D_alpha_z\
        = np.zeros(ts), np.zeros(ts), np.zeros(ts)
    kernelparts(size, size, numCoefs, numLines,
                alpha_C, alpha_L,
                fcoefs_amp, fcoefs_mu, fcoefs_sig,
                lines_mu, lines_sig,
                norms, b1, fz1, b2, fz2,
                True, KL, KC, D_alpha_C, D_alpha_L, D_alpha_z)
    t2 = time.time()
    t_raw += (t2 - t1)

    relerr1 += np.abs(kern.KC/KC - 1.) / NREPEAT
    relerr2 += np.abs(kern.KL/KL - 1.) / NREPEAT
    relerr3 += np.abs(kern.D_alpha_C/D_alpha_C - 1.) / NREPEAT
    relerr4 += np.abs(kern.D_alpha_L/D_alpha_L - 1.) / NREPEAT

print 'Relative error on KC:', relerr1.mean(), relerr1.std()
print 'Relative error on KL:', relerr2.mean(), relerr2.std()
print 'Relative error on D_alpha_C:', relerr3.mean(), relerr3.std()
print 'Relative error on D_alpha_L:', relerr4.mean(), relerr4.std()
print "=> kernelparts (raw): %s s" % (t_raw / NREPEAT)
print "=> kernelparts (constr): %s s" % (t_constr / NREPEAT)
print "=> kernelparts (interp): %s s" % (t_interp / NREPEAT)


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print 'elapsed time: %f ms' % self.msecs


X = random_X_bzlt(size, numBands=numBands)
X_inducing = random_X_bzlt(nInducing, numBands=numBands)
fcoefs_amp, fcoefs_mu, fcoefs_sig \
    = random_filtercoefs(numBands, numCoefs)
lines_mu, lines_sig = random_linecoefs(numLines)
var_C, var_L, alpha_C, alpha_L, alpha_T = random_hyperparams()
alpha = np.random.uniform(low=1e-4, high=1e-3, size=1)
beta = np.random.uniform(low=0.3, high=0.7, size=1)
kern = Photoz_kernel(fcoefs_amp, fcoefs_mu, fcoefs_sig,
                     lines_mu, lines_sig, var_C, var_L,
                     alpha_C, alpha_L, alpha_T)
kern.construct_interpolators()
dL_dm = np.ones((size, 1))
dL_dK = 1
norms = np.sqrt(2*np.pi) * np.sum(fcoefs_amp * fcoefs_sig, axis=1)


print '--------'
b1 = X[:, 0].astype(int)
b2 = X[:, 0].astype(int)
fz1 = 1 + X[:, 1]
fz2 = 1 + X[:, 1]
with Timer() as t:
    for i in range(NREPEAT):
        KC, KL = np.zeros((size, size)), np.zeros((size, size))
        D_alpha_C, D_alpha_L, D_alpha_z \
            = np.zeros((size, size)), np.zeros((size, size)),\
            np.zeros((size, size))
        kernelparts(size, size, numCoefs, numLines,
                    alpha_C, alpha_L,
                    fcoefs_amp, fcoefs_mu, fcoefs_sig,
                    lines_mu, lines_sig, norms,
                    b1, fz1, b2, fz2, True,
                    KL, KC,
                    D_alpha_C, D_alpha_L, D_alpha_z)
print "=> kernelparts (raw): %s s" % (t.secs / NREPEAT)

print '--------'

with Timer() as t:
    for i in range(NREPEAT):
        X = random_X_bzlt(size, numBands=numBands)
print "=> Random X: %s s" % (t.secs / NREPEAT)
tX = (t.secs / NREPEAT)

with Timer() as t:
    for i in range(NREPEAT):
        X = random_X_bzlt(size, numBands=numBands)
        v = kern.K(X)
print "=> K (X varying): %s s" % (t.secs / NREPEAT)

with Timer() as t:
    for i in range(NREPEAT):
        v = kern.K(X)
print "=> K (X fixed): %s s" % (t.secs / NREPEAT)

with Timer() as t:
    for i in range(NREPEAT):
        X = random_X_bzlt(size, numBands=numBands)
        v = kern.update_gradients_diag(dL_dK, X)
print "=> update_gradients_diag (X varying): %s s" % (t.secs / NREPEAT)

with Timer() as t:
    for i in range(NREPEAT):
        v = kern.update_gradients_diag(dL_dK, X)
print "=> update_gradients_diag (X fixed): %s s" % (t.secs / NREPEAT)

with Timer() as t:
    for i in range(NREPEAT):
        X = random_X_bzlt(size, numBands=numBands)
        v = kern.update_gradients_full(dL_dK, X)
print "=> update_gradients_full (X varying): %s s" % (t.secs / NREPEAT)

with Timer() as t:
    for i in range(NREPEAT):
        v = kern.update_gradients_full(dL_dK, X)
print "=> update_gradients_full (X fixed): %s s" % (t.secs / NREPEAT)

with Timer() as t:
    for i in range(NREPEAT):
        X = random_X_bzlt(size, numBands=numBands)
        v = kern.gradients_X(dL_dK, X)
print "=> gradients_X (X varying): %s s" % (t.secs / NREPEAT)

with Timer() as t:
    for i in range(NREPEAT):
        v = kern.gradients_X(dL_dK, X)
print "=> gradients_X (X fixed): %s s" % (t.secs / NREPEAT)

print '-------'

mf = Photoz_mean_function(alpha, beta, fcoefs_amp, fcoefs_mu, fcoefs_sig)

with Timer() as t:
    for i in range(NREPEAT):
        X = random_X_bzlt(size, numBands=numBands)
        v = mf.f(X)
print "=> f (X varying): %s s" % (t.secs / NREPEAT)

with Timer() as t:
    for i in range(NREPEAT):
        v = mf.f(X)
print "=> f (X fixed): %s s" % (t.secs / NREPEAT)

with Timer() as t:
    for i in range(NREPEAT):
        X = random_X_bzlt(size, numBands=numBands)
        v = mf.update_gradients(dL_dm, X)
print "=> update_gradients (X varying): %s s" % (t.secs / NREPEAT)

with Timer() as t:
    for i in range(NREPEAT):
        v = mf.update_gradients(dL_dm, X)
print "=> update_gradients (X fixed): %s s" % (t.secs / NREPEAT)

with Timer() as t:
    for i in range(NREPEAT):
        X = random_X_bzlt(size, numBands=numBands)
        v = mf.gradients_X(dL_dm, X)
print "=> gradients_X (X varying): %s s" % (t.secs / NREPEAT)

with Timer() as t:
    for i in range(NREPEAT):
        v = mf.gradients_X(dL_dm, X)
print "=> gradients_X (X fixed): %s s" % (t.secs / NREPEAT)

print '--------'

X = random_X_bzlt(nObj)
used_indices = np.arange(nObj)
bands, redshifts, luminosities, types = np.split(X, 4, axis=1)

noisy_fluxes = np.random.uniform(low=0., high=1., size=size)\
    .reshape((nObj, numBands))
flux_variances = np.random.uniform(low=0., high=1., size=size)\
    .reshape((nObj, numBands))

with Timer() as t:
    gp = PhotozGP(
        redshifts, luminosities, types, used_indices,
        noisy_fluxes, flux_variances, bandsUsed,
        fcoefs_amp, fcoefs_mu, fcoefs_sig,
        lines_mu, lines_sig,
        alpha, beta, var_C, var_L,
        alpha_C, alpha_L, alpha_T,
        X_inducing=X_inducing,
        redshiftGrid=redshiftGrid,
        use_interpolators=use_interpolators
        )
print "Created GP in %s s" % (t.secs / NREPEAT)

with Timer() as t:
    for i in range(NREPEAT):
        gp.parameters_changed()
print "=> parameters_changed (all fixed): %s s" % (t.secs / NREPEAT)

with Timer() as t:
    for i in range(NREPEAT):
        gp.set_redshifts(np.random.uniform(0, 3, redshifts.shape))
print "=> parameters_changed (z varying): %s s" % (t.secs / NREPEAT)

with Timer() as t:
    for i in range(NREPEAT):
        gp.set_luminosities(np.random.uniform(0, 3, luminosities.shape))
print "=> parameters_changed (l varying): %s s" % (t.secs / NREPEAT)

with Timer() as t:
    for i in range(NREPEAT):
        gp.set_types(np.random.uniform(0, 1, types.shape))
print "=> parameters_changed (t varying): %s s" % (t.secs / NREPEAT)

scalar_params, array_params = gp.get_param_names_and_indices()
with Timer() as t:
    for i in range(NREPEAT):
        gp.set_unfixed_parameters(gp.unfixed_param_array,
                                  scalar_params, array_params)
print "=> set_unfixed_parameters (X fixed): %s s" % (t.secs / NREPEAT)

print '--------'

hmc = GPy.inference.mcmc.HMC(gp, stepsize=1e-4)
with Timer() as t:
    hmc_samples = hmc.sample(num_samples=NREPEAT, hmc_iters=1)
print "=> HMC iterations (all varying): %s s" % (t.secs / NREPEAT)

gp.kern.var_L.fix()
gp.kern.alpha_C.fix()
gp.kern.alpha_L.fix()
gp.mean_function.alpha.fix()
gp.mean_function.beta.fix()

hmc = GPy.inference.mcmc.HMC(gp, stepsize=1e-4)
with Timer() as t:
    hmc_samples = hmc.sample(num_samples=NREPEAT, hmc_iters=1)
print "=> HMC iterations (CL not varying): %s s" % (t.secs / NREPEAT)

gp.unfixed_redshifts.fix()
gp.unfixed_types.constrain_bounded(0, 1)
gp.unfixed_luminosities.constrain_bounded(0, 10)

hmc = HMC(gp, stepsize=1e-2)
with Timer() as t:
    hmc_samples, derived_params = hmc.sample(num_samples=NREPEAT, hmc_iters=1)
print "=> HMC iterations (CL and z not varying): %s s" % (t.secs / NREPEAT)

gp.unfixed_redshifts.fix()
gp.unfixed_types.fix()
gp.unfixed_luminosities.fix()

hmc = HMC(gp, stepsize=1e-2)
with Timer() as t:
    hmc_samples, derived_params = hmc.sample(num_samples=NREPEAT, hmc_iters=1)
print "=> HMC iterations (CL and ztl not varying): %s s" % (t.secs / NREPEAT)

print '--------'
