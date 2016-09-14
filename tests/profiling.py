# -*- coding: utf-8 -*-
"""Test routines from photoz_kernels.py"""

import time
import numpy as np

from delight.photoz_kernels_cy import kernelparts

from delight.photoz_gp import PhotozGP
from delight.utils import *

from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel


NREPEAT = 2
nObj = 10
nObjGP = 4
nInducing = 5
numBands = 5
size = numBands * nObj
redshiftGrid = np.linspace(0, 3, num=30)
use_interpolators = True

extranoise = 1e-8
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
    X = random_X_bzl(size, numBands=numBands)
    X2 = random_X_bzl(size, numBands=numBands)
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

print('Relative error on KC:', relerr1.mean(), relerr1.std())
print('Relative error on KL:', relerr2.mean(), relerr2.std())
print('Relative error on D_alpha_C:', relerr3.mean(), relerr3.std())
print('Relative error on D_alpha_L:', relerr4.mean(), relerr4.std())
print("=> kernelparts (raw): %s s" % (t_raw / NREPEAT))
print("=> kernelparts (constr): %s s" % (t_constr / NREPEAT))
print("=> kernelparts (interp): %s s" % (t_interp / NREPEAT))


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
            print('elapsed time: %f ms' % self.msecs)


X = random_X_bzl(size, numBands=numBands)
if nInducing > 0:
    X_inducing = random_X_bzl(nInducing, numBands=numBands)
else:
    X_inducing = None
fcoefs_amp, fcoefs_mu, fcoefs_sig \
    = random_filtercoefs(numBands, numCoefs)
lines_mu, lines_sig = random_linecoefs(numLines)
var_C, var_L, alpha_C, alpha_L, alpha_T = random_hyperparams()
kern = Photoz_kernel(fcoefs_amp, fcoefs_mu, fcoefs_sig,
                     lines_mu, lines_sig, var_C, var_L,
                     alpha_C, alpha_L, alpha_T)
kern.construct_interpolators()
dL_dm = np.ones((size, 1))
dL_dK = 1
norms = np.sqrt(2*np.pi) * np.sum(fcoefs_amp * fcoefs_sig, axis=1)


print('--------')
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
print("=> kernelparts (raw): %s s" % (t.secs / NREPEAT))

print('--------')

with Timer() as t:
    for i in range(NREPEAT):
        X = random_X_bzl(size, numBands=numBands)
print("=> Random X: %s s" % (t.secs / NREPEAT))
tX = (t.secs / NREPEAT)

with Timer() as t:
    for i in range(NREPEAT):
        X = random_X_bzl(size, numBands=numBands)
        v = kern.K(X)
print("=> K (X varying): %s s" % (t.secs / NREPEAT))

with Timer() as t:
    for i in range(NREPEAT):
        v = kern.K(X)
print("=> K (X fixed): %s s" % (t.secs / NREPEAT))

print('-------')

mf = Photoz_mean_function(0.0, fcoefs_amp, fcoefs_mu, fcoefs_sig)

with Timer() as t:
    for i in range(NREPEAT):
        X = random_X_bzl(size, numBands=numBands)
        v = mf.f(X)
print("=> f (X varying): %s s" % (t.secs / NREPEAT))

with Timer() as t:
    for i in range(NREPEAT):
        v = mf.f(X)
print("=> f (X fixed): %s s" % (t.secs / NREPEAT))

print('--------')
