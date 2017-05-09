#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
cimport numpy as np
from cython.parallel import prange
from cpython cimport bool
cimport cython
from libc.math cimport sqrt, M_PI, exp, pow


def find_positions(
        int NO1, int nz,
        double[:] fz1,
        long[:] p1s,
        double[:] fzGrid
        ):

    cdef long p1, o1
    for o1 in prange(NO1, nogil=True):
        for p1 in range(nz-1):
            if fz1[o1] >= fzGrid[p1] and fz1[o1] <= fzGrid[p1+1]:
                p1s[o1] = p1
                break;


def bilininterp_precomputedbins(
            int numBands, int nobj,
            double[:, :] Kinterp, # nbands x nobj
            double[:] v1s, # nobj (val in grid1)
            double[:] v2s, # nobj (val in grid1)
            long[:] p1s, # nobj (pos in grid1)
            long[:] p2s, # nobj (pos in grid2)
            double[:] grid1,
            double[:] grid2,
            double[:, :, :] Kgrid): # nbands x ngrid1 x ngrid2

    cdef int p1, p2, o, b
    cdef double dzm2, v1, v2
    for o in prange(nobj, nogil=True):
        p1 = p1s[o]
        p2 = p2s[o]
        v1 = v1s[o]
        v2 = v2s[o]
        dzm2 = 1. / (grid1[p1+1] - grid1[p1]) / (grid2[p2+1] - grid2[p2])
        for b in range(numBands):
            Kinterp[b, o] = dzm2 * (
                (grid1[p1+1] - v1) * (grid2[p2+1] - v2) * Kgrid[b, p1, p2]
                + (v1 - grid1[p1]) * (grid2[p2+1] - v2) * Kgrid[b, p1+1, p2]
                + (grid1[p1+1] - v1) * (v2 - grid2[p2]) * Kgrid[b, p1, p2+1]
                + (v1 - grid1[p1]) * (v2 - grid2[p2]) * Kgrid[b, p1+1, p2+1]
                )


def kernel_parts_interp(
            int NO1, int NO2,
            double[:,:] Kinterp,
            long[:] b1,
            double[:] fz1,
            long[:] p1s,
            long [:] b2,
            double[:] fz2,
            long[:] p2s,
            double[:] fzGrid,
            double[:,:,:,:] Kgrid):

    cdef int p1, p2, o1, o2
    cdef double dzm2, opz1, opz2
    for o1 in prange(NO1, nogil=True):
        opz1 = fz1[o1]
        p1 = p1s[o1]
        for o2 in range(NO2):
            opz2 = fz2[o2]
            p2 = p2s[o2]
            dzm2 = 1. / (fzGrid[p1+1] - fzGrid[p1]) / (fzGrid[p2+1] - fzGrid[p2])
            Kinterp[o1, o2] = dzm2 * (
                (fzGrid[p1+1] - opz1) * (fzGrid[p2+1] - opz2) * Kgrid[b1[o1], b2[o2], p1, p2]
                + (opz1 - fzGrid[p1]) * (fzGrid[p2+1] - opz2) * Kgrid[b1[o1], b2[o2], p1+1, p2]
                + (fzGrid[p1+1] - opz1) * (opz2 - fzGrid[p2]) * Kgrid[b1[o1], b2[o2], p1, p2+1]
                + (opz1 - fzGrid[p1]) * (opz2 - fzGrid[p2]) * Kgrid[b1[o1], b2[o2], p1+1, p2+1]
                )



def approx_flux_likelihood_cy(
        double [:, :] like,  # nz, nt
        long nz,
        long nt,
        long nf,
        double[:] f_obs,  # nf
        double[:] f_obs_var,  # nf
        double[:,:,:] f_mod,  # nz, nt, nf
        double[:,:,:] f_mod_covar, # nz, nt, nf
        double[:] ell_hat, # 1
        double[:] ell_var # 1
    ):

    cdef long i, i_t, i_z, i_f, niter=2
    cdef double var, FOT, FTT, FOO, chi2, ellML, denom
    for i_t in prange(nt, nogil=True):
        for i_z in range(nz):
            ellML = 0
            for i in range(niter):
                FOT = ell_hat[i_z] / ell_var[i_z]
                FTT = 1. / ell_var[i_z]
                FOO = ell_hat[i_z]**2 / ell_var[i_z]
                denom = 1
                for i_f in range(nf):
                    var = (f_obs_var[i_f] + ellML**2 * f_mod_covar[i_z, i_t, i_f])
                    FOT = FOT + f_mod[i_z, i_t, i_f] * f_obs[i_f] / var
                    FTT = FTT + pow(f_mod[i_z, i_t, i_f], 2) / var
                    FOO = FOO + pow(f_obs[i_f], 2) / var
                    if i == 1:
                        denom = denom * sqrt(var*2*M_PI)
                ellML = FOT / FTT
                if i == 1:
                    chi2 = FOO - pow(FOT, 2) / FTT
                    denom = denom * sqrt(2*M_PI*ell_var[i_z])
                    denom = denom * sqrt(FTT / (2*M_PI))
                    like[i_z, i_t] = exp(-0.5*chi2) / denom  # nz * nt
