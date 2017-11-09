#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
cimport numpy as np
from cython.parallel import prange
from cpython cimport bool
cimport cython
from libc.math cimport sqrt, M_PI, exp, pow, log
from libc.stdlib cimport malloc, free


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
    cdef double var, FOT, FTT, FOO, chi2, ellML, logDenom, loglikemax
    for i_z in prange(nz, nogil=True):
        for i_t in range(nt):
            ellML = 0
            for i in range(niter):
                FOT = ell_hat[i_z] / ell_var[i_z]
                FTT = 1. / ell_var[i_z]
                FOO = ell_hat[i_z]**2 / ell_var[i_z]
                logDenom = 0
                for i_f in range(nf):
                    var = (f_obs_var[i_f] + ellML**2 * f_mod_covar[i_z, i_t, i_f])
                    FOT = FOT + f_mod[i_z, i_t, i_f] * f_obs[i_f] / var
                    FTT = FTT + pow(f_mod[i_z, i_t, i_f], 2) / var
                    FOO = FOO + pow(f_obs[i_f], 2) / var
                    if i == niter - 1:
                        logDenom = logDenom + log(var*2*M_PI)
                ellML = FOT / FTT
                if i == niter - 1:
                    chi2 = FOO - pow(FOT, 2) / FTT
                    logDenom = logDenom + log(2*M_PI*ell_var[i_z])
                    logDenom = logDenom + log(FTT / (2*M_PI))
                    like[i_z, i_t] = -0.5*chi2 - 0.5*logDenom  # nz * nt

    if True:
        loglikemax = like[0, 0]
        for i_z in range(nz):
            for i_t in range(nt):
                if like[i_z, i_t] > loglikemax:
                    loglikemax = like[i_z, i_t]
        for i_z in range(nz):
            for i_t in range(nt):
                like[i_z, i_t] = exp(like[i_z, i_t] - loglikemax)


cdef double gauss_prob(double x, double mu, double var) nogil:
    return exp(- 0.5 * pow(x - mu, 2.)/var) / sqrt(2.*M_PI*var)


cdef double gauss_lnprob(double x, double mu, double var) nogil:
    return - 0.5 * pow(x - mu, 2)/var - 0.5 * log(2*M_PI*var)


cdef double logsumexp(double* arr, long dim) nogil:
    cdef int i
    cdef double result = 0.0
    cdef double largest_in_a = arr[0]
    for i in range(1, dim):
        if (arr[i] > largest_in_a):
            largest_in_a = arr[i]
    for i in range(dim):
        result += exp(arr[i] - largest_in_a)
    return largest_in_a + log(result)


def photoobj_evidences_marglnzell(
    double [:] logevidences, # nobj
    double [:] alphas, # nt
    long nobj, long numTypes, long nz, long nf,
    double [:, :] f_obs,  # nobj * nf
    double [:, :] f_obs_var,  # nobj * nf
    double [:, :, :] f_mod,  # nt * nz * nf
    double [:] z_grid_centers, # nz
    double [:] z_grid_sizes, # nz
    double [:] mu_ell, # nt
    double [:] mu_lnz, double [:] var_ell,  # nt
    double [:] var_lnz, double [:] rho  # nt
        ):

    cdef long o, i_t, i_z, i_f
    cdef double FOT, FTT, FOO, chi2, ellML, logDenom
    cdef double mu_ell_prime, var_ell_prime, lnprior_lnz
    cdef double *logpost

    for o in range(nobj):#prange(nobj, nogil=True):

        logpost = <double *> malloc(sizeof(double) * (nz*numTypes))
        for i_z in range(nz):
            for i_t in range(numTypes):
                mu_ell_prime = mu_ell[i_t] + rho[i_t] * (log(z_grid_centers[i_z]) - mu_lnz[i_t]) / var_lnz[i_t]
                var_ell_prime = (var_ell[i_t] - pow(rho[i_t], 2) / var_lnz[i_t])
                FOT = mu_ell_prime / var_ell_prime
                FTT = 1. / var_ell_prime
                FOO = pow(mu_ell_prime, 2) / var_ell_prime
                logDenom = 0
                for i_f in range(nf):
                    FOT = FOT + f_mod[i_t, i_z, i_f] * f_obs[o, i_f] / f_obs_var[o, i_f]
                    FTT = FTT + pow(f_mod[i_t, i_z, i_f], 2) / f_obs_var[o, i_f]
                    FOO = FOO + pow(f_obs[o, i_f], 2) / f_obs_var[o, i_f]
                    logDenom = logDenom + log(f_obs_var[o, i_f]*2*M_PI)
                # ellML = FOT / FTT
                chi2 = FOO - pow(FOT, 2) / FTT
                logDenom = logDenom + log(var_ell_prime) + log(FTT)
                lnprior_lnz = gauss_lnprob(log(z_grid_centers[i_z]), mu_lnz[i_t], var_lnz[i_t])
                logpost[i_t*nz+i_z] = log(alphas[i_t]) - 0.5*chi2 - 0.5*logDenom  + lnprior_lnz + log(z_grid_sizes[i_z])

        for i_t in range(numTypes):
            logevidences[o] = logsumexp(logpost, nz*numTypes)

        free(logpost)


def specobj_evidences_margell(
    double [:] logevidences, # nobj
    double [:] alphas, # nt
    long nobj, long numTypes, long nf,
    double [:, :] f_obs,  # nobj * nf
    double [:, :] f_obs_var,  # nobj * nf
    double [:, :, :] f_mod,  # nt * nobj * nf
    double [:] redshifts, # nobj
    double [:] mu_ell, # nt
    double [:] mu_lnz, double [:] var_ell,  # nt
    double [:] var_lnz, double [:] rho  # nt
        ):

    cdef long o, i_t, i_f
    cdef double FOT, FTT, FOO, chi2, ellML, logDenom
    cdef double mu_ell_prime, var_ell_prime, lnprior_lnz
    cdef double *logpost

    for o in range(nobj):#prange(nobj, nogil=True):

        logpost = <double *> malloc(sizeof(double) * (numTypes))
        for i_t in range(numTypes):
            mu_ell_prime = mu_ell[i_t] + rho[i_t] * (log(redshifts[o]) - mu_lnz[i_t]) / var_lnz[i_t]
            var_ell_prime = (var_ell[i_t] - pow(rho[i_t], 2) / var_lnz[i_t])
            FOT = mu_ell_prime / var_ell_prime
            FTT = 1. / var_ell_prime
            FOO = pow(mu_ell_prime, 2) / var_ell_prime
            logDenom = 0
            for i_f in range(nf):
                FOT = FOT + f_mod[i_t, o, i_f] * f_obs[o, i_f] / f_obs_var[o, i_f]
                FTT = FTT + pow(f_mod[i_t, o, i_f], 2) / f_obs_var[o, i_f]
                FOO = FOO + pow(f_obs[o, i_f], 2) / f_obs_var[o, i_f]
                logDenom = logDenom + log(f_obs_var[o, i_f]*2*M_PI)
            # ellML = FOT / FTT
            chi2 = FOO - pow(FOT, 2) / FTT
            logDenom = logDenom + log(var_ell_prime) + log(FTT)
            lnprior_lnz = gauss_lnprob(log(redshifts[o]), mu_lnz[i_t], var_lnz[i_t])
            logpost[i_t] = log(alphas[i_t]) - 0.5*chi2 - 0.5*logDenom + lnprior_lnz

        for i_t in range(numTypes):
            logevidences[o] = logsumexp(logpost, numTypes)

        free(logpost)


def photoobj_lnpost_zgrid_margell(
    double [:, :, :] lnpost, # nobj * nt * nz
    double [:] alphas, # nt
    long nobj, long numTypes, long nz, long nf,
    double [:, :] f_obs,  # nobj * nf
    double [:, :] f_obs_var,  # nobj * nf
    double [:, :, :] f_mod,  # nt * nz * nf
    double [:] z_grid_centers, # nz
    double [:] z_grid_sizes, # nz
    double [:] mu_ell, # nt
    double [:] mu_lnz, double [:] var_ell,  # nt
    double [:] var_lnz, double [:] rho  # nt
        ):

    cdef long o, i_t, i_z, i_f
    cdef double FOT, FTT, FOO, chi2, ellML, logDenom
    cdef double mu_ell_prime, var_ell_prime, lnprior_lnz

    for o in prange(nobj, nogil=True):

        for i_z in range(nz):
            for i_t in range(numTypes):
                mu_ell_prime = mu_ell[i_t] + rho[i_t] * (log(z_grid_centers[i_z]) - mu_lnz[i_t]) / var_lnz[i_t]
                var_ell_prime = (var_ell[i_t] - pow(rho[i_t], 2) / var_lnz[i_t])
                FOT = mu_ell_prime / var_ell_prime
                FTT = 1. / var_ell_prime
                FOO = pow(mu_ell_prime, 2) / var_ell_prime
                logDenom = 0
                for i_f in range(nf):
                    FOT = FOT + f_mod[i_t, i_z, i_f] * f_obs[o, i_f] / f_obs_var[o, i_f]
                    FTT = FTT + pow(f_mod[i_t, i_z, i_f], 2) / f_obs_var[o, i_f]
                    FOO = FOO + pow(f_obs[o, i_f], 2) / f_obs_var[o, i_f]
                    logDenom = logDenom + log(f_obs_var[o, i_f]*2*M_PI)
                # ellML = FOT / FTT
                chi2 = FOO - pow(FOT, 2) / FTT
                logDenom = logDenom + log(var_ell_prime) + log(FTT)
                lnprior_lnz = gauss_lnprob(log(z_grid_centers[i_z]), mu_lnz[i_t], var_lnz[i_t])
                lnpost[o, i_t, i_z] = log(alphas[i_t]) - 0.5*chi2 - 0.5*logDenom + lnprior_lnz
