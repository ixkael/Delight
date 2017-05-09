#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
cimport numpy as np
from cython.parallel import prange
from cpython cimport bool
cimport cython
from libc.math cimport sqrt, M_PI, exp, pow


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


def kernelparts_diag(
        int NO1, int NC, int NL,
        double alpha_C, double alpha_L,
        double[:,:] fcoefs_amp,
        double[:,:] fcoefs_mu,
        double[:,:] fcoefs_sig,
        double[:] lines_mu,
        double[:] lines_sig,
        double[:] norms,
        long[:] b1,
        double[:] fz1,
        bool grad_needed,
        double[:] KL,
        double[:] KC,
        double[:] D_alpha_C,
        double[:] D_alpha_L
    ):

    cdef double sqrt2pi = sqrt(2 * M_PI)
    cdef int l1, l2, o1, i, j
    cdef double theexp, opz1, opz2, mu1, mu2, sig1, sig2, amp1, amp2, sigma, mul1, mul2

    for o1 in prange(NO1, nogil=True):
        KC[o1] = 0
        KL[o1] = 0
        opz1 = fz1[o1]
        opz2 = fz1[o1]
        for i in range(NC):
            mu1 = fcoefs_mu[b1[o1],i]
            amp1 = fcoefs_amp[b1[o1],i]
            sig1 = fcoefs_sig[b1[o1],i]
            for j in range(NC):
                mu2 = fcoefs_mu[b1[o1],j]
                amp2 = fcoefs_amp[b1[o1],j]
                sig2 = fcoefs_sig[b1[o1],j]
                sigma = sqrt( pow(opz1*sig2,2) + pow(opz2*sig1,2) + pow(opz1*opz2*alpha_C,2) )
                theexp = amp1 * amp2 * 2 * M_PI * sig1 * sig2 * exp(-0.5*pow((opz1*mu2 - opz2*mu1)/sigma,2)) / sigma
                KC[o1] += alpha_C * theexp
                if grad_needed is True:
                    D_alpha_C[o1] += theexp * (1 - pow(alpha_C*opz1*opz2/sigma,2) + pow(alpha_C*(opz1*mu2 - opz2*mu1)*opz1*opz2,2) /pow(sigma,4)  )

                if NL > 0:
                    for l1 in range(NL):
                        mul1 = lines_mu[l1]
                        for l2 in range(l1):
                            mul2 = lines_mu[l2]
                            KL[o1] += 2 * amp1 * amp2 * exp(-0.5*(pow((mu1 - opz1*mul1)/sig1,2) + pow((mu2 - opz2*mul2)/sig2,2) + pow((mul1-mul2)/alpha_L,2)))
                            if grad_needed is True:
                                D_alpha_L[o1] += 2 * amp1 * amp2 * exp(-0.5*(pow((mu1 - opz1*mul1)/sig1,2) + pow((mu2 - opz2*mul2)/sig2,2) + pow((mul1-mul2)/alpha_L,2))) * pow(mul1-mul2,2) / pow(alpha_L,3)

                        # Last term needed once
                        l2 = l1
                        mul2 = lines_mu[l2]
                        KL[o1] += amp1 * amp2 * exp(-0.5*(pow((mu1 - opz1*mul1)/sig1,2) + pow((mu2 - opz2*mul2)/sig2,2) + pow((mul1-mul2)/alpha_L,2)))
                        if grad_needed is True:
                            D_alpha_L[o1] += amp1 * amp2 * exp(-0.5*(pow((mu1 - opz1*mul1)/sig1,2) + pow((mu2 - opz2*mul2)/sig2,2) + pow((mul1-mul2)/alpha_L,2))) * pow(mul1-mul2,2) / pow(alpha_L,3)

        KC[o1] /= norms[b1[o1]] * norms[b1[o1]]
        KL[o1] /= norms[b1[o1]] * norms[b1[o1]]

        if grad_needed is True:
            D_alpha_C[o1] /= norms[b1[o1]] * norms[b1[o1]]
            D_alpha_L[o1] /= norms[b1[o1]] * norms[b1[o1]]


def kernelparts(
        int NO1, int NO2, int NC, int NL,
        double alpha_C, double alpha_L,
        double[:,:] fcoefs_amp,
        double[:,:] fcoefs_mu,
        double[:,:] fcoefs_sig,
        double[:] lines_mu,
        double[:] lines_sig,
        double [:] norms,
        long[:] b1,
        double[:] fz1,
        long[:] b2,
        double[:] fz2,
        bool grad_needed,
        double[:,:] KL,
        double[:,:] KC,
        double [:,:] D_alpha_C,
        double [:,:] D_alpha_L,
        double [:,:] D_alpha_z
    ):

    cdef double sqrt2pi = sqrt(2 * M_PI)
    cdef int l1, l2, o1, o2, i, j
    cdef double theexp, opz1, opz2, mu1, mu2, amp1, amp2, sig1, sig2, sigma, mul1, mul2
    #, sigl1, sigl2

    for o1 in prange(NO1, nogil=True):
        for o2 in range(NO2):
            opz1 = fz1[o1]
            opz2 = fz2[o2]
            #KC[o1,o2] = 0
            #KL[o1,o2] = 0
            #if grad_needed is True:
            #    D_alpha_L[o1,o2] = 0
            #    D_alpha_C[o1,o2] = 0
            #    D_alpha_z[o1,o2] = 0
            for i in range(NC):
                mu1 = fcoefs_mu[b1[o1],i]
                amp1 = fcoefs_amp[b1[o1],i]
                sig1 = fcoefs_sig[b1[o1],i]
                for j in range(NC):
                    mu2 = fcoefs_mu[b2[o2],j]
                    amp2 = fcoefs_amp[b2[o2],j]
                    sig2 = fcoefs_sig[b2[o2],j]
                    sigma = sqrt( pow(opz1*sig2,2) + pow(opz2*sig1,2) + pow(opz1*opz2*alpha_C,2) )
                    theexp = amp1 * amp2 * 2 * M_PI * sig1 * sig2 * exp(-0.5*pow((opz1*mu2 - opz2*mu1)/sigma,2)) / sigma
                    KC[o1,o2] += alpha_C * theexp
                    if grad_needed is True:
                        D_alpha_C[o1,o2] += theexp * (1 - pow(alpha_C*opz1*opz2/sigma,2)  + pow(alpha_C*(opz1*mu2 - opz2*mu1)*opz1*opz2,2) /pow(sigma,4)  )
                        D_alpha_z[o1,o2] += alpha_C * theexp * ( (sig2**2 * opz1 + opz1 * opz2**2 * alpha_C**2) * ((mu2*opz1 - mu1*opz2)**2 / pow(sigma,4)  -  1 / sigma**2) \
                              - mu2 * (mu2*opz1 - mu1*opz2) / sigma**2 )

                    if NL > 0:
                        for l1 in range(NL):
                            mul1 = lines_mu[l1]
                            for l2 in range(l1):
                                mul2 = lines_mu[l2]
                                KL[o1,o2] += 2 * amp1 * amp2 * exp(-0.5*(pow((mu1 - opz1*mul1)/sig1,2) + pow((mu2 - opz2*mul2)/sig2,2) + pow((mul1-mul2)/alpha_L,2)))
                                if grad_needed is True:
                                    D_alpha_L[o1,o2] += 2 * amp1 * amp2 * exp(-0.5*(pow((mu1 - opz1*mul1)/sig1,2) + pow((mu2 - opz2*mul2)/sig2,2) + pow((mul1-mul2)/alpha_L,2))) * pow(mul1-mul2,2) / pow(alpha_L,3)

                            # Last term needed once
                            l2 = l1
                            mul2 = lines_mu[l2]
                            KL[o1,o2] += amp1 * amp2 * exp(-0.5*(pow((mu1 - opz1*mul1)/sig1,2) + pow((mu2 - opz2*mul2)/sig2,2) + pow((mul1-mul2)/alpha_L,2)))
                            if grad_needed is True:
                                D_alpha_L[o1,o2] += amp1 * amp2 * exp(-0.5*(pow((mu1 - opz1*mul1)/sig1,2) + pow((mu2 - opz2*mul2)/sig2,2) + pow((mul1-mul2)/alpha_L,2))) * pow(mul1-mul2,2) / pow(alpha_L,3)

            KC[o1,o2] /= norms[b1[o1]] * norms[b2[o2]]
            KL[o1,o2] /= norms[b1[o1]] * norms[b2[o2]]

            if grad_needed is True:
                D_alpha_C[o1,o2] /= norms[b1[o1]] * norms[b2[o2]]
                D_alpha_L[o1,o2] /= norms[b1[o1]] * norms[b2[o2]]
                D_alpha_z[o1,o2] /= norms[b1[o1]] * norms[b2[o2]]
