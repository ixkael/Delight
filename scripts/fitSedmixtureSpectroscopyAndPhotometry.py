
import numpy as np
from mpi4py import MPI
from time import time
import sys
sys.path.append("/Users/bl/Dropbox/repos/Delight/")
from delight.utils_cy import *

comm = MPI.COMM_WORLD
threadNum = comm.Get_rank()
numThreads = comm.Get_size()

# --------------------------------------

numsamples = int(sys.argv[1])

prefix = "./sv3dhst_"

useSpec = True
usePhoto = not useSpec
importanceSampling = not useSpec

numTypes = 10
nz = 1000
z_grid_bounds = np.logspace(-2, 0.4, nz+1)
muell_range = [0.1, 100]
mulnz_range = [-1, 0.2]
varell_range = [0.1, 100]
varlnz_range = [0.001, 0.1]

# --------------------------------------

if useSpec:
    specdata = np.load(prefix+'specdata.npz')

    assert specdata["redshifts"].size == specdata["nobj"]
    assert specdata["sedfeatures"].shape[0] == specdata["features_zgrid"].size
    assert specdata["sedfeatures"].shape[1] == specdata["numFeatures"]
    assert specdata["sedfeatures"].shape[2] == specdata["numBands"]
    assert specdata["fluxes"].shape[1] == specdata["numBands"]
    assert specdata["fluxes"].shape[0] == specdata["nobj"]
    assert specdata["fluxes"].shape[1] == specdata["numBands"]
    assert specdata["fluxVars"].shape[0] == specdata["nobj"]
    assert specdata["fluxVars"].shape[1] == specdata["numBands"]

    f_mod_features_spec = np.zeros((specdata["nobj"], specdata["numFeatures"], specdata["numBands"]))
    for it in range(specdata["numFeatures"]):
        for ib in range(specdata["numBands"]):
            f_mod_features_spec[:, it, ib] = np.interp(specdata["redshifts"],
                specdata["features_zgrid"], specdata["sedfeatures"][:, it, ib])

if usePhoto:
    photdata = np.load(prefix+'photdata.npz')

    if useSpec:
        assert photdata["numFeatures"] == specdata["numFeatures"]

    assert photdata["sedfeatures"].shape[0] == photdata["features_zgrid"].size
    assert photdata["sedfeatures"].shape[1] == photdata["numFeatures"]
    assert photdata["sedfeatures"].shape[2] == photdata["numBands"]
    assert photdata["fluxes"].shape[0] == photdata["nobj"]
    assert photdata["fluxes"].shape[1] == photdata["numBands"]
    assert photdata["fluxVars"].shape[0] == photdata["nobj"]
    assert photdata["fluxVars"].shape[1] == photdata["numBands"]

    z_grid_centers = (z_grid_bounds[1:] + z_grid_bounds[:-1]) / 2.0
    z_grid_sizes = (z_grid_bounds[1:] - z_grid_bounds[:-1])

    f_mod_features_phot = np.zeros((nz, photdata["numFeatures"], photdata["numBands"]))
    for it in range(photdata["numFeatures"]):
        for ib in range(photdata["numBands"]):
            f_mod_features_phot[:, it, ib] = np.interp(z_grid_centers,
                photdata["features_zgrid"], photdata["sedfeatures"][:, it, ib])


def hypercube2simplex(zs):
    fac = np.concatenate((1 - zs, np.array([1])))
    zsb = np.concatenate((np.array([1]), zs))
    fs = np.cumprod(zsb) * fac
    return fs

if useSpec:
    numFeatures = specdata["numFeatures"]
if usePhoto:
    numFeatures = photdata["numFeatures"]

def logposterior(x):
    if not importanceSampling:
        if np.any(x < param_ranges_min) or np.any(x > param_ranges_max):
            ind = x < param_ranges_min
            ind |= x > param_ranges_max
            print("parameters outside of allowed range: ", np.where(ind)[0])
            return -np.inf
    zs = x[0:numTypes-1]
    fs = hypercube2simplex(zs)
    order = np.argsort(fs)
    alpha_zs = x[numTypes-1:numTypes-1+numTypes*(numFeatures-1)]
    alpha_fs = np.vstack([hypercube2simplex(alpha_zs[i*(numFeatures-1):(i+1)*(numFeatures-1)])
                          for i in order])  # numTypes * numFeatures
    betas = x[numTypes-1+numTypes*(numFeatures-1):]
    mu_ell, mu_lnz, var_ell, var_lnz, corr = np.split(betas, 5)
    mu_ell, mu_lnz, var_ell, var_lnz, corr =\
        mu_ell[order], mu_lnz[order], var_ell[order], var_lnz[order], corr[order]
    rho = corr * np.sqrt(var_ell * var_lnz)

    logpost = 0

    if useSpec:
        f_mod_spec = np.dot(alpha_fs, f_mod_features_spec)
        speclnevidences = np.zeros((specdata["nobj"], ))
        specobj_evidences_margell(
            speclnevidences, fs,
            specdata["nobj"], numTypes, specdata["numBands"],
            specdata["fluxes"], specdata["fluxVars"], f_mod_spec,
            specdata["redshifts"],
            mu_ell, mu_lnz, var_ell, var_lnz, rho)
        logpost += np.sum(speclnevidences)

    if usePhoto:
        photolnevidences = np.zeros((photdata["nobj"], ))
        f_mod_phot = np.dot(alpha_fs, f_mod_features_phot)
        photoobj_evidences_marglnzell(
            photolnevidences, fs,
            photdata["nobj"], numTypes, nz, photdata["numBands"],
            photdata["fluxes"], photdata["fluxVars"], f_mod_phot,
            z_grid_centers, z_grid_sizes,
            mu_ell, mu_lnz, var_ell, var_lnz, rho)
        ind = np.where(np.isfinite(photolnevidences))[0]
        logpost += np.sum(photolnevidences[ind])

    return logpost

fname = prefix+"samples_thread"+str(threadNum+1)+"on"+str(numThreads)

if importanceSampling:

    samples = np.genfromtxt(fname+".txt")
    t1 = time()
    for i in range(numsamples):
        samples[i, 0] += logposterior(samples[i, 1:])
    t2 = time()
    print('Thread', threadNum+1, "on", numThreads, ': Finished sampling! Took', (t2-t1)/numsamples, 'sec per sample')
    np.savetxt(fname+"_importancesampled.txt", samples[0:numsamples, :])
    print("Wrote to file", fname+"_importancesampled.txt")

else:

    param_ranges = \
        [[0, 1]] * (numTypes-1) +\
        [[0, 1]] * (numTypes*(numFeatures-1)) +\
        [muell_range] * numTypes +\
        [mulnz_range] * numTypes +\
        [varell_range] * numTypes +\
        [varlnz_range] * numTypes +\
        [[-.9, .9]] * numTypes
    ndim = len(param_ranges)
    param_ranges_min = np.array([rr[0] for rr in param_ranges])
    param_ranges_max = np.array([rr[1] for rr in param_ranges])

    t1 = time()
    samples = np.zeros((numsamples, ndim+1))
    for i in range(numsamples):
        samples[i, 1:] = param_ranges_min + (param_ranges_max - param_ranges_min) * np.random.uniform(0.1, 0.9, size=ndim)
        samples[i, 0] = logposterior(samples[i, 1:])
    t2 = time()
    print('Thread', threadNum+1, "on", numThreads, ': Finished sampling! Took', (t2-t1)/numsamples, 'sec per sample')

    t1 = time()
    order = np.argsort(samples[:, 0])[::-1]
    samples = samples[order, :]
    t2 = time()
    print('Thread', threadNum+1, "on", numThreads, ': Finished sorting samples. Took', (t2-t1), 'sec')

    np.savetxt(fname+".txt", samples)
    print("Wrote to file", fname+".txt")
