
import sys
from mpi4py import MPI
import numpy as np
from delight.io import *
from delight.utils import *
from delight.photoz_gp import PhotozGP
from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel

comm = MPI.COMM_WORLD
threadNum = comm.Get_rank()
numThreads = comm.Get_size()

# Parse parameters file
if len(sys.argv) < 2:
    raise Exception('Please provide a parameter file')
params = parseParamFile(sys.argv[1], verbose=False)
if threadNum == 0:
    print("--- DELIGHT-APPLY ---")

# Read filter coefficients, compute normalization of filters
bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, norms\
    = readBandCoefficients(params)
numBands = bandCoefAmplitudes.shape[0]

redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)
f_mod = readSEDs(params)
numZbins = redshiftDistGrid.size - 1
numZ = redshiftGrid.size

numObjectsTraining = np.sum(1 for line in open(params['training_catFile']))
numObjectsTarget = np.sum(1 for line in open(params['target_catFile']))
redshiftsInTarget = ('redshift' in params['target_bandOrder'])
Ncompress = params['Ncompress']

firstLine = int(threadNum * numObjectsTarget / float(numThreads))
lastLine = int(min(numObjectsTarget,
               (threadNum + 1) * numObjectsTarget / float(numThreads)))
numLines = lastLine - firstLine
if threadNum == 0:
    print('Number of Training Objects', numObjectsTraining)
    print('Number of Target Objects', numObjectsTarget)
comm.Barrier()
print('Thread ', threadNum, ' analyzes lines ', firstLine, ' to ', lastLine)

gp = PhotozGP(f_mod, bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
              params['lines_pos'], params['lines_width'],
              params['V_C'], params['V_L'],
              params['alpha_C'], params['alpha_L'],
              redshiftGridGP, use_interpolators=True)

# Create local files to store results
numMetrics = 7 + len(params['confidenceLevels'])
localPDFs = np.zeros((numLines, numZ))
localMetrics = np.zeros((numLines, numMetrics))
localCompressIndices = np.zeros((numLines,  Ncompress), dtype=int)
localCompEvidences = np.zeros((numLines,  Ncompress))

# Looping over chunks of the training set to prepare model predictions over z
numChunks = params['training_numChunks']
for chunk in range(numChunks):
    TR_firstLine = int(chunk * numObjectsTraining / float(numChunks))
    TR_lastLine = int(min(numObjectsTraining,
                      (chunk + 1) * numObjectsTarget / float(numChunks)))
    targetIndices = np.arange(TR_firstLine, TR_lastLine)
    numTObjCk = TR_lastLine - TR_firstLine
    redshifts = np.zeros((numTObjCk, ))
    model_mean = np.zeros((numZ, numTObjCk, numBands))
    model_covar = np.zeros((numZ, numTObjCk, numBands))
    loc = TR_firstLine - 1
    trainingDataIter = getDataFromFile(params, TR_firstLine, TR_lastLine,
                                       prefix="training_", ftype="gpparams")
    for loc, (z, ell, bands, X, B, flatarray) in enumerate(trainingDataIter):
        redshifts[loc] = z
        gp.setCore(X, B, f_mod.shape[0],
                   flatarray[0:f_mod.shape[0]+B+B*(B+1)//2])
        model_mean[:, loc, :], model_covar[:, loc, :] =\
            gp.predictAndInterpolate(redshiftGrid, ell=ell)
        model_mean[:, loc, :] /= ell
        model_covar[:, loc, :] /= ell**2

    if params['compressionFilesFound']:
        fC = open(params['compressMargLikFile'])
        fCI = open(params['compressIndicesFile'])
        itCompM = itertools.islice(fC, firstLine, lastLine)
        iterCompI = itertools.islice(fCI, firstLine, lastLine)
    targetDataIter = getDataFromFile(params, firstLine, lastLine,
                                     prefix="target_", getXY=False, CV=False)
    for loc, (z, ell, bands, fluxes, fluxesVar, bCV, dCV, dVCV)\
            in enumerate(targetDataIter):

        if params['compressionFilesFound']:
            indices = np.array(next(iterCompI).split(' '), dtype=int)
            sel = np.in1d(targetIndices, indices, assume_unique=True)
            # like_grid = scalefree_flux_likelihood(
            #    fluxes / ell, fluxesVar / ell**2,
            #    model_mean[:, :, bands][:, sel, :],
            #    f_mod_var=model_var[:, :, bands][:, sel, :]
            # )
            # like_grid = flux_likelihood(
            #    fluxes, fluxesVar,
            #    ell * model_mean[:, :, bands][:, sel, :],
            #    ell**2 * model_covar[:, :, bands][:, sel, :]
            # )
            like_grid = approx_flux_likelihood(
                fluxes,
                fluxesVar,
                ell * model_mean[:, :, bands][:, sel, :],
                ell**2 * model_covar[:, :, bands][:, sel, :],
                1,
                params['ellFracStd']**2.
            )
        else:
            # like_grid = scalefree_flux_likelihood(
            #    fluxes / ell, fluxesVar / ell**2,
            #    model_mean[:, :, bands],  # model mean
            #    f_mod_var=model_var[:, :, bands]  # model var
            # )
            # like_grid = flux_likelihood(
            #    fluxes, fluxesVar,
            #    ell * model_mean[:, :, bands],
            #    ell**2 * model_covar[:, :, bands]
            # )
            like_grid = approx_flux_likelihood(
                fluxes,
                fluxesVar,
                ell * model_mean[:, :, bands],
                ell**2 * model_covar[:, :, bands],
                1,
                params['ellFracStd']**2.
            )
        localPDFs[loc, :] += like_grid.sum(axis=1)
        evidences = np.trapz(like_grid, x=redshiftGrid, axis=0)

        if not params['compressionFilesFound']:
            if localCompressIndices[loc, :].sum() == 0:
                sortind = np.argsort(evidences)[::-1][0:Ncompress]
                localCompressIndices[loc, :] = targetIndices[sortind]
                localCompEvidences[loc, :] = evidences[sortind]
            else:
                dind = np.concatenate((targetIndices,
                                       localCompressIndices[loc, :]))
                devi = np.concatenate((evidences,
                                       localCompEvidences[loc, :]))
                sortind = np.argsort(devi)[::-1][0:Ncompress]
                localCompressIndices[loc, :] = dind[sortind]
                localCompEvidences[loc, :] = devi[sortind]

        if chunk == numChunks - 1\
                and redshiftsInTarget\
                and localPDFs[loc, :].sum() > 0:
            localMetrics[loc, :] = computeMetrics(
                                    z, redshiftGrid,
                                    localPDFs[loc, :],
                                    params['confidenceLevels'])



    if params['compressionFilesFound']:
        fC.close()
        fCI.close()

comm.Barrier()
if threadNum == 0:
    globalPDFs = np.zeros((numObjectsTarget, numZ))
    globalCompressIndices = np.zeros((numObjectsTarget, Ncompress), dtype=int)
    globalCompEvidences = np.zeros((numObjectsTarget, Ncompress))
    globalMetrics = np.zeros((numObjectsTarget, numMetrics))
else:
    globalPDFs = None
    globalCompressIndices = None
    globalCompEvidences = None
    globalMetrics = None

firstLines = [int(k*numObjectsTarget/numThreads)
              for k in range(numThreads)]
lastLines = [int(min(numObjectsTarget, (k+1)*numObjectsTarget/numThreads))
             for k in range(numThreads)]
numLines = [lastLines[k] - firstLines[k] for k in range(numThreads)]

sendcounts = tuple([numLines[k] * numZ for k in range(numThreads)])
displacements = tuple([firstLines[k] * numZ for k in range(numThreads)])
comm.Gatherv(localPDFs,
             [globalPDFs, sendcounts, displacements, MPI.DOUBLE])

sendcounts = tuple([numLines[k] * Ncompress for k in range(numThreads)])
displacements = tuple([firstLines[k] * Ncompress for k in range(numThreads)])
comm.Gatherv(localCompressIndices,
             [globalCompressIndices, sendcounts, displacements, MPI.LONG])
comm.Gatherv(localCompEvidences,
             [globalCompEvidences, sendcounts, displacements, MPI.DOUBLE])
comm.Barrier()

sendcounts = tuple([numLines[k] * numMetrics for k in range(numThreads)])
displacements = tuple([firstLines[k] * numMetrics for k in range(numThreads)])
comm.Gatherv(localMetrics,
             [globalMetrics, sendcounts, displacements, MPI.DOUBLE])

comm.Barrier()

if threadNum == 0:
    fmt = '%.2e'
    fname = params['redshiftpdfFileComp'] if params['compressionFilesFound']\
        else params['redshiftpdfFile']
    np.savetxt(fname, globalPDFs, fmt=fmt)
    if redshiftsInTarget:
        np.savetxt(params['metricsFile'], globalMetrics, fmt=fmt)
    if not params['compressionFilesFound']:
        np.savetxt(params['compressMargLikFile'],
                   globalCompEvidences, fmt=fmt)
        np.savetxt(params['compressIndicesFile'],
                   globalCompressIndices, fmt="%i")
