
import sys
#from mpi4py import MPI
import numpy as np
from scipy.interpolate import interp1d

from delight.io import *
from delight.utils import *
from delight.photoz_gp import PhotozGP
from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel

#comm = MPI.COMM_WORLD
threadNum = 0
numThreads = 1
if threadNum == 0:
    print("--- TEMPLATE FITTING ---")

# Parse parameters file
if len(sys.argv) < 2:
    raise Exception('Please provide a parameter file')
paramFileName = sys.argv[1]
params = parseParamFile(paramFileName, verbose=False)
if threadNum == 0:
    print('Thread number / number of threads: ', threadNum+1, numThreads)
    print('Input parameter file:', paramFileName)

DL = approx_DL()
redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)
numZ = redshiftGrid.size

# Locate which columns of the catalog correspond to which bands.
bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn,\
    refBandColumn = readColumnPositions(params, prefix="target_")

dir_seds = params['templates_directory']
dir_filters = params['bands_directory']
lambdaRef = params['lambdaRef']
sed_names = params['templates_names']
f_mod = np.zeros((redshiftGrid.size, len(sed_names),
                  len(params['bandNames'])))
for t, sed_name in enumerate(sed_names):
    f_mod[:, t, :] = np.loadtxt(dir_seds + '/' + sed_name +
                                '_fluxredshiftmod.txt')

numObjectsTarget = np.sum(1 for line in open(params['target_catFile']))
firstLine = int(threadNum * numObjectsTarget / float(numThreads))
lastLine = int(min(numObjectsTarget,
               (threadNum + 1) * numObjectsTarget / float(numThreads)))
numLines = lastLine - firstLine
if threadNum == 0:
    print('Number of Target Objects', numObjectsTarget)
#comm.Barrier()
print('Thread ', threadNum, ' analyzes lines ', firstLine, ' to ', lastLine)

numMetrics = 7 + len(params['confidenceLevels'])
# Create local files to store results
localPDFs = np.zeros((numLines, numZ))
localMetrics = np.zeros((numLines, numMetrics))

# Now loop over target set to compute likelihood function
loc = - 1
trainingDataIter = getDataFromFile(params, firstLine, lastLine,
                                   prefix="target_", getXY=False)
for z, normedRefFlux, bands, fluxes, fluxesVar,\
        bCV, fCV, fvCV in trainingDataIter:
    loc += 1
    # like_grid, _ = scalefree_flux_likelihood(
    #    fluxes, fluxesVar,
    #    f_mod[:, :, bands])
    # ell_hat_z = normedRefFlux * 4 * np.pi\
    #    * params['fluxLuminosityNorm'] \
    #    * (DL(redshiftGrid)**2. * (1+redshiftGrid))[:, None]
    ell_hat_z = 1
    params['ellPriorSigma'] = 1e12
    like_grid = approx_flux_likelihood(
        fluxes, fluxesVar, f_mod[:, :, bands],
        normalized=True, marginalizeEll=True,
        ell_hat=ell_hat_z, ell_var=(ell_hat_z*params['ellPriorSigma'])**2)
    b_in = np.array(params['p_t'])[None, :]
    beta2 = np.array(params['p_z_t'])**2.0
    p_z = b_in * redshiftGrid[:, None] / beta2[None, :] *\
        np.exp(-0.5 * redshiftGrid[:, None]**2 / beta2[None, :])
    like_grid *= p_z
    localPDFs[loc, :] += like_grid.sum(axis=1)
    if localPDFs[loc, :].sum() > 0:
        localMetrics[loc, :] = computeMetrics(
                                    z, redshiftGrid,
                                    localPDFs[loc, :],
                                    params['confidenceLevels'])

#comm.Barrier()
if threadNum == 0:
    globalPDFs = np.zeros((numObjectsTarget, numZ))
    globalMetrics = np.zeros((numObjectsTarget, numMetrics))
else:
    globalPDFs = None
    globalMetrics = None

firstLines = [int(k*numObjectsTarget/numThreads)
              for k in range(numThreads)]
lastLines = [int(min(numObjectsTarget, (k+1)*numObjectsTarget/numThreads))
             for k in range(numThreads)]
numLines = [lastLines[k] - firstLines[k] for k in range(numThreads)]

sendcounts = tuple([numLines[k] * numZ for k in range(numThreads)])
displacements = tuple([firstLines[k] * numZ for k in range(numThreads)])


print('localPDFs.shape = ', localPDFs.shape)
print('globalPDFs.shape = ', globalPDFs.shape)
print('localMetrics.shape = ', localMetrics.shape)
print('globalMetrics.shape = ', globalMetrics.shape)


#comm.Gatherv(localPDFs,[globalPDFs, sendcounts, displacements, MPI.DOUBLE])
globalPDFs = localPDFs

sendcounts = tuple([numLines[k] * numMetrics for k in range(numThreads)])
displacements = tuple([firstLines[k] * numMetrics for k in range(numThreads)])


#comm.Gatherv(localMetrics,[globalMetrics, sendcounts, displacements, MPI.DOUBLE])
globalMetrics = localMetrics

#comm.Barrier()

if threadNum == 0:
    fmt = '%.2e'
    np.savetxt(params['redshiftpdfFileTemp'], globalPDFs, fmt=fmt)
    if redshiftColumn >= 0:
        np.savetxt(params['metricsFileTemp'], globalMetrics, fmt=fmt)
