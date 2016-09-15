
import sys
from mpi4py import MPI
import numpy as np
from scipy.interpolate import interp1d
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
f_mod = np.zeros((redshiftGrid.size, len(sed_names), len(bandNames)))
for t, sed_name in enumerate(sed_names):
    seddata = np.genfromtxt(dir_seds + '/' + sed_name + '.sed')
    seddata[:, 1] *= seddata[:, 0]**2. / 3e18
    ref = np.interp(lambdaRef, seddata[:, 0], seddata[:, 1])
    seddata[:, 1] /= ref
    sed_interp = interp1d(seddata[:, 0], seddata[:, 1])
    for jf, band in enumerate(bandNames):
        fname_in = dir_filters + '/' + band + '.res'
        data = np.genfromtxt(fname_in)
        xf, yf = data[:, 0], data[:, 1]
        yf /= xf  # divide by lambda
        ind = np.where(yf > 0.01*np.max(yf))[0]
        lambdaMin, lambdaMax = xf[ind[0]], xf[ind[-1]]
        norm = np.trapz(yf, x=xf)
        for iz in range(redshiftGrid.size):
            opz = (redshiftGrid[iz] + 1)
            xf_z = np.linspace(lambdaMin / opz, lambdaMax / opz, num=5000)
            yf_z = interp1d(xf / opz, yf)(xf_z)
            ysed = sed_interp(xf_z)
            f_mod[iz, t, jf] = np.trapz(ysed * yf_z, x=xf_z) / norm
            f_mod[iz, t, jf] *= opz**2. / DL(redshiftGrid[iz])**2. / (4*np.pi)


numObjectsTarget = np.sum(1 for line in open(params['target_catFile']))
firstLine = int(threadNum * numObjectsTarget / float(numThreads))
lastLine = int(min(numObjectsTarget,
               (threadNum + 1) * numObjectsTarget / float(numThreads)))
numLines = lastLine - firstLine
if threadNum == 0:
    print('Number of Target Objects', numObjectsTarget)
comm.Barrier()
print('Thread ', threadNum, ' analyzes lines ', firstLine, ' to ', lastLine)

numMetrics = 5 + len(params['confidenceLevels'])
# Create local files to store results
localPDFs = np.zeros((numLines, numZ))
localMetrics = np.zeros((numLines, numMetrics))

# Now loop over target set to compute likelihood function
loc = - 1
trainingDataIter = getDataFromFile(params, firstLine, lastLine,
                                    prefix="target_", getXY=False)
for z, ell, bands, fluxes, fluxesVar in trainingDataIter:
    loc += 1
    like_grid = scalefree_flux_likelihood(
        fluxes, fluxesVar,
        f_mod[:, :, bands]
    )
    localPDFs[loc, :] += like_grid.sum(axis=1)
    localMetrics[loc, :] = computeMetrics(
                                z, redshiftGrid,
                                localPDFs[loc, :],
                                params['confidenceLevels'])

comm.Barrier()
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
comm.Gatherv(localPDFs,
             [globalPDFs, sendcounts, displacements, MPI.DOUBLE])

sendcounts = tuple([numLines[k] * numMetrics for k in range(numThreads)])
displacements = tuple([firstLines[k] * numMetrics for k in range(numThreads)])
comm.Gatherv(localMetrics,
             [globalMetrics, sendcounts, displacements, MPI.DOUBLE])

comm.Barrier()

if threadNum == 0:
    fmt = '%.2e'
    np.savetxt(params['redshiftpdfFileTemp'], globalPDFs, fmt=fmt)
    if redshiftColumn >= 0:
        np.savetxt(params['metricsFile'], globalMetrics, fmt=fmt)
