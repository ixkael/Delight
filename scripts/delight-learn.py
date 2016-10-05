
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
    print("--- DELIGHT-LEARN ---")

# Read filter coefficients, compute normalization of filters
bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, norms\
    = readBandCoefficients(params)
numBands = bandCoefAmplitudes.shape[0]

redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)
f_mod = readSEDs(params)

numObjectsTraining = np.sum(1 for line in open(params['training_catFile']))
print('Number of Training Objects', numObjectsTraining)
firstLine = int(threadNum * numObjectsTraining / numThreads)
lastLine = int(min(numObjectsTraining,
               (threadNum + 1) * numObjectsTraining / numThreads))
numLines = lastLine - firstLine
comm.Barrier()
print('Thread ', threadNum, ' analyzes lines ', firstLine, ' to ', lastLine)

gp = PhotozGP(f_mod, bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
              params['lines_pos'], params['lines_width'],
              params['V_C'], params['V_L'],
              params['alpha_C'], params['alpha_L'],
              redshiftGridGP, use_interpolators=True)

B = numBands
numCol = 3 + B + B*(B+1)//2 + B + f_mod.shape[0]
localData = np.zeros((numLines, numCol))
fmt = '%i ' + '%.6e ' * (localData.shape[1] - 1)

loc = - 1
crossValidate = params['training_crossValidate']
trainingDataIter1 = getDataFromFile(params, firstLine, lastLine,
                                    prefix="training_", getXY=True,
                                    CV=crossValidate)
if crossValidate:
    chi2sLocal = None
    bandIndicesCV, bandNamesCV, bandColumnsCV,\
        bandVarColumnsCV, redshiftColumnCV =\
        readColumnPositions(params, prefix="training_CV_", refFlux=False)

for z, ell,\
    bands, fluxes, fluxesVar,\
    bandsCV, fluxesCV, fluxesVarCV,\
        X, Y, Yvar in trainingDataIter1:
    loc += 1
    gp.setData(X, Y, Yvar)
    lB = bands.size
    alpha_hat, ell_hat = 0, ell # gp.estimateAlphaEll()
    localData[loc, 0] = lB
    localData[loc, 1] = z
    localData[loc, 2] = ell_hat
    localData[loc, 3:3+bands.size] = bands
    localData[loc, 3+lB:3+f_mod.shape[0]+lB+lB*(lB+1)//2+lB] = gp.getCore()

    if crossValidate:
        model_mean, model_covar\
            = gp.predictAndInterpolate(np.array([z]), ell=ell_hat)
        model_mean /= ell_hat
        model_covar /= ell_hat**2
        if chi2sLocal is None:
            chi2sLocal = np.zeros((numObjectsTraining, bandIndicesCV.size))
        ind = np.array([list(bandIndicesCV).index(b) for b in bandsCV])
        chi2sLocal[firstLine + loc, ind] =\
            - 0.5 * (model_mean[0, bandsCV] - fluxesCV / ell_hat)**2 /\
            (model_var[0, bandsCV] + fluxesVarCV / ell_hat**2)

# use MPI to get the totals
comm.Barrier()
if threadNum == 0:
    reducedData = np.zeros((numObjectsTraining, numCol))
else:
    reducedData = None

if crossValidate:
    chi2sGlobal = np.zeros_like(chi2sLocal)
    comm.Allreduce(chi2sLocal, chi2sGlobal, op=MPI.SUM)
    comm.Barrier()

firstLines = [int(k*numObjectsTraining/numThreads)
              for k in range(numThreads)]
lastLines = [int(min(numObjectsTraining, (k+1)*numObjectsTraining/numThreads))
             for k in range(numThreads)]
sendcounts = tuple([(lastLines[k] - firstLines[k]) * numCol
                    for k in range(numThreads)])
displacements = tuple([firstLines[k] * numCol
                       for k in range(numThreads)])

comm.Gatherv(localData, [reducedData, sendcounts, displacements, MPI.DOUBLE])
comm.Barrier()

if threadNum == 0:
    np.savetxt(params['training_paramFile'], reducedData, fmt=fmt)
    if crossValidate:
        np.savetxt(params['training_CVfile'], chi2sGlobal)
