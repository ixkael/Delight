
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
verbose = (True if (threadNum == 0) else False)
params = parseParamFile(sys.argv[1], verbose=verbose)

# Read filter coefficients, compute normalization of filters
bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, norms\
    = readBandCoefficients(params)
numBands = bandCoefAmplitudes.shape[0]

redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)

numObjectsTraining = np.sum(1 for line in open(params['training_catFile']))
print('Number of Training Objects', numObjectsTraining)
firstLine = int(threadNum * numObjectsTraining / numThreads)
lastLine = int(min(numObjectsTraining,
               (threadNum + 1) * numObjectsTraining / numThreads))
numLines = lastLine - firstLine
comm.Barrier()
print('Thread ', threadNum, ' analyzes lines ', firstLine, ' to ', lastLine)

gp = PhotozGP(0.0, bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
              params['lines_pos'], params['lines_width'],
              params['V_C'], params['V_L'],
              params['alpha_C'], params['alpha_L'],
              redshiftGridGP, use_interpolators=True)

B = numBands
numCol = 4 + B + B*(B+1)//2 + B
localData = np.zeros((numLines, numCol))
fmt = '%i ' + '%.6e ' * (localData.shape[1] - 1)

loc = - 1
trainingDataIter1 = getDataFromFile(params, firstLine, lastLine,
                                    prefix="training_", getXY=True)
for z, ell, bands, fluxes, fluxesVar, X, Y, Yvar in trainingDataIter1:
    loc += 1
    gp.setData(X, Y, Yvar)
    gp.optimizeAlpha()
    localData[loc, 0] = bands.size
    localData[loc, 1] = z
    localData[loc, 2] = ell
    localData[loc, 3:3+bands.size] = bands
    localData[loc, 3+bands.size:] = gp.getCore()

# use MPI to get the totals
comm.Barrier()
if threadNum == 0:
    reducedData = np.zeros((numObjectsTraining, numCol))
else:
    reducedData = None
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
