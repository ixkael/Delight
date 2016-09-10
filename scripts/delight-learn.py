
import sys
from mpi4py import MPI
import numpy as np
import itertools
from delight.utils import parseParamFile,\
    readColumnPositions, readBandCoefficients
from delight.utils import approx_DL
from delight.photoz_gp import PhotozGP
from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel

comm = MPI.COMM_WORLD
threadNum = comm.Get_rank()
numThreads = comm.Get_size()

# Parse parameters file
if len(sys.argv) < 2:
    raise Exception('Please provide a parameter file')
paramFileName = sys.argv[1]
params = parseParamFile(paramFileName)
if threadNum == 0:
    print('Input parameter file:', paramFileName)
    print('Parameters read:')
    for k, v in params.items():
        print('> ', "%-20s" % k, end="")
        print(' '.join([str(x) for x in v])) if type(v) is list\
            else print(v)

# Read filter coefficients, compute normalization of filters
bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, norms\
    = readBandCoefficients(params)
DL = approx_DL()  # Luminosity distance function.
numBands = bandCoefAmplitudes.shape[0]

redshiftGrid\
    = np.arange(0, params['redshiftMax'], params['redshiftBinSizeGPpred'])

# Create Gaussian process mean fct and kernel
alpha = 0.0
mean_fct = Photoz_mean_function(
    alpha, bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
    g_AB=1.0, lambdaRef=4.5e3, DL_z=DL)
kernel = Photoz_kernel(
    bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
    params['lines_pos'], params['lines_width'],
    params['V_C'], params['V_L'], params['alpha_C'], params['alpha_L'],
    g_AB=1.0, DL_z=DL, redshiftGrid=redshiftGrid, use_interpolators=True)

# Locate which columns of the catalog correspond to which bands.
bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn,\
    refBandColumn = readColumnPositions(params, pfx="training_")
refBandNorm = norms[params['bandNames'].index(params['referenceBand'])]

numObjectsTraining = np.sum(1 for line in open(params['training_catFile']))
firstLine = int(threadNum * numObjectsTraining / numThreads)
lastLine = int(min(numObjectsTraining,
               (threadNum + 1) * numObjectsTraining / numThreads))
numLines = lastLine - firstLine
if threadNum == 0:
    print('Number of Training Objects', numObjectsTraining)
comm.Barrier()
print('Thread ', threadNum, ' analyzes lines ', firstLine, ' to ', lastLine)
B = numBands
numCol = 4 + B + B*(B+1)//2 + B
localData = np.zeros((numLines, numCol))
fmt = '%i ' + '%.6e ' * (localData.shape[1] - 1)

loc = - 1
with open(params['training_catFile']) as f:
    for line in itertools.islice(f, firstLine, lastLine):
        loc += 1
        data = np.array(line.split(' '), dtype=float)

        refFlux = data[refBandColumn]
        z = data[redshiftColumn]
        luminosity_estimate = refFlux\
            * ((1+z)**2./DL(z)**2. / refBandNorm)

        # drop bad values and find how many bands are valid
        mask = np.isfinite(data[bandColumns])
        mask &= np.isfinite(data[bandVarColumns])
        mask &= data[bandColumns] > 0.0
        mask &= data[bandVarColumns] > 0.0
        bandsUsed = np.where(mask)[0]
        numBandsUsed = mask.sum()

        if (refFlux <= 0) or (not np.isfinite(refFlux))\
                or (z < 0) or (numBandsUsed <= 1):
            continue  # not valid data - skip to next valid object

        Y = np.zeros((numBandsUsed, 1))
        Yvar = np.zeros((numBandsUsed, 1))
        X = np.ones((numBandsUsed, 3))
        for off, iband in enumerate(bandIndices[mask]):
            X[off, 0] = iband
            X[off, 1] = z
            X[off, 2] = luminosity_estimate
            Y[off, 0] = data[bandColumns[off]]
            Yvar[off, 0] = data[bandVarColumns[off]]

        gp = PhotozGP(mean_fct, kernel, X=X, Y=Y, Yvar=Yvar)
        gp.optimize_alpha()

        B = numBandsUsed
        # Order: B, z, ell, alpha, BandsUsed, L, beta
        localData[loc, 0] = B
        localData[loc, 1] = z
        localData[loc, 2] = luminosity_estimate
        localData[loc, 3] = gp.mean_fct.alpha
        localData[loc, 4:4+B] = bandIndices[bandsUsed]
        localData[loc, 4+B:4+B+B*(B+1)//2] = gp.L[np.tril_indices(B)]
        localData[loc, 4+B+B*(B+1)//2:4+B+B*(B+1)//2+B] = gp.beta.ravel()

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
