
import sys
from mpi4py import MPI
import matplotlib.pyplot as plt
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
paramFileName = sys.argv[1]
params = parseParamFile(paramFileName)
if threadNum == 0:
    print('Thread number / number of threads: ', threadNum+1, numThreads)
    print('Input parameter file:', paramFileName)

# Read filter coefficients, compute normalization of filters
bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, norms\
    = readBandCoefficients(params)
DL = approx_DL()  # Luminosity distance function.
numBands = bandCoefAmplitudes.shape[0]

redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)
numZbins = redshiftDistGrid.size - 1
numZ = redshiftGrid.size

numObjectsTarget = np.sum(1 for line in open(params['metricsFile']))

firstLine = int(threadNum * numObjectsTarget / float(numThreads))
lastLine = int(min(numObjectsTarget,
               (threadNum + 1) * numObjectsTarget / float(numThreads)))
numLines = lastLine - firstLine
if threadNum == 0:
    print('Number of Target Objects', numObjectsTarget)
comm.Barrier()
print('Thread ', threadNum, ' analyzes lines ', firstLine, ' to ', lastLine)

# Create local files to store results
numConfLevels = len(params['confidenceLevels'])
localNobj = np.zeros((numZbins, ))
localConfFractions = np.zeros((numConfLevels, numZbins))
localStackedPdfs = np.zeros((numZ, numZbins))
localZspecmean = np.zeros((numZbins, ))
localBinlocs = np.zeros((numObjectsTarget, 2))

# Now loop over target set to compute likelihood function
loc = - 1
fpdf = open(params['redshiftpdfFile'])
fmet = open(params['metricsFile'])
iterpdf = itertools.islice(fpdf, firstLine, lastLine)
itermet = itertools.islice(fmet, firstLine, lastLine)
for loc in range(numLines):
    pdf = np.array(next(iterpdf).split(' '), dtype=float)

    metrics = np.array(next(itermet).split(' '), dtype=float)
    ztrue, zmean, zmap, pdfAtZ, cumPdfAtZ = metrics[0:5]
    confidencelevels = metrics[5:]

    zmeanBinLoc = -1
    for i in range(numZbins):
        if zmean >= redshiftDistGrid[i] and zmean < redshiftDistGrid[i+1]:
            zmeanBinLoc = i
            localBinlocs[loc, 0] = i
            localBinlocs[loc, 1] = ztrue
            localNobj[i] += 1
            localZspecmean[i] += ztrue

    for i in range(numConfLevels):
        if pdfAtZ >= confidencelevels[i]:
            localConfFractions[i, zmeanBinLoc] += 1
    pdf /= np.trapz(pdf, x=redshiftGrid)
    localStackedPdfs[:, zmeanBinLoc] += pdf / numObjectsTarget

comm.Barrier()
if threadNum == 0:
    globalConfFractions = np.zeros_like(localConfFractions)
    globalStackedPdfs = np.zeros_like(localStackedPdfs)
    globalNobj = np.zeros_like(localNobj)
    globalZspecmean = np.zeros_like(localZspecmean)
    globalBinlocs = np.zeros_like(localBinlocs)
else:
    globalConfFractions = None
    globalStackedPdfs = None
    globalNobj = None
    globalZspecmean = None
    globalBinlocs = None

comm.Allreduce(localConfFractions, globalConfFractions, op=MPI.SUM)
comm.Allreduce(localStackedPdfs, globalStackedPdfs, op=MPI.SUM)
comm.Allreduce(localNobj, globalNobj, op=MPI.SUM)
comm.Allreduce(localZspecmean, globalZspecmean, op=MPI.SUM)
comm.Allreduce(localBinlocs, globalBinlocs, op=MPI.SUM)
comm.Barrier()

if threadNum == 0:

    globalConfFractions /= globalNobj[None, :]
    globalZspecmean /= globalNobj

    fig, axs = plt.subplots(numZbins // 2 + 1, 2, figsize=(10, 10))
    axs = axs.ravel()
    for i in range(numZbins):
        print("> N(z) bin", i, "zlo", redshiftDistGrid[i], "zhi", redshiftDistGrid[i+1], "nobj=", globalNobj[i])
        if globalNobj[i] > 0:
            pdfzmean = np.average(redshiftGrid, weights=globalStackedPdfs[:, i])
            print("  > zspecmean", '%.3g' %globalZspecmean[i], "pdfzmean", '%.3g' %pdfzmean)
            for k in range(numConfLevels):
                print("  > CI:", params['confidenceLevels'][k], '%.3g' % globalConfFractions[k, i], end="")
            print("")
        ind = (globalBinlocs[:, 0] == i)
        pdf = globalStackedPdfs[:, i] / np.trapz(globalStackedPdfs[:, i], x=redshiftGrid)
        axs[i].plot(redshiftGrid, pdf)
        axs[i].hist(globalBinlocs[ind, 1], 50, normed=True, range=[0, redshiftGrid[-1]], histtype='step')
    fig.tight_layout()
    plt.show()
