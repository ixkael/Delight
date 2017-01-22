
import sys
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
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

# Read filter coefficients, compute normalization of filters
bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, norms\
    = readBandCoefficients(params)
DL = approx_DL()  # Luminosity distance function.
numBands = bandCoefAmplitudes.shape[0]

redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)
numZbins = redshiftDistGrid.size - 1
numZ = redshiftGrid.size

fig, axs = plt.subplots(numZbins, 2, figsize=(10, 10))

for iax, extra in enumerate(['', 'Temp']):
    numObjectsTarget = np.sum(1 for line in open(params['metricsFile'+extra]))
    fpdf = open(params['redshiftpdfFile'+extra])
    fmet = open(params['metricsFile'+extra])

    # Create local files to store results
    numConfLevels = len(params['confidenceLevels'])
    # Now loop over target set to compute likelihood function
    loc = -1
    targetDataIter = getDataFromFile(params, 0, numObjectsTarget,
                                     prefix="target_", getXY=False)

    bias_zmap = np.zeros((redshiftDistGrid.size, ))
    bias_zmean = np.zeros((redshiftDistGrid.size, ))
    confFractions = np.zeros((numConfLevels, redshiftDistGrid.size))
    binnobj = np.zeros((redshiftDistGrid.size, ))
    binlocsz = np.zeros((numObjectsTarget, 2))
    bias_nz = np.zeros((redshiftDistGrid.size, ))
    stackedPdfs = np.zeros((redshiftGrid.size, redshiftDistGrid.size))

    # Now loop over target set to compute likelihood function
    loc = - 1
    iterpdf = itertools.islice(fpdf, 0, numObjectsTarget)
    itermet = itertools.islice(fmet, 0, numObjectsTarget)
    for loc in range(numObjectsTarget):
        pdf = np.array(next(iterpdf).split(' '), dtype=float)
        metrics = np.array(next(itermet).split(' '), dtype=float)
        ztrue, zmean, zstdzmean, zmap, zstdzmap, pdfAtZ, cumPdfAtZ\
            = metrics[0:7]
        confidencelevels = metrics[7:]
        zmeanBinLoc = -1
        if True:  # np.abs(zmean - zmap) < 2.5:
            for i in range(numZbins):
                if zmap >= redshiftDistGrid[i]\
                        and zmap < redshiftDistGrid[i+1]:
                    zmeanBinLoc = i
                    bias_zmap[i] += ztrue - zmap  # np.abs(ztrue - zmap)
                    bias_zmean[i] += ztrue - zmean  # np.abs(ztrue - zmean)
                    binnobj[i] += 1
                    binlocsz[loc, 0] = i
                    binlocsz[loc, 1] = ztrue
                    bias_nz[i] += ztrue
            for i in range(numConfLevels):
                if pdfAtZ >= confidencelevels[i]:
                    confFractions[i, zmeanBinLoc] += 1
            # pdf /= np.trapz(pdf, x=redshiftGrid)
            stackedPdfs[:, zmeanBinLoc]\
                += pdf / numObjectsTarget

    confFractions /= binnobj[None, :]
    bias_nz /= binnobj
    for i in range(numZbins):
        if stackedPdfs[:, i].sum():
            bias_nz[i] -= np.average(redshiftGrid, weights=stackedPdfs[:, i])
    ind = binnobj > 0
    bias_zmap /= binnobj
    bias_zmean /= binnobj

    print(' >> bias_zmap %.3g' % np.abs(bias_zmap[ind]).mean(),
          'bias_zmean %.3g' % np.abs(bias_zmean[ind]).mean(),
          'N(z) bias %.3g' % np.abs(bias_nz[ind]).mean(), ' <<')
    print(' > bias_zmap : ', ' '.join(['%.3g' % x for x in bias_zmap]))
    print(' > bias_zmean : ', ' '.join(['%.3g' % x for x in bias_zmean]))
    print(' > nzbias : ', ' '.join(['%.3g' % x for x in bias_nz]))
    for i in range(numConfLevels):
        print(' >', params['confidenceLevels'][i], ' :: ',
              ' '.join(['%.3g' % x for x in confFractions[i, :]]))

    for i in range(numZbins):
        print("> N(z) bin", i, "zlo", redshiftDistGrid[i], "zhi",
              redshiftDistGrid[i+1], "nobj=", binnobj[i])
        if binnobj[i] > 1:
            pdfzmean = np.average(redshiftGrid,
                                  weights=stackedPdfs[:, i])
            ind = (binlocsz[:, 0] == i)
            pdf = stackedPdfs[:, i]
            if pdf.sum() > 0:
                pdf /= np.trapz(pdf, x=redshiftGrid)
            axs[i, iax].plot(redshiftGrid, pdf,
                             label='Inferred', color='b')
            if ind.sum() > 1:
                density = stats.kde.gaussian_kde(binlocsz[ind, 1])
                axs[i, iax].plot(redshiftGrid, density(redshiftGrid),
                                 label='Data KDE '+extra, c='k')
                axs[i, iax].hist(binlocsz[ind, 1], 50, normed=True,
                                 range=[0, redshiftGrid[-1]], histtype='step',
                                 label='Data hist '+extra, color='gray')
            axs[i, iax].axvline(redshiftDistGrid[i], ls='dashed', color='k')
            axs[i, iax].axvline(redshiftDistGrid[i+1], ls='dashed', color='k')

axs[0, 0].legend(loc='upper right')
fig.tight_layout()
plt.show()
