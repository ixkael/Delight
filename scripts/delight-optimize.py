
import sys
from mpi4py import MPI
import numpy as np
from delight.io import *
from delight.utils import *
from delight.photoz_gp import PhotozGP
from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel

# Parse parameters file
if len(sys.argv) < 2:
    raise Exception('Please provide a parameter file')
params = parseParamFile(sys.argv[1], verbose=False)

# Read filter coefficients, compute normalization of filters
bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, norms\
    = readBandCoefficients(params)
numBands = bandCoefAmplitudes.shape[0]

redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)
f_mod = readSEDs(params)

numZbins = redshiftDistGrid.size - 1
numZ = redshiftGrid.size
numConfLevels = len(params['confidenceLevels'])
numObjectsTraining = np.sum(1 for line in open(params['training_catFile']))
numObjectsTarget = np.sum(1 for line in open(params['target_catFile']))
print('Number of Training Objects', numObjectsTraining)
print('Number of Target Objects', numObjectsTarget)

for alpha_C in [1e2, 1e3]:
    alpha_L = 1e2
    V_C, V_L = 1.0, 1.0
    gp = PhotozGP(
        f_mod,
        bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
        params['lines_pos'], params['lines_width'],
        V_C, V_L, alpha_C, alpha_L,
        redshiftGridGP, use_interpolators=True)

    for extraFracFluxError in [1e-2, 2e-2, 5e-2]:
        model_mean = np.zeros((numZ, numObjectsTraining, numBands))
        model_var = np.zeros((numZ, numObjectsTraining, numBands))
        params['training_extraFracFluxError'] = extraFracFluxError
        params['target_extraFracFluxError'] = extraFracFluxError

        for V_C in [1e-2, 1e-1, 5e-1]:

            gp.var_C = V_C
            gp.var_L = V_L
            loc = - 1
            trainingDataIter = getDataFromFile(
                params, 0, numObjectsTraining,
                prefix="training_", getXY=True)
            for z, ell, bands, fluxes, fluxesVar, bCV, fCV, fvCV, X, Y, Yvar\
                    in trainingDataIter:
                loc += 1
                gp.setData(X, Y, Yvar)
                #alpha, ell = gp.estimateAlphaEll()
                model_mean[:, loc, :], model_var[:, loc, :] =\
                    gp.predictAndInterpolate(redshiftGrid, ell=ell, z=z)

            loc = - 1
            targetDataIter = getDataFromFile(params, 0, numObjectsTarget,
                                             prefix="target_", getXY=False)

            bias_zmap = np.zeros((redshiftDistGrid.size, ))
            bias_zmean = np.zeros((redshiftDistGrid.size, ))
            confFractions = np.zeros((numConfLevels, redshiftDistGrid.size))
            binnobj = np.zeros((redshiftDistGrid.size, ))
            bias_nz = np.zeros((redshiftDistGrid.size, ))
            stackedPdfs = np.zeros((redshiftGrid.size, redshiftDistGrid.size))
            for z, ell, bands, fluxes, fluxesVar, bCV, fCV, fvCV\
                    in targetDataIter:
                loc += 1
                like_grid = scalefree_flux_likelihood(
                    fluxes / ell, fluxesVar / ell**2.,
                    model_mean[:, :, bands],  # model mean
                    f_mod_var=model_var[:, :, bands] * V_C  # model var SCALED
                )
                pdf = like_grid.sum(axis=1)
                if pdf.sum() == 0:
                    print("NULL PDF with galaxy", loc)
                if pdf.sum() > 0:
                    metrics\
                        = computeMetrics(z, redshiftGrid, pdf,
                                         params['confidenceLevels'])
                    ztrue, zmean, zstdzmean, zmap, zstdzmean,\
                        pdfAtZ, cumPdfAtZ = metrics[0:7]
                    confidencelevels = metrics[7:]
                    zmeanBinLoc = -1
                    for i in range(numZbins):
                        if zmean >= redshiftDistGrid[i]\
                                and zmean < redshiftDistGrid[i+1]:
                            zmeanBinLoc = i
                            bias_zmap[i] += ztrue - zmap  # np.abs(ztrue - zmap)
                            bias_zmean[i] += ztrue - zmean  # np.abs(ztrue - zmean)
                            binnobj[i] += 1
                            bias_nz[i] += ztrue
                    for i in range(numConfLevels):
                        if pdfAtZ >= confidencelevels[i]:
                            confFractions[i, zmeanBinLoc] += 1
                    # pdf /= np.trapz(pdf, x=redshiftGrid)
                    stackedPdfs[:, zmeanBinLoc]\
                        += pdf / numObjectsTraining
            confFractions /= binnobj[None, :]
            bias_nz /= binnobj
            for i in range(numZbins):
                if stackedPdfs[:, i].sum():
                    bias_nz[i] -= np.average(redshiftGrid, weights=stackedPdfs[:, i])
            ind = binnobj > 0
            bias_zmap /= binnobj
            bias_zmean /= binnobj
            print("")
            print("alphaC", alpha_C, "extraFracFluxError", extraFracFluxError, "V_C", V_C)
            print(' >> bias_zmap %.3g' % np.abs(bias_zmap[ind]).mean(), 'bias_zmean %.3g' % np.abs(bias_zmean[ind]).mean(), 'N(z) bias %.3g' % np.abs(bias_nz[ind]).mean(), ' <<')
            print(' > bias_zmap : ', ' '.join(['%.3g' % x for x in bias_zmap]))
            print(' > bias_zmean : ', ' '.join(['%.3g' % x for x in bias_zmean]))
            print(' > nzbias : ', ' '.join(['%.3g' % x for x in bias_nz]))
            for i in range(numConfLevels):
                print(' >', params['confidenceLevels'][i], ' :: ', ' '.join(['%.3g' % x for x in confFractions[i, :]]))
