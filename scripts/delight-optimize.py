
import sys
from mpi4py import MPI
import numpy as np
from delight.io import *
from delight.utils import *
from delight.photoz_gp import PhotozGP
from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel
from delight.utils_cy import approx_flux_likelihood_cy

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
DL = approx_DL()

dir_seds = params['templates_directory']
dir_filters = params['bands_directory']
lambdaRef = params['lambdaRef']
sed_names = params['templates_names']
f_mod_grid = np.zeros((redshiftGrid.size, len(sed_names),
                       len(params['bandNames'])))
for t, sed_name in enumerate(sed_names):
    f_mod_grid[:, t, :] = np.loadtxt(dir_seds + '/' + sed_name +
                                     '_fluxredshiftmod.txt')

numZbins = redshiftDistGrid.size - 1
numZ = redshiftGrid.size
numConfLevels = len(params['confidenceLevels'])
numObjectsTraining = np.sum(1 for line in open(params['training_catFile']))
numObjectsTarget = np.sum(1 for line in open(params['target_catFile']))
print('Number of Training Objects', numObjectsTraining)
print('Number of Target Objects', numObjectsTarget)

for ellPriorSigma in [1.0, 10.0]:
    alpha_C = 1e3
    alpha_L = 1e2
    V_C, V_L = 1.0, 1.0
    gp = PhotozGP(
        f_mod,
        bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
        params['lines_pos'], params['lines_width'],
        V_C, V_L, alpha_C, alpha_L,
        redshiftGridGP, use_interpolators=True)

    for extraFracFluxError in [1e-2]:
        redshifts = np.zeros((numObjectsTraining, ))
        bestTypes = np.zeros((numObjectsTraining, ), dtype=int)
        ellMLs = np.zeros((numObjectsTraining, ))
        model_mean = np.zeros((numZ, numObjectsTraining, numBands))
        model_covar = np.zeros((numZ, numObjectsTraining, numBands))
        # params['training_extraFracFluxError'] = extraFracFluxError
        params['target_extraFracFluxError'] = extraFracFluxError

        for V_C in [0.1, 1.0]:

            gp.var_C = V_C
            gp.var_L = V_L
            loc = - 1
            trainingDataIter = getDataFromFile(
                params, 0, numObjectsTraining,
                prefix="training_", getXY=True)
            for z, normedRefFlux, bands, fluxes,\
                fluxesVar, bCV, fCV, fvCV, X, Y, Yvar\
                    in trainingDataIter:
                loc += 1
                redshifts[loc] = z
                themod = np.zeros((1, f_mod_grid.shape[1], bands.size))
                for it in range(f_mod_grid.shape[1]):
                    for ib, band in enumerate(bands):
                        themod[0, it, ib] = np.interp(z, redshiftGrid,
                                                      f_mod_grid[:, it, band])
                chi2_grid, theellMLs = scalefree_flux_likelihood(
                    fluxes,
                    fluxesVar,
                    themod,
                    returnChi2=True
                )
                bestTypes[loc] = np.argmin(chi2_grid)
                ellMLs[loc] = theellMLs[0, bestTypes[loc]]
                X[:, 2] = ellMLs[loc]
                gp.setData(X, Y, Yvar, bestTypes[loc])
                model_mean[:, loc, :], model_covar[:, loc, :] =\
                    gp.predictAndInterpolate(redshiftGrid, ell=ellMLs[loc])

            for redshiftSigma in [0.1, 1.0]:

                loc = - 1
                targetDataIter = getDataFromFile(params, 0, numObjectsTarget,
                                                 prefix="target_", getXY=False)

                bias_zmap = np.zeros((redshiftDistGrid.size, ))
                bias_zmean = np.zeros((redshiftDistGrid.size, ))
                confFractions = np.zeros((numConfLevels,
                                          redshiftDistGrid.size))
                binnobj = np.zeros((redshiftDistGrid.size, ))
                bias_nz = np.zeros((redshiftDistGrid.size, ))
                stackedPdfs = np.zeros((redshiftGrid.size,
                                        redshiftDistGrid.size))
                cis = np.zeros((numObjectsTarget, ))
                zmeanBinLocs = np.zeros((numObjectsTarget, ), dtype=int)
                for z, normedRefFlux, bands, fluxes, fluxesVar, bCV, fCV, fvCV\
                        in targetDataIter:
                    loc += 1
                    like_grid = np.zeros((model_mean.shape[0],
                                          model_mean.shape[1]))
                    ell_hat_z = normedRefFlux * 4 * np.pi\
                        * params['fluxLuminosityNorm'] \
                        * (DL(redshiftGrid)**2. * (1+redshiftGrid))
                    ell_hat_z[:] = 1
                    approx_flux_likelihood_cy(
                        like_grid,
                        model_mean.shape[0], model_mean.shape[1], bands.size,
                        fluxes, fluxesVar,
                        model_mean[:, :, bands],
                        V_C*model_covar[:, :, bands],
                        ell_hat_z, (ell_hat_z*ellPriorSigma)**2)
                    like_grid *= np.exp(-0.5*((redshiftGrid[:, None] -
                                               redshifts[None, :]) /
                                              redshiftSigma)**2)
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
                                bias_zmap[i] += ztrue - zmap
                                bias_zmean[i] += ztrue - zmean
                                binnobj[i] += 1
                                bias_nz[i] += ztrue
                        zmeanBinLocs[loc] = zmeanBinLoc
                        for i in range(numConfLevels):
                            if pdfAtZ >= confidencelevels[i]:
                                confFractions[i, zmeanBinLoc] += 1
                        stackedPdfs[:, zmeanBinLoc]\
                            += pdf / numObjectsTraining
                        ind = pdf >= pdfAtZ
                        pdf /= np.trapz(pdf, x=redshiftGrid)
                        cis[loc] = np.trapz(pdf[ind], x=redshiftGrid[ind])

                confFractions /= binnobj[None, :]
                bias_nz /= binnobj
                for i in range(numZbins):
                    if stackedPdfs[:, i].sum():
                        bias_nz[i] -= np.average(redshiftGrid,
                                                 weights=stackedPdfs[:, i])
                ind = binnobj > 0
                bias_zmap /= binnobj
                bias_zmean /= binnobj
                print("")
                print(' =======================================')
                print("  ellSTD", ellPriorSigma,
                      "fluxError", extraFracFluxError,
                      "V_C", V_C, "zSTD", redshiftSigma)
                cis_pdf, e = np.histogram(cis, 50, range=[0, 1])
                cis_pdfcum = np.cumsum(cis_pdf) / np.sum(cis_pdf)
                print("-------------------------------->>  %.3g"
                      % (np.max(np.abs(np.abs(e[1:] - cis_pdfcum)))))
                print(">>", end="")
                for i in range(numZbins):
                    ind2 = zmeanBinLocs == i
                    if ind2.sum() > 2:
                        cis_pdf, e = np.histogram(cis[ind2], 50, range=[0, 1])
                        cis_pdfcum = np.cumsum(cis_pdf) / np.sum(cis_pdf)
                        print("  %.3g" % (np.max(np.abs(e[1:] - cis_pdfcum))),
                              end=" ")
                # print("")
                # print(' >>>> mean z bias %.3g'
                # % np.abs(bias_zmean[ind]).mean(),
                # 'mean N(z) bias %.3g' % np.abs(bias_nz[ind]).mean(), ' <<<<')
                # print(' >>>> max z bias %.3g'
                # % np.abs(bias_zmean[ind]).max(),
                # 'max N(z) bias %.3g' % np.abs(bias_nz[ind]).max(), ' <<<<')
                # print(' > bias_zmap : ',
                # '  '.join(['%.3g' % x for x in bias_zmap]))
                # print(' > z bias : ',
                # '  '.join([('%.3g' % x) if np.isfinite(x)
                # else '.' for x in bias_zmean]))
                # print(' > nzbias : ',
                # '  '.join([('%.3g' % x) if np.isfinite(x)
                # else '.' for x in bias_nz]))
                # print(' --------------------------------')
                # for i in range(numConfLevels):
                #     print(' >', params['confidenceLevels'][i],
                # ' :: ', '  '.join([('%.3g' % x) if np.isfinite(x)
                # else '.' for x in confFractions[i, :]]))
                # print(' =======================================')
