
import sys
from mpi4py import MPI
import numpy as np
import itertools
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
params = parseParamFile(sys.argv[1], verbose=True)

# Read filter coefficients, compute normalization of filters
bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, norms\
    = readBandCoefficients(params)
numBands = bandCoefAmplitudes.shape[0]

redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)
numZbins = redshiftDistGrid.size - 1
numZ = redshiftGrid.size

numObjectsTraining = np.sum(1 for line in open(params['training_catFile']))
numObjectsTarget = np.sum(1 for line in open(params['target_catFile']))
if threadNum == 0:
    print('Number of Training Objects', numObjectsTraining)
    print('Number of Target Objects', numObjectsTarget)

V_C_grid = np.logspace(-1, numThreads - 2, numThreads)
alpha_C_grid = [1e3]
i_V_C = 1*threadNum
V_C = V_C_grid[i_V_C]
numVC, numAlpha = V_C_grid.size, len(alpha_C_grid)
i_V_C = 1*threadNum
V_C = V_C_grid[i_V_C]
V_L = 1e3 * V_C
alpha_L = 1e2

numConfLevels = len(params['confidenceLevels'])
localNobj = np.zeros((numVC, numAlpha, numZbins))
localConfFractions = np.zeros((numVC, numAlpha, numConfLevels, numZbins))
localStackedPdfs = np.zeros((numVC, numAlpha, numZ, numZbins))
localZspecmean = np.zeros((numVC, numAlpha, numZbins))

comm.Barrier()
for ialpha, alpha_C in enumerate(alpha_C_grid):

    # Create Gaussian process mean fct and kernel
    gp = PhotozGP(0.0, bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
                  params['lines_pos'], params['lines_width'],
                  V_C, V_L,
                  alpha_C, alpha_L,
                  redshiftGridGP, use_interpolators=True)

    model_mean = np.zeros((numZ, numObjectsTraining, numBands))
    model_var = np.zeros((numZ, numObjectsTraining, numBands))

    loc = - 1
    trainingDataIter = getDataFromFile(params, 0, numObjectsTraining,
                                       prefix="training_", getXY=True)
    for z, ell, bands, fluxes, fluxesVar, X, Y, Yvar in trainingDataIter:
        loc += 1
        gp.setData(X, Y, Yvar)
        gp.optimizeAlpha()
        model_mean[:, loc, :], model_var[:, loc, :], g =\
            gp.predictAndInterpolate(redshiftGrid, ell=ell, z=z)

    loc = - 1
    targetDataIter = getDataFromFile(params, 0, numObjectsTarget,
                                     prefix="target_", getXY=False)
    for z, ell, bands, fluxes, fluxesVar in targetDataIter:
        loc += 1
        like_grid = scalefree_flux_likelihood(
            fluxes / ell, fluxesVar / ell**2.,
            model_mean[:, :, bands],  # model mean
            f_mod_var=model_var[:, :, bands]  # model var
        )
        pdf = like_grid.sum(axis=1)
        if pdf.sum() == 0:
            print("NULL PDF with galaxy", loc)
        metrics\
            = computeMetrics(z, redshiftGrid, pdf,
                             params['confidenceLevels'])
        ztrue, zmean, zmap, pdfAtZ, cumPdfAtZ = metrics[0:5]
        confidencelevels = metrics[5:]
        zmeanBinLoc = -1
        for i in range(numZbins):
            if zmean >= redshiftDistGrid[i]\
                    and zmean < redshiftDistGrid[i+1]:
                zmeanBinLoc = i
                localNobj[i_V_C, ialpha, i] += 1
                localZspecmean[i_V_C, ialpha, i] += ztrue
        for i in range(numConfLevels):
            if pdfAtZ >= confidencelevels[i]:
                localConfFractions[i_V_C, ialpha, i, zmeanBinLoc] += 1
        pdf /= np.trapz(pdf, x=redshiftGrid)
        localStackedPdfs[i_V_C, ialpha, :, zmeanBinLoc]\
            += pdf / numObjectsTraining


globalConfFractions = np.zeros_like(localConfFractions)
globalNobj = np.zeros_like(localNobj)
globalZspecmean = np.zeros_like(localZspecmean)
globalStackedPdfs = np.zeros_like(localStackedPdfs)

comm.Allreduce(localZspecmean, globalZspecmean, op=MPI.SUM)
comm.Allreduce(localConfFractions, globalConfFractions, op=MPI.SUM)
comm.Allreduce(localNobj, globalNobj, op=MPI.SUM)
comm.Allreduce(localStackedPdfs, globalStackedPdfs, op=MPI.SUM)
comm.Barrier()


if threadNum == 0:
    metric = np.zeros((numVC, numAlpha, numZbins))
    globalConfFractions /= globalNobj[:, :, None, :]
    globalZspecmean /= globalNobj
    for i_V_C, V_C in enumerate(V_C_grid):
        for ialpha, alpha_C in enumerate(alpha_C_grid):
            print("")
            print(" V_C", V_C, "alpha", alpha_C)
            for i in range(numZbins):
                print(" N(z) bin", i,
                      "zlo", redshiftDistGrid[i],
                      "zhi", redshiftDistGrid[i+1],
                      "nobj=", globalNobj[i_V_C, ialpha, i], end="")
                if globalNobj[i_V_C, ialpha, i] > 0:
                    pdfzmean = np.average(
                        redshiftGrid,
                        weights=globalStackedPdfs[i_V_C, ialpha, :, i])
                    metric[i_V_C, ialpha, i]\
                        = globalZspecmean[i_V_C, ialpha, i] - pdfzmean
                    print(globalZspecmean[i_V_C, ialpha, i], pdfzmean)
                    for k in range(numConfLevels):
                        print("  > CI:", params['confidenceLevels'][k],
                              '%.g' % globalConfFractions[i_V_C, ialpha, k, i],
                              end="")
                print("")

    for i_V_C, V_C in enumerate(V_C_grid):
        for ialpha, alpha_C in enumerate(alpha_C_grid):
            print("\n")
            print(" V_C", V_C, "alpha", alpha_C,
                  "Nobj", globalNobj[i_V_C, ialpha, :].sum())
            for i in range(numZbins):
                print("  Redshift mean bias in", i,
                      "th bin:", metric[i_V_C, ialpha, i],
                      globalNobj[i_V_C, ialpha, i])
    print("\n")
    for i_V_C, V_C in enumerate(V_C_grid):
        for ialpha, alpha_C in enumerate(alpha_C_grid):
            num = (globalNobj[i_V_C, ialpha, :] > 0).sum()
            print(" V_C", V_C, "alpha", alpha_C, "Nobj",
                  globalNobj[i_V_C, ialpha, :].sum(),
                  "Average redshift mean bias: ",
                  np.abs(metric[i_V_C, ialpha, :]).sum() / num)
    print("\n")
