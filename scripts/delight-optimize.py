
import sys
from mpi4py import MPI
import numpy as np
import itertools
from delight.utils import parseParamFile,\
    readColumnPositions, readBandCoefficients
from delight.utils import approx_DL, scalefree_flux_likelihood, computeMetrics
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


redshiftDistGrid = np.arange(0, params['redshiftMax'],
                             params['redshiftDisBinSize'])
redshiftGrid\
    = np.arange(0, params['redshiftMax'], params['redshiftBinSize'])
redshiftGridGP\
    = np.arange(0, params['redshiftMax'], params['redshiftBinSizeGPpred'])
numZbins = redshiftDistGrid.size - 1
numZ = redshiftGrid.size
numZGP = redshiftGridGP.size
xv, yv = np.meshgrid(redshiftGridGP, np.arange(numBands),
                     sparse=False, indexing='xy')
X_pred = np.ones((numBands*numZGP, 3))
X_pred[:, 0] = yv.flatten()
X_pred[:, 1] = xv.flatten()

# Locate which columns of the catalog correspond to which bands.
bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn,\
    refBandColumn = readColumnPositions(params, pfx="training_")
refBandNorm = norms[params['bandNames'].index(params['referenceBand'])]

numObjectsTraining = np.sum(1 for line in open(params['training_catFile']))
if threadNum == 0:
    print('Number of Training Objects', numObjectsTraining)
comm.Barrier()

#V_C_grid = np.logspace(-1, 2, numThreads)
V_C_grid = np.logspace(0, 2, 3)
alpha_C_grid = [5e2, 1e3]

i_V_C = 1*threadNum
V_C = V_C_grid[i_V_C]

alpha_L = 1e2
alpha = 0.0

numVC, numAlpha = V_C_grid.size, len(alpha_C_grid)
numConfLevels = len(params['confidenceLevels'])
localNobj = np.zeros((numVC, numAlpha, numZbins))
localConfFractions = np.zeros((numVC, numAlpha, numConfLevels, numZbins))
localStackedPdfs = np.zeros((numVC, numAlpha, numZ, numZbins))
localZspecmean = np.zeros((numVC, numAlpha, numZbins))

comm.Barrier()
for i_V_C, V_C in enumerate(V_C_grid):
    V_L = 1e3 * V_C
    for ialpha, alpha_C in enumerate(alpha_C_grid):

        # Create Gaussian process mean fct and kernel
        mean_fct = Photoz_mean_function(
            alpha, bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
            g_AB=1.0, lambdaRef=4.5e3, DL_z=DL)
        kernel = Photoz_kernel(
            bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
            params['lines_pos'], params['lines_width'], V_C, V_L, alpha_C, alpha_L,
            g_AB=1.0, DL_z=DL, redshiftGrid=redshiftGridGP,
            use_interpolators=True)

        model_mean = np.zeros((numZ, numObjectsTraining, numBands))
        model_var = np.zeros((numZ, numObjectsTraining, numBands))

        loc = - 1
        with open(params['training_catFile']) as f:
            for line in itertools.islice(f, 0, numObjectsTraining):
                loc += 1
                data = np.array(line.split(' '), dtype=float)
                refFlux = data[refBandColumn]
                z = data[redshiftColumn]
                ell = refFlux * ((1+z)**2./DL(z)**2. / refBandNorm) * 1000

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
                    X[off, 2] = ell
                    Y[off, 0] = data[bandColumns[off]]
                    Yvar[off, 0] = data[bandVarColumns[off]]

                gp = PhotozGP(mean_fct, kernel, X=X, Y=Y, Yvar=Yvar)
                gp.optimize_alpha()

                if params['training_optimize']:
                    y_pred, y_pred_fullcov = gp.predict(X_pred)
                    for i in range(numBands):
                        y_pred_bin\
                            = y_pred[i*numZGP:(i+1)*numZGP].ravel() / ell
                        y_var_bin\
                            = np.diag(y_pred_fullcov)[i*numZGP:(i+1)*numZGP]\
                            / ell**2
                        model_mean[:, loc,  i] =\
                            np.interp(redshiftGrid, redshiftGridGP, y_pred_bin)
                        model_var[:, loc,  i] =\
                            np.interp(redshiftGrid, redshiftGridGP, y_var_bin)

        loc = - 1
        with open(params['training_catFile']) as f:
            for line in itertools.islice(f, 0, numObjectsTraining):
                loc += 1
                data = np.array(line.split(' '), dtype=float)
                refFlux = data[refBandColumn]
                z = data[redshiftColumn]

                # drop bad values and find how many bands are valid
                mask = np.isfinite(data[bandColumns])
                mask &= np.isfinite(data[bandVarColumns])
                mask &= data[bandColumns] > 0.0
                mask &= data[bandVarColumns] > 0.0

                like_grid = scalefree_flux_likelihood(
                    data[bandColumns[mask]],  # fluxes
                    data[bandVarColumns[mask]],  # flux var
                    model_mean[:, :, bandIndices[mask]],  # model mean
                    f_mod_var=model_var[:, :, bandIndices[mask]]  # model var
                )
                pdf = like_grid.sum(axis=1)
                if pdf.sum() > 0:
                    metrics\
                        = computeMetrics(z, redshiftGrid, pdf, params['confidenceLevels'])
                    ztrue, zmean, zmap, pdfAtZ, cumPdfAtZ = metrics[0:5]
                    confidencelevels = metrics[5:]
                    zmeanBinLoc = -1
                    for i in range(numZbins):
                        if zmean >= redshiftDistGrid[i] and zmean < redshiftDistGrid[i+1]:
                            zmeanBinLoc = i
                            localNobj[i_V_C, ialpha, i] += 1
                            localZspecmean[i_V_C, ialpha, i] += ztrue

                    for i in range(numConfLevels):
                        if pdfAtZ >= confidencelevels[i]:
                            localConfFractions[i_V_C, ialpha, i, zmeanBinLoc] += 1
                    pdf /= np.trapz(pdf, x=redshiftGrid)
                    localStackedPdfs[i_V_C, ialpha, :, zmeanBinLoc]\
                        += pdf / numObjectsTraining


if threadNum == 0:
    globalConfFractions = np.zeros_like(localConfFractions)
    globalStackedPdfs = np.zeros_like(localStackedPdfs)
    globalNobj = np.zeros_like(localNobj)
    globalZspecmean = np.zeros_like(localZspecmean)
else:
    globalConfFractions = None
    globalStackedPdfs = None
    globalNobj = None
    globalZspecmean = None

comm.Barrier()
comm.Allreduce(localStackedPdfs, globalStackedPdfs, op=MPI.SUM)
comm.Allreduce(localNobj, globalNobj, op=MPI.SUM)
comm.Allreduce(localZspecmean, globalZspecmean, op=MPI.SUM)
comm.Allreduce(localConfFractions, globalConfFractions, op=MPI.SUM)
comm.Barrier()


if threadNum == 0:

    metric = np.zeros((numVC, numAlpha, numZbins))
    globalConfFractions /= globalNobj[:, :, None, :]
    globalZspecmean /= globalNobj
    for i_V_C, V_C in enumerate(V_C_grid):
        for ialpha, alpha_C in enumerate(alpha_C_grid):
            print("V_C", V_C, "alpha", alpha_C)
            for i in range(numZbins):
                print("> N(z) bin", i, "zlo", redshiftDistGrid[i], "zhi", redshiftDistGrid[i+1], "nobj=", globalNobj[i_V_C, ialpha, i])
                if globalNobj[i_V_C, ialpha, i] > 0:
                    pdfzmean = np.average(redshiftGrid, weights=globalStackedPdfs[i_V_C, ialpha, :, i])
                    print("  > zspecmean", '%.3g' %globalZspecmean[i_V_C, ialpha,i], "pdfzmean", '%.3g' %pdfzmean)
                    metric[i_V_C, ialpha, i] = globalZspecmean[i_V_C, ialpha,i] - pdfzmean
                    for k in range(numConfLevels):
                        print("  > CI:", params['confidenceLevels'][k], '%.3g' % globalConfFractions[i_V_C, ialpha, k, i], end="")
                    print("")

    for i_V_C, V_C in enumerate(V_C_grid):
        for ialpha, alpha_C in enumerate(alpha_C_grid):
            print("V_C", V_C, "alpha", alpha_C)
            for i in range(numZbins):
                print("  Redshift mean bias in", i, "th bin:", metric[i_V_C, ialpha, i])
    for i_V_C, V_C in enumerate(V_C_grid):
        for ialpha, alpha_C in enumerate(alpha_C_grid):
            print(">>> V_C", V_C, "alpha", alpha_C, "Average redshift mean bias: ", np.abs(metric[i_V_C, ialpha, :]).sum() / numZbins)
