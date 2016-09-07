
import sys
from mpi4py import MPI
import numpy as np
import itertools
from delight.utils import parseParamFile,\
    readColumnPositions, readBandCoefficients
from delight.utils import approx_DL, scalefree_flux_likelihood
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
if comm.rank == 0:
    print('Thread number / number of threads: ', threadNum+1, numThreads)
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

redshiftGrid = np.arange(0, params['redshiftMax'], params['redshiftBinSize'])
numZ = redshiftGrid.size
xv, yv = np.meshgrid(redshiftGrid, np.arange(numBands),
                     sparse=False, indexing='xy')
X_pred = np.ones((numBands*numZ, 3))
X_pred[:, 0] = yv.flatten()
X_pred[:, 1] = xv.flatten()

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
    refBandColumn = readColumnPositions(params, pfx="target_")
refBandNorm = norms[params['bandNames'].index(params['referenceBand'])]

Ncompress = params['Ncompress']
numObjectsTraining = np.sum(1 for line in open(params['training_catFile']))
numObjectsTarget = np.sum(1 for line in open(params['target_catFile']))

firstLine = int(threadNum * numObjectsTarget / float(numThreads))
lastLine = int(min(numObjectsTarget,
               (threadNum + 1) * numObjectsTarget / float(numThreads)))
if comm.rank == 0:
    print('Number of Training Objects', numObjectsTraining)
    print('Number of Target Objects', numObjectsTarget)
    print('Lines to analyze (in target): ', firstLine, lastLine)

# Create local files to store results
localPDFs = np.zeros((numObjectsTraining, numZ))
localCompressIndices = np.zeros((numObjectsTarget,  Ncompress))
localCompEvidences = np.zeros((numObjectsTarget,  Ncompress))

# Looping over chunks of the training set to prepare model predictions over z
numChunks = params['training_numChunks']
for chunk in range(numChunks):
    TR_firstLine = int(chunk * numObjectsTraining / float(numChunks))
    TR_lastLine = int(min(numObjectsTraining,
                      (chunk + 1) * numObjectsTarget / float(numChunks)))
    numTObjCk = TR_lastLine - TR_firstLine
    model_mean = np.zeros((numZ, numTObjCk, numBands))
    model_var = np.zeros((numZ, numTObjCk, numBands))
    loc = TR_firstLine - 1
    targetIndices = np.arange(TR_firstLine, TR_lastLine)
    with open(params['training_paramFile']) as f:
        for line in itertools.islice(f, TR_firstLine, TR_lastLine):
            loc += 1
            data = np.fromstring(line, dtype=float, sep=' ')
            # Order: B, z, ell, alpha, BandsUsed, L, beta
            B = int(data[0])
            ell = data[2]
            X = np.zeros((B, 3))
            X[:, 0] = data[4:4+B]  # bandsUsed
            X[:, 1] = data[1]  # z
            X[:, 2] = ell  # ell
            L = np.zeros((B, B))
            L[np.tril_indices(B)] = data[4+B:4+B+B*(B+1)//2]
            beta = data[4+B+B*(B+1)//2:4+B+B*(B+1)//2+B]
            gp = PhotozGP(mean_fct, kernel, X=X, L=L, beta=beta)
            gp.mean_fct.alpha = data[3]  # alpha
            X_pred[:, 2] = ell  # ell
            y_pred, y_pred_fullcov = gp.predict(X_pred)
            for i in range(numBands):
                model_mean[:, loc-TR_firstLine,  i] =\
                    y_pred[i*numZ:(i+1)*numZ].ravel() / ell
                model_var[:, loc-TR_firstLine,  i] =\
                    np.diag(y_pred_fullcov)[i*numZ:(i+1)*numZ] / ell**2.
    # Now loop over target set to compute likelihood function
    loc = firstLine - 1
    with open(params['target_catFile']) as f:
        for line in itertools.islice(f, firstLine, lastLine):
            loc += 1
            data = np.array(line.split(' '), dtype=float)

            refFlux = data[refBandColumn]
            z = data[redshiftColumn]
            luminosity_estimate = refFlux\
                * ((1+z)**2./DL(z)**2. / refBandNorm) * 1000.

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

            like_grid = scalefree_flux_likelihood(
                data[bandColumns[mask]],  # fluxes
                data[bandVarColumns[mask]],  # flux var
                model_mean[:, :, bandIndices[mask]],  # model mean
                f_mod_var=model_var[:, :, bandIndices[mask]]  # model var
            )
            localPDFs[loc, :] += like_grid.sum(axis=1)
            evidences = np.trapz(like_grid, x=redshiftGrid, axis=0)
            if localCompressIndices[loc, :].sum() == 0:
                sortind = np.argsort(evidences)[::-1][0:Ncompress]
                localCompressIndices[loc, :] = targetIndices[sortind]
                localCompEvidences[loc, :] = evidences[sortind]
            else:
                dind = np.concatenate((targetIndices,
                                       localCompressIndices[loc, :]))
                devi = np.concatenate((evidences,
                                       localCompEvidences[loc, :]))
                sortind = np.argsort(devi)[::-1][0:Ncompress]
                localCompressIndices[loc, :] = dind[sortind]
                localCompEvidences[loc, :] = devi[sortind]

comm.Barrier()
if comm.rank == 0:
    globalPDFs = np.zeros_like(localPDFs)
    globalCompressIndices = np.zeros_like(localCompressIndices)
    globalCompEvidences = np.zeros_like(localCompEvidences)
else:
    globalPDFs = None
    globalCompressIndices = None
    globalCompEvidences = None
comm.Reduce([localPDFs, MPI.DOUBLE],
            [globalPDFs, MPI.DOUBLE],
            op=MPI.SUM, root=0)
comm.Reduce([localCompressIndices, MPI.DOUBLE],
            [globalCompressIndices, MPI.DOUBLE],
            op=MPI.SUM, root=0)
comm.Reduce([localCompEvidences, MPI.DOUBLE],
            [globalCompEvidences, MPI.DOUBLE],
            op=MPI.SUM, root=0)
comm.Barrier()

fmt = '%.2e'
np.savetxt(params['redshiftpdfFile'], globalPDFs, fmt=fmt)
np.savetxt(params['compressMargLikFile'], globalCompressIndices, fmt=fmt)
np.savetxt(params['compressIndicesFile'], globalCompEvidences, fmt='%i')

## POST PROCESSING : find levels, z mean, z map, frac, etc
