
import sys
import numpy as np
import itertools
from delight.io import *
from delight.photoz_gp import PhotozGP
import matplotlib.pyplot as plt

# Parse parameters file
if len(sys.argv) < 2:
    raise Exception('Please provide a parameter file')
params = parseParamFile(sys.argv[1], verbose=True)

# Read filter coefficients, compute normalization of filters
bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, norms\
    = readBandCoefficients(params)
numBands = bandCoefAmplitudes.shape[0]

redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)

numObjectsTraining = np.sum(1 for line in open(params['training_catFile']))
print('Number of Training Objects', numObjectsTraining)

gp = PhotozGP(0.0, bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
              params['lines_pos'], params['lines_width'],
              params['V_C'], params['V_L'],
              params['alpha_C'], params['alpha_L'],
              redshiftGridGP, use_interpolators=True)

numZ = redshiftGrid.size
model_mean = np.zeros((numZ, numObjectsTraining, numBands))
model_var = np.zeros((numZ, numObjectsTraining, numBands))

loc = - 1
trainingDataIter1 = getDataFromFile(params, 0, numObjectsTraining,
                                    prefix="training_", getXY=True)
for z, ell, bands, fluxes, fluxesVar, X, Y, Yvar in trainingDataIter1:
    loc += 1

    gp.setData(X, Y, Yvar)

    fig, axs = plt.subplots(3, 2, figsize=(10, 5))
    axs = axs.ravel()
    for off, iband in enumerate(bands):
        axs[iband].errorbar(z, fluxes[off] / ell * z**2,
                            np.sqrt(fluxesVar[off]) / ell * z**2,
                            fmt='-o')
    fac = redshiftGrid**2

    model_mean, model_var = gp.predictAndInterpolate(redshiftGrid,
                                                     ell=ell, z=z)
    model_sig = np.sqrt(model_var)
    for i in range(numBands):
        axs[i].fill_between(redshiftGrid,
                            (model_mean[:, i] - model_sig[:, i])*fac,
                            (model_mean[:, i] + model_sig[:, i])*fac,
                            color='b', alpha=0.25)
        axs[i].plot(redshiftGrid, model_mean[:, i]*fac, c='b',
                    label='alpha = %.2g' % gp.mean_fct.alpha)

    gp.optimizeAlpha()

    model_mean, model_var\
        = gp.predictAndInterpolate(redshiftGrid, ell=ell, z=z)
    model_sig = np.sqrt(model_var)
    for i in range(numBands):
        axs[i].fill_between(redshiftGrid,
                            (model_mean[:, i] - model_sig[:, i])*fac,
                            (model_mean[:, i] + model_sig[:, i])*fac,
                            color='r', alpha=0.25)
        axs[i].plot(redshiftGrid, model_mean[:, i]*fac, c='r',
                    label='alpha = %.2g' % gp.mean_fct.alpha)

        axs[i].set_xscale('log')
        axs[i].set_yscale('log')
        axs[i].set_xlim([redshiftGrid[0], redshiftGrid[-1]])
    fig.savefig('data/pdfs-'+str(loc)+'.png')
