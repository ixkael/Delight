import sys
import numpy as np
import matplotlib.pyplot as plt
from delight.photoz_gp import PhotozGP
from delight.io import *
from delight.utils import *

if len(sys.argv) < 2:
    raise Exception('Please provide a parameter file')
params = parseParamFile(sys.argv[1], verbose=False, catFilesNeeded=False)
redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)
numObjects = params['numObjects']
noiseLevel = params['noiseLevel']
f_mod = readSEDs(params)


bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, norms\
    = readBandCoefficients(params)
gp = PhotozGP(f_mod, bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
              params['lines_pos'], params['lines_width'],
              params['V_C'], params['V_L'],
              params['alpha_C'], params['alpha_L'],
              redshiftGridGP, use_interpolators=True)
DL = approx_DL()

bandIndices_TRN, bandNames_TRN, bandColumns_TRN,\
    bandVarColumns_TRN, redshiftColumn_TRN,\
    refBandColumn_TRN = readColumnPositions(params, prefix='training_')
bandIndices_TAR, bandNames_TAR, bandColumns_TAR,\
    bandVarColumns_TAR, redshiftColumn_TAR,\
    refBandColumn_TAR = readColumnPositions(params, prefix='target_')
bandNames = params['bandNames']
refBandLoc = np.where(bandNames_TRN == params['training_referenceBand'])[0]
refBandNorm = norms[bandNames.index(params['training_referenceBand'])]

loc = -1
for it, sed_name in enumerate(params['templates_names']):
    for num in range(5):
        loc += 1
        z = np.random.uniform(low=redshiftGrid[0], high=redshiftGrid[-1])
        Y = np.zeros((bandIndices_TRN.size, 1))
        Yvar = np.zeros((bandIndices_TRN.size, 1))
        X = np.ones((bandIndices_TRN.size, 3))
        for i, iband in enumerate(bandIndices_TRN):
            trueFlux = f_mod[it, iband](z)
            noise = trueFlux * noiseLevel
            Y[i, 0] = trueFlux + noise * np.random.randn()
            Yvar[i, 0] = noise**2.

        refFlux = Y[refBandLoc, 0]
        ell = 1.0  # refFlux*refBandNorm*DL(z)**2.*params['fluxLuminosityNorm']
        for i, iband in enumerate(bandIndices_TRN):
            X[i, 0] = iband
            X[i, 1] = z
            X[i, 2] = ell

        gp.setData(X, Y, Yvar)

        ell = 1.0
        gp.X[:, 2] = 1.0
        gp.setData(gp.X, gp.Y, gp.Yvar)

        model_mean, model_var\
            = gp.predictAndInterpolate(redshiftGrid, ell=ell, z=z)
        model_sig = np.sqrt(model_var)

        numBandsMax = bandIndices_TRN.size\
            if bandIndices_TRN.size > bandIndices_TAR.size\
            else bandIndices_TAR.size
        fig, axs = plt.subplots(2, numBandsMax, figsize=(2.7*numBandsMax, 6),
                                sharex=False, sharey=True)
        for i, inm in enumerate(bandNames_TRN):
            axs[0, i].errorbar(z, Y[i, 0] * z**2,
                               np.sqrt(Yvar[i, 0]) * z**2,
                               fmt='-o', markersize=5)

        fac = ell * redshiftGrid**2
        ylims = [0.25*np.min(model_mean[2:-2, :]*fac[2:-2, None]),
                 2*np.max(model_mean[2:-2, :]*fac[2:-2, None])]
        fullarr = np.concatenate(([f_mod[it, ib](redshiftGrid[2:-2])*fac[2:-2]
                                   for ib in bandIndices_TAR]))
        ylims = [0.25*np.min(fullarr), 2*np.max(fullarr)]
        for i, (ib, inm) in enumerate(zip(bandIndices_TRN, bandNames_TRN)):
            axs[0, i].axvspan(z-0.5, z+0.5, color='gray', alpha=0.1)
            axs[0, i].axvline(z, ls='dashed', c='k')
            axs[0, i].fill_between(redshiftGrid,
                                   (model_mean[:, ib] - model_sig[:, ib])*fac,
                                   (model_mean[:, ib] + model_sig[:, ib])*fac,
                                   color='b', alpha=0.2)
            axs[0, i].plot(redshiftGrid, model_mean[:, ib]*fac, c='b')
            axs[0, i].plot(redshiftGrid, f_mod[it, ib](redshiftGrid)*fac,
                           c='k')
            axs[0, i].set_title(inm)
            axs[0, i].set_yscale('log')
            axs[0, i].set_ylim(ylims)
            axs[0, i].set_xlabel('Redshift')
            axs[0, i].set_xlim([redshiftGrid[0], redshiftGrid[-1]])

        for i, (ib, inm) in enumerate(zip(bandIndices_TAR, bandNames_TAR)):
            axs[1, i].axvspan(z-0.5, z+0.5, color='gray', alpha=0.1)
            axs[1, i].axvline(z, ls='dashed', c='k')
            axs[1, i].fill_between(redshiftGrid,
                                   (model_mean[:, ib] - model_sig[:, ib])*fac,
                                   (model_mean[:, ib] + model_sig[:, ib])*fac,
                                   color='b', alpha=0.2)
            axs[1, i].plot(redshiftGrid, model_mean[:, ib]*fac, c='b')
            axs[1, i].plot(redshiftGrid, f_mod[it, ib](redshiftGrid)*fac,
                           c='k')
            axs[1, i].set_title(inm)
            axs[1, i].set_yscale('log')
            axs[1, i].set_ylim(ylims)
            axs[1, i].set_xlim([redshiftGrid[0], redshiftGrid[-1]])
            axs[1, i].set_xlabel('Redshift')

        axs[0, 0].set_ylabel(r'Measured fluxes ($\times z^2$)')
        axs[1, 0].set_ylabel(r'Predicted fluxes ($\times z^2$)')
        fig.tight_layout()
        fig.savefig('data/sim-fluxes-'+str(loc)+'.pdf')
