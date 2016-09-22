
import sys
import numpy as np
import itertools
from delight.io import *
from delight.utils import *
from delight.photoz_gp import PhotozGP
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Parse parameters file
if len(sys.argv) < 2:
    raise Exception('Please provide a parameter file')
params = parseParamFile(sys.argv[1], verbose=False)

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
bandIndices_TRN, bandNames_TRN, bandColumns_TRN,\
    bandVarColumns_TRN, redshiftColumn_TRN,\
    refBandColumn_TRN = readColumnPositions(params, prefix='training_')
bandIndices_TAR, bandNames_TAR, bandColumns_TAR,\
    bandVarColumns_TAR, redshiftColumn_TAR,\
    refBandColumn_TAR = readColumnPositions(params, prefix='target_')
bandNames = params['bandNames']

newRedshiftMax = 1.0

loc = - 1
trainingDataIter = getDataFromFile(params, 0, numObjectsTraining,
                                   prefix="training_", getXY=True)
targetDataIter = getDataFromFile(params, 0, numObjectsTraining,
                                 prefix="target_", getXY=False)
for z, ell, bands, fluxes, fluxesVar, X, Y, Yvar in trainingDataIter:
    loc += 1

    gp.setData(X, Y, Yvar)
    alpha, ell = gp.estimateAlphaEll()

    wavs = np.linspace(bandCoefPositions.min(),
                       bandCoefPositions.max(), num=300)
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for mu in params['lines_pos']:
        axs[0].axvline(mu*(1+z), c='gray', ls='dashed')
    filterMidWav = 0*bandIndices_TRN
    filterStdWav = 0*bandIndices_TRN
    for i, ib in enumerate(bandIndices_TRN):
        y = 0*wavs
        for amp, mu, sig in zip(bandCoefAmplitudes[ib, :],
                                bandCoefPositions[ib, :],
                                bandCoefWidths[ib, :]):
            y += amp * np.exp(-0.5*((wavs-mu)/sig)**2)
        filterMidWav[i] = np.average(wavs, weights=y)
        filterStdWav[i] = np.sqrt(
            np.average((wavs-filterMidWav[i])**2, weights=y))
        axs[1].plot(wavs, y, c='k')
        axs[0].errorbar(filterMidWav[i], Y[i, 0],
                        yerr=np.sqrt(Yvar[i, 0]), xerr=1.5*filterStdWav[i],
                        fmt='-o', markersize=5, color='k', lw=2)
    sed, fac, cov, filters = gp.drawSED(z, ell, wavs)
    sed = np.interp(wavs, filterMidWav[:], Y[:, 0])
    sedfluxes = np.zeros((bandIndices_TRN.size, ))
    for i, ib in enumerate(bandIndices_TRN):
        sedfluxes[i] = np.trapz(filters[ib]*sed, x=wavs) /\
            np.trapz(filters[ib], x=wavs)
    lp = np.sum(-0.5*(sedfluxes - fluxes)**2/fluxesVar)
    numsamples = 2000
    seds = np.zeros((numsamples, wavs.size))
    off = 0
    for i in range(numsamples):
        sed_p = 1*sed + fac * np.random.multivariate_normal(0*wavs, cov/10**2)
        for i, ib in enumerate(bandIndices_TRN):
            sedfluxes[i] = np.trapz(filters[ib]*sed_p, x=wavs) /\
                np.trapz(filters[ib], x=wavs)
        lp_prime = np.sum(-0.5*(sedfluxes - fluxes)**2/fluxesVar)
        if np.random.rand() <= np.exp(lp_prime - lp):
            sed = 1*sed_p
            seds[off, :] = sed_p
            off += 1
            lp = 1*lp_prime
    print("Number of accepted samples:", off)
    sedmean, sedstd = seds[:off, :].mean(axis=0), seds[:off, :].std(axis=0)
    axs[0].plot(wavs, sedmean, c='b')
    axs[0].fill_between(wavs, sedmean+sedstd, sedmean-sedstd,
                        color='b', alpha=0.2)
    for i in np.random.choice(off, 2, replace=False):
        axs[0].plot(wavs, seds[i, :], c='k', alpha=0.3)
    axs[0].set_ylabel('Flux')
    axs[1].set_ylabel('Filters')
    axs[1].set_xlabel('Wavelength')
    #axs[0].set_yscale('log')
    axs[1].set_xlim([wavs[0], wavs[-1]])
    axs[1].set_ylim([0, 1.1*np.max(filters)])
    axs[1].set_yticks([])
    fig.tight_layout()
    fig.savefig('data/data-sed-'+str(loc)+'.png')

    model_mean, model_var\
        = gp.predictAndInterpolate(redshiftGrid, ell=ell, z=z)
    model_sig = np.sqrt(model_var)
    z_TAR, ell_TAR, bands_TAR, fluxes_TAR, fluxesVar_TAR = next(targetDataIter)
    fluxes_TAR /= ell_TAR
    fluxesVar_TAR /= ell_TAR**2
    like_grid = scalefree_flux_likelihood(
        fluxes_TAR, fluxesVar_TAR,
        model_mean[:, None, bands_TAR],  # model mean
        f_mod_var=model_var[:, None, bands_TAR]  # model var
    ).sum(axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(redshiftGrid, like_grid, c='k')
    ax.set_ylabel('Likelihood')
    ax.set_xlabel('Redshift')
    ax.axvline(z_TAR, c='r', ls='dashed')
    ax.set_xlim([0, newRedshiftMax])
    ax.set_ylim([0, 1.1*np.max(like_grid)])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    fig.tight_layout()
    fig.savefig('data/data-pdfs-'+str(loc)+'.png')

    numBandsMax = bandIndices_TRN.size\
        if bandIndices_TRN.size > bandIndices_TAR.size\
        else bandIndices_TAR.size

    fig, axs = plt.subplots(2, numBandsMax, figsize=(2.2*numBandsMax, 5),
                            sharex=True, sharey=True)
    for i, inm in enumerate(bandNames_TRN):
        axs[0, i].errorbar(z, Y[i, 0]/ell,
                           np.sqrt(Yvar[i, 0])/ell, fmt='-o', markersize=5)
    for i, ib in enumerate(bands_TAR):
        axs[1, i].axvline(z_TAR, c='r', ls='dashed')
        axs[1, i].axhline(fluxes_TAR[i], c='r')
        axs[1, i].axhspan(fluxes_TAR[i] - np.sqrt(fluxesVar_TAR[i]),
                          fluxes_TAR[i] + np.sqrt(fluxesVar_TAR[i]),
                          color='r', alpha=0.2)

    ylims = [0.25*np.min(model_mean[2:-2, :]),
             2*np.max(model_mean[2:-2, :])]
    for i, (ib, inm) in enumerate(zip(bandIndices_TRN, bandNames_TRN)):
        axs[0, i].axvline(z, ls='dashed', c='k')
        axs[0, i].fill_between(redshiftGrid,
                               (model_mean[:, ib] - model_sig[:, ib]),
                               (model_mean[:, ib] + model_sig[:, ib]),
                               color='b', alpha=0.2)
        axs[0, i].plot(redshiftGrid, model_mean[:, ib], c='b',
                       label='alpha = %.2g' % gp.mean_fct.alpha)
        axs[0, i].set_title(inm)
        axs[0, i].set_yscale('log')
        axs[0, i].set_ylim(ylims)
        axs[0, i].set_xlim([redshiftGrid[0], newRedshiftMax])

    for i, (ib, inm) in enumerate(zip(bandIndices_TAR, bandNames_TAR)):
        axs[1, i].axvline(z, ls='dashed', c='k')
        axs[1, i].fill_between(redshiftGrid,
                               (model_mean[:, ib] - model_sig[:, ib]),
                               (model_mean[:, ib] + model_sig[:, ib]),
                               color='b', alpha=0.2)
        axs[1, i].plot(redshiftGrid, model_mean[:, ib], c='b',
                       label='alpha = %.2g' % gp.mean_fct.alpha)
        axs[1, i].set_title(inm)
        axs[1, i].set_yscale('log')
        axs[1, i].set_ylim(ylims)
        axs[1, i].set_xlim([redshiftGrid[0], newRedshiftMax])
        axs[1, i].set_xlabel('Redshift')

    axs[0, 0].set_ylabel('Measured fluxes')
    axs[1, 0].set_ylabel('Predicted fluxes')
    fig.tight_layout()
    fig.savefig('data/data-fluxes-'+str(loc)+'.png')

    if loc > 20:
        exit(1)
