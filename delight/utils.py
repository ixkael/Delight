
import numpy as np
import configparser
import os
import collections


class approx_DL():
    """
    Approximate luminosity_distance relation,
    agrees with astropy.FlatLambdaCDM(H0=70, Om0=0.3, Ob0=None) better than 1%
    """
    def __call__(self, z):
        return 30.5 * z**0.04 - 21.7

    def derivative(self, z):
        return 1.22 / z**0.96

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def random_X_bzlt(size, numBands=5, redshiftMax=3.0):
    """Create random (but reasonable) input space for photo-z GP """
    X = np.zeros((size, 4))
    X[:, 0] = np.random.randint(low=0, high=numBands-1, size=size)
    X[:, 1] = np.random.uniform(low=0.1, high=redshiftMax, size=size)
    X[:, 2] = np.random.uniform(low=0.5, high=10.0, size=size)
    X[:, 3] = np.random.uniform(low=0.1, high=0.9, size=size)
    return X


def random_filtercoefs(numBands, numCoefs):
    """Create random (but reasonable) coefficients describing
    numBands photometric filters as sum of gaussians"""
    fcoefs_amp\
        = np.random.uniform(low=0., high=1., size=numBands*numCoefs)\
        .reshape((numBands, numCoefs))
    fcoefs_mu\
        = np.random.uniform(low=3e3, high=1e4, size=numBands*numCoefs)\
        .reshape((numBands, numCoefs))
    fcoefs_sig\
        = np.random.uniform(low=30, high=500, size=numBands*numCoefs)\
        .reshape((numBands, numCoefs))
    return fcoefs_amp, fcoefs_mu, fcoefs_sig


def random_linecoefs(numLines):
    """Create random (but reasonable) coefficients describing lines in SEDs"""
    lines_mu = np.random.uniform(low=1e3, high=1e4, size=numLines)
    lines_sig = np.random.uniform(low=5, high=50, size=numLines)
    return lines_mu, lines_sig


def random_hyperparams():
    """Create random (but reasonable) hyperparameters for photo-z GP"""
    alpha_T, var_C, var_L = np.random.uniform(low=0.5, high=2.0, size=3)
    alpha_C, alpha_L = np.random.uniform(low=10.0, high=1000.0, size=2)
    return var_C, var_L, alpha_C, alpha_L, alpha_T


def parseParamFile(fileName):
    """
    Parser for cfg input parameter files,
    see examples for details.
    """
    config = configparser.ConfigParser()
    if not os.path.isfile(fileName):
        raise Exception(fileName+' : file not found')
    config.read(fileName)
    config.sections()

    for secName in ['Bands', 'Training', 'Target', 'Other']:
        if not config.has_section(secName):
            raise Exception(secName+' not found in parameter file')

    params = collections.OrderedDict()

    params['rootDir'] = config.get('Other', 'rootDir')
    if not os.path.isdir(params['rootDir']):
        raise Exception(params['rootDir']+' is not a valid directory')

    # Parsing Bands
    params['bandNames'] = config.get('Bands', 'Names').split(' ')
    for band in params['bandNames']:
        fname = config.get('Bands', band)
        if not os.path.isfile(fname):
            raise Exception(fname+' : file does not exist')
        params['bandFile_'+band] = fname

    # Parsing Training
    params['training_catFile'] = config.get('Training', 'catFile')
    params['training_numChunks'] = config.getint('Training', 'numChunks')
    params['training_paramFile'] = config.get('Training', 'paramFile')
    if not os.path.isfile(params['training_catFile']):
        raise Exception(params['training_catFile']+' : file does not exist')
    params['referenceBand'] = config.get('Training', 'referenceBand')
    if params['referenceBand'] not in params['bandNames']:
        raise Exception(params['referenceBand']+' : is not a valid band name')
    params['training_bandOrder']\
        = config.get('Training', 'bandOrder').split(' ')
    for band in params['training_bandOrder']:
        if (band not in params['bandNames'])\
                and (band[:-4] not in params['bandNames'])\
                and (band != '_')\
                and (band != 'redshift'):
            raise Exception(band+' does not exist')
    if 'redshift' not in params['training_bandOrder']:
        raise Exception('redshift should be included in training')

    # Parsing Target
    params['target_catFile'] = config.get('Target', 'catFile')
    if not os.path.isfile(params['target_catFile']):
        raise Exception(params['target_catFile']+' : file does not exist')
    params['target_bandOrder']\
        = config.get('Target', 'bandOrder').split(' ')
    for band in params['target_bandOrder']:
        if (band not in params['bandNames'])\
                and (band[:-4] not in params['bandNames'])\
                and (band != '_')\
                and (band != 'redshift'):
            raise Exception(band+' does not exist')
    params['compressIndicesFile'] = config.get('Target', 'compressIndicesFile')
    params['compressMargLikFile'] = config.get('Target', 'compressMargLikFile')
    params['Ncompress'] = config.getint('Target', 'Ncompress')
    params['useCompression'] = config.getboolean("Target", 'useCompression')
    params['redshiftpdfFile'] = config.get('Target', 'redshiftpdfFile')

    # Parsing other parameters
    params['alpha_C'] = config.getfloat('Other', 'alpha_C')
    params['alpha_L'] = config.getfloat('Other', 'alpha_L')
    params['V_C'] = config.getfloat('Other', 'V_C')
    params['V_L'] = config.getfloat('Other', 'V_L')
    params['redshiftMax'] = config.getfloat('Other', 'redshiftMax')
    params['redshiftBinSize'] = config.getfloat('Other', 'redshiftBinSize')
    params['redshiftDisBinSize']\
        = config.getfloat('Other', 'redshiftDisBinSize')
    params['lines_pos']\
        = [float(x) for x in
           config.get('Other', 'lines_pos').split(' ')]
    params['lines_width']\
        = [float(x) for x in
           config.get('Other', 'lines_width').split(' ')]
    params['confidenceLevels']\
        = [float(x) for x in
           config.get('Other', 'confidenceLevels').split(' ')]

    return params


def readColumnPositions(params, pfx="training_"):
    bandIndices = np.array([ib for ib, b in enumerate(params['bandNames'])
                            if b in params[pfx+'bandOrder']])
    bandNames = np.array(params['bandNames'])[bandIndices]
    bandColumns = np.array([params[pfx+'bandOrder'].index(b)
                            for b in bandNames])
    bandVarColumns = np.array([params[pfx+'bandOrder'].index(b+'_var')
                               for b in bandNames])
    redshiftColumn = params[pfx+'bandOrder'].index('redshift')
    refBandColumn = params[pfx+'bandOrder'].index(params['referenceBand'])
    return bandIndices, bandNames, bandColumns, bandVarColumns,\
        redshiftColumn, refBandColumn


def readBandCoefficients(params):
    bandCoefAmplitudes =\
        np.vstack([np.loadtxt(params['bandFile_'+band])[:, 0]
                   for band in params['bandNames']])
    bandCoefPositions =\
        np.vstack([np.loadtxt(params['bandFile_'+band])[:, 1]
                   for band in params['bandNames']])
    bandCoefWidths =\
        np.vstack([np.loadtxt(params['bandFile_'+band])[:, 2]
                   for band in params['bandNames']])
    norms =\
        np.sqrt(2*np.pi) * np.sum(bandCoefAmplitudes * bandCoefWidths, axis=1)
    return bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, norms


def scalefree_flux_likelihood(f_obs, f_obs_var, f_mod, f_mod_var=None):
    nz, nt, nf = f_mod.shape
    if f_mod_var is not None:
        var = 1./(1./f_obs_var[None, None, :] + 1./f_mod_var)
    else:
        var = f_obs_var  # nz * nt * nf
    invvar = np.where(f_obs/var < 1e-6, 0.0, var**-1.0)  # nz * nt * nf
    FOT = np.sum(f_mod * f_obs * invvar, axis=2)  # nz * nt
    FTT = np.sum(f_mod**2 * invvar, axis=2)  # nz * nt
    FOO = np.dot(invvar, f_obs**2)  # nz * nt
    chi2 = FOO - FOT**2.0 / FTT  # nz * nt
    like = np.exp(-0.5*chi2) / np.sqrt(FTT)  # nz * nt
    return like
