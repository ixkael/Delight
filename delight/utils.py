
import numpy as np
import ConfigParser
import os

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
    config = ConfigParser.ConfigParser()
    if not os.path.isfile(fileName):
        raise Exception(fileName+' : file not found')
    config.read(fileName)
    config.sections()

    for secName in ['Bands', 'Training', 'Target', 'Other']:
        if not config.has_section(secName):
            raise Exception(secName+' not found in parameter file')

    params = {}

    params['rootDir'] = config.get('Other', 'rootDir')
    if not os.path.isdir(params['rootDir']):
        raise Exception(params['rootDir']+' is not a valid directory')

    # Parsing Bands
    params['bandNames'] = config.get('Bands', 'Names').split(' ')
    for band in params['bandNames']:
        fname = config.get('Bands', band)
        if not os.path.isfile(fname):
            raise Exception(fname+' : file does not exist')
        params[band+'_file'] = fname

    # Parsing Training
    params['training_file'] = config.get('Training', 'file')
    if not os.path.isfile(params['training_file']):
        raise Exception(params['training_file']+' : file does not exist')
    params['referenceBand'] = config.get('Training', 'referenceBand')
    if params['referenceBand'] not in params['bandNames']:
        raise Exception(params['referenceBand']+' : is not a valid band name')
    params['training_bandOrder']\
        = config.get('Training', 'bandOrder').split(' ')
    for band in params['training_bandOrder']:
        if (band not in params['bandNames'])\
                and (band[:-4] not in params['bandNames'])\
                and (band != 'redshift'):
            raise Exception(band+' does not exist')
    if 'redshift' not in params['training_bandOrder']:
        raise Exception('redshift should be included in training')

    # Parsing Target
    params['target_file'] = config.get('Target', 'file')
    if not os.path.isfile(params['target_file']):
        raise Exception(params['target_file']+' : file does not exist')
    params['target_bandOrder']\
        = config.get('Target', 'bandOrder').split(' ')
    for band in params['target_bandOrder']:
        if (band not in params['bandNames'])\
                and (band[:-4] not in params['bandNames'])\
                and (band != 'redshift'):
            raise Exception(band+' does not exist')
    params['compressFile'] = config.get('Target', 'compressFile')
    if not os.path.isfile(params['compressFile']):
        raise Exception(params['compressFile']+' : file does not exist')
    params['Ncompress'] = config.getint('Target', 'Ncompress')

    # Parsing other parameters
    params['alpha_C'] = config.getfloat('Other', 'alpha_C')
    params['alpha_L'] = config.getfloat('Other', 'alpha_L')
    params['V_C'] = config.getfloat('Other', 'V_C')
    params['V_L'] = config.getfloat('Other', 'V_L')
    params['redshiftMax'] = config.getfloat('Other', 'redshiftMax')
    params['redshiftBinSize'] = config.getfloat('Other', 'redshiftBinSize')
    params['redshiftDistributionBinSize']\
        = config.getfloat('Other', 'redshiftDistributionBinSize')
    params['confidenceLevels']\
        = [float(x) for x in
           config.get('Other', 'confidenceLevels').split(' ')]

    return params


def parseBandCoefficientsFile(fileName):
    pass
