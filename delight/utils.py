
import numpy as np

def random_X(size, numTypes=8, numBands=5, redshiftMax=3.0):
    """Create random (but reasonable) input space for photo-z GP """
    X = np.zeros((size, 3))
    X[:,0] = np.random.uniform(low=0, high=numTypes-1, size=size) / float(numTypes)
    X[:,1] = np.random.randint(low=0, high=numBands-1, size=size)
    X[:,2] = np.random.uniform(low=0, high=redshiftMax, size=size)
    return X

def random_filtercoefs(numBands, numCoefs):
    """Create random (but reasonable) coefficients describing numBands photometric filters as sum of gaussians"""
    fcoefs_amp = np.random.uniform(low=0.5, high=1, size=numBands*numCoefs).reshape((numBands, numCoefs))
    fcoefs_mu = np.random.uniform(low=2e3, high=8e3, size=numBands*numCoefs).reshape((numBands, numCoefs))
    fcoefs_sig = np.random.uniform(low=100, high=500, size=numBands*numCoefs).reshape((numBands, numCoefs))
    return fcoefs_amp, fcoefs_mu, fcoefs_sig

def random_linecoefs(numLines):
    """Create random (but reasonable) coefficients describing lines in SEDs"""
    lines_mu = np.random.uniform(low=1e3, high=1e4, size=numLines)
    lines_sig = np.random.uniform(low=5, high=50, size=numLines)
    return lines_mu, lines_sig

def random_hyperparams():
    """Create random (but reasonable) hyperparameters for photo-z GP"""
    alpha_T, var_T = np.random.uniform(low=0.2, high=2.0, size=2)
    alpha_C, alpha_L = np.random.uniform(low=10.0, high=1000.0, size=2)
    return var_T, alpha_C, alpha_L, alpha_T
