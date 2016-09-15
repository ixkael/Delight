
import numpy as np


class approx_DL():
    """
    Approximate luminosity_distance relation,
    agrees with astropy.FlatLambdaCDM(H0=70, Om0=0.3, Ob0=None) better than 1%
    """
    def __call__(self, z):
        return np.exp(30.5 * z**0.04 - 21.7)

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def random_X_bzl(size, numBands=5, redshiftMax=3.0):
    """Create random (but reasonable) input space for photo-z GP """
    X = np.zeros((size, 3))
    X[:, 0] = np.random.randint(low=0, high=numBands-1, size=size)
    X[:, 1] = np.random.uniform(low=0.1, high=redshiftMax, size=size)
    X[:, 2] = np.random.uniform(low=0.5, high=10.0, size=size)
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


def CIlevel(redshiftGrid, PDF, fraction, numlevels=100):
    evidence = np.trapz(PDF, redshiftGrid)
    for level in np.linspace(0, PDF.max(), num=numlevels):
        ind = np.where(PDF <= level)
        resint = np.trapz(PDF[ind], redshiftGrid[ind])
        if resint >= fraction*evidence:
            return level


def computeMetrics(ztrue, redshiftGrid, PDF, confIntervals):
    zmean = np.average(redshiftGrid, weights=PDF)
    zmap = redshiftGrid[np.argmax(PDF)]
    pdfAtZ = np.interp(ztrue, redshiftGrid, PDF)
    cumPdfAtZ = np.interp(ztrue, redshiftGrid, PDF.cumsum())
    confidencelevels = [
        CIlevel(redshiftGrid, PDF, 1.0 - confI) for confI in confIntervals
    ]
    return [ztrue, zmean, zmap, pdfAtZ, cumPdfAtZ] + confidencelevels
