import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import leastsq

numCoefs = 12  # number of components for the fit
bandNames = ['u', 'g', 'r', 'i', 'z']  # Bands
fmt = '.res'
max_redshift = 2.0  # for plotting purposes
root = './data/SDSS_FILTERS'
make_plots = False


# Function we will optimize
def dfunc(p, x, yd):
    y = 0*x
    n = p.size//2
    for i in range(n):
        y += np.abs(p[i]) * np.exp(-0.5*((mus[i]-x)/np.abs(p[n+i]))**2.0)
    return yd - y


# Loop over bands
for band in bandNames:

    fname_in = root + '/' + band + fmt
    data = np.genfromtxt(fname_in)
    coefs = np.zeros((numCoefs, 3))
    x, y = data[:, 0], data[:, 1]
    y /= x  # divide by lambda
    # Only consider range where >1% max
    ind = np.where(y > 0.01*np.max(y))[0]
    lambdaMin, lambdaMax = x[ind[0]], x[ind[-1]]
    # Initialize values for amplitude and width of the components
    sig0 = np.repeat((lambdaMax-lambdaMin)/numCoefs/4, numCoefs)
    # Components uniformly distributed in the range
    mus = np.linspace(lambdaMin+sig0[0], lambdaMax-sig0[-1], num=numCoefs)
    amp0 = interp1d(x, y)(mus)
    p0 = np.concatenate((amp0, sig0))
    popt, pcov = leastsq(dfunc, p0, args=(x, y))
    coefs[:, 0] = np.abs(popt[0:numCoefs])  # amplitudes
    coefs[:, 1] = mus  # positions
    coefs[:, 2] = np.abs(popt[numCoefs:2*numCoefs])  # widths

    fname_out = root + '/' + band + '_gaussian_coefficients.txt'
    np.savetxt(fname_out, coefs, header=fname_in)

    xf = np.linspace(lambdaMin, lambdaMax, num=1000)
    yy = 0*xf
    for i in range(numCoefs):
        yy += coefs[i, 0] * np.exp(-0.5*((coefs[i, 1] - xf)/coefs[i, 2])**2.0)

    if make_plots:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x[ind], y[ind], lw=3, label='True filter', c='k')
        ax.plot(xf, yy, lw=2, c='r', label='Gaussian fit')

    coefs_redshifted = 1*coefs
    coefs_redshifted[:, 1] /= (1. + max_redshift)
    coefs_redshifted[:, 2] /= (1. + max_redshift)
    lambdaMin_redshifted, lambdaMax_redshifted\
        = lambdaMin / (1. + max_redshift), lambdaMax / (1. + max_redshift)
    xf = np.linspace(lambdaMin_redshifted, lambdaMax_redshifted, num=1000)
    yy = 0*xf
    for i in range(numCoefs):
        yy += coefs_redshifted[i, 0] *\
            np.exp(-0.5*((coefs_redshifted[i, 1] - xf) /
                   coefs_redshifted[i, 2])**2.0)

    if make_plots:
        ax.plot(xf, yy, lw=2, c='b', label='G fit at z='+str(max_redshift))
        title = band + ' band (' + fname_in +\
            ') with %i' % numCoefs+' components'
        ax.set_title(title)
        ax.set_ylim([0, data[:, 1].max()*1.2])
        ax.set_yticks([])
        ax.set_xlabel('$\lambda$')
        ax.legend(loc='upper center', frameon=False, ncol=3)

        fig.tight_layout()
        fname_fig = root + '/' + band + '_gaussian_approximation.png'
        fig.savefig(fname_fig)
