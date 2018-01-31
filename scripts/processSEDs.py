import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


from delight.io import *
from delight.utils import *

if len(sys.argv) < 2:
    raise Exception('Please provide a parameter file')
params = parseParamFile(sys.argv[1], verbose=False, catFilesNeeded=False)
bandNames = params['bandNames']
dir_seds = params['templates_directory']
dir_filters = params['bands_directory']
lambdaRef = params['lambdaRef']
sed_names = params['templates_names']
fmt = '.dat'
DL = approx_DL()
redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)
numZ = redshiftGrid.size

# Loop over SEDs
for sed_name in sed_names:
    seddata = np.genfromtxt(dir_seds + '/' + sed_name + fmt)
    seddata[:, 1] *= seddata[:, 0]
    ref = np.interp(lambdaRef, seddata[:, 0], seddata[:, 1])
    seddata[:, 1] /= ref
    sed_interp = interp1d(seddata[:, 0], seddata[:, 1])

    f_mod = np.zeros((redshiftGrid.size, len(bandNames)))
    # Loop over bands
    for jf, band in enumerate(bandNames):
        fname_in = dir_filters + '/' + band + '.res'
        data = np.genfromtxt(fname_in)
        xf, yf = data[:, 0], data[:, 1]
        #yf /= xf  # divide by lambda
        # Only consider range where >1% max
        ind = np.where(yf > 0.01*np.max(yf))[0]
        lambdaMin, lambdaMax = xf[ind[0]], xf[ind[-1]]
        norm = np.trapz(yf/xf, x=xf)

        for iz in range(redshiftGrid.size):
            opz = (redshiftGrid[iz] + 1)
            xf_z = np.linspace(lambdaMin / opz, lambdaMax / opz, num=5000)
            yf_z = interp1d(xf / opz, yf)(xf_z)
            ysed = sed_interp(xf_z)
            f_mod[iz, jf] = np.trapz(ysed * yf_z, x=xf_z) / norm
            f_mod[iz, jf] *= opz**2. / DL(redshiftGrid[iz])**2. / (4*np.pi)

    np.savetxt(dir_seds + '/' + sed_name + '_fluxredshiftmod.txt', f_mod)
