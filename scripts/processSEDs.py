import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from delight.utils import approx_DL

bandNames = ['u', 'g', 'r', 'i', 'z']  # Bands
dir_seds = './data/CWW_SEDs'
dir_filters = './data/SDSS_FILTERS'
lambdaRef = 4.5e3
sed_names = ['El_B2004a', 'Sbc_B2004a', 'Scd_B2004a',
             'SB3_B2004a', 'Im_B2004a', 'SB2_B2004a',
             'ssp_25Myr_z008', 'ssp_5Myr_z008']
redshiftGrid = np.arange(0, 3, 0.01)
DL = approx_DL()

# Loop over SEDs
for sed_name in sed_names:
    seddata = np.genfromtxt(dir_seds + '/' + sed_name + fmt)
    seddata[:, 1] *= seddata[:, 0]**2. / 3e18
    ref = np.interp(lambdaRef, seddata[:, 0], seddata[:, 1])
    seddata[:, 1] /= ref
    sed_interp = interp1d(seddata[:, 0], seddata[:, 1])

    f_mod = np.zeros((redshiftGrid.size, len(bandNames)))
    # Loop over bands
    for jf, band in enumerate(bandNames):
        fname_in = dir_filters + '/' + band + '.res'
        data = np.genfromtxt(fname_in)
        xf, yf = data[:, 0], data[:, 1]
        yf /= xf  # divide by lambda
        # Only consider range where >1% max
        ind = np.where(yf > 0.01*np.max(yf))[0]
        lambdaMin, lambdaMax = xf[ind[0]], xf[ind[-1]]
        norm = np.trapz(yf, x=xf)

        for iz in range(redshiftGrid.size):
            opz = (redshiftGrid[iz] + 1)
            xf_z = np.linspace(lambdaMin / opz, lambdaMax / opz, num=5000)
            yf_z = interp1d(xf / opz, yf)(xf_z)
            ysed = sed_interp(xf_z)
            f_mod[iz, jf] = np.trapz(ysed * yf_z, x=xf_z) / norm
            f_mod[iz, jf] *= opz**2. / DL(redshiftGrid[iz])**2. / (4*np.pi)

    np.savetxt(dir_seds + '/' + sed_name + '_fluxredshiftmod.txt', f_mod)
