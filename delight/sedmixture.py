# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline, UnivariateSpline
from delight.utils import approx_DL
# from specutils import extinction
from astropy import units as u


class PhotometricFilter:
    """Photometric filter response"""
    def __init__(self, bandName, tabulatedWavelength, tabulatedResponse):
        self.bandName = bandName
        self.wavelengthGrid = tabulatedWavelength
        self.tabulatedResponse = tabulatedResponse
        self.interp = interp1d(tabulatedWavelength, tabulatedResponse)
        self.norm = np.trapz(tabulatedResponse/tabulatedWavelength,
                             x=tabulatedWavelength)
        ind = np.where(
            tabulatedResponse > 0.001*np.max(tabulatedResponse)
                            )[0]
        self.lambdaMin = tabulatedWavelength[ind[0]]
        self.lambdaMax = tabulatedWavelength[ind[-1]]

    def __call__(self, wavelength):
        return self.interp(wavelength)


# class DustModel:
#     """
#     Extinction model from Cardelli, Clayton & Mathis (1988)
#     """
#     def __init__(self):
#         self.r_v = 3.1
#
#     def __call__(self, wave, a_v):
#         return extinction.extinction_d03(wave * u.Angstrom,
#                                          a_v, r_v=self.r_v)
#
#
# class SpectralTemplate_zd:
#     """
#     SED template, tabulated, to be interpolated  on aredshift and dust grid
#     """
#     def __init__(self,
#                  tabulatedWavelength, tabulatedSpectrum, photometricBands,
#                  redshiftGrid=None, dustGrid=None):
#         self.DL = approx_DL()
#         self.DustModel = DustModel()
#         self.photometricBands = photometricBands
#         self.numBands = len(photometricBands)
#         self.fbinterps = {}
#         self.sed_interp = interp1d(tabulatedWavelength, tabulatedSpectrum)
#         if redshiftGrid is None:
#             self.redshiftGrid = np.logspace(np.log10(1e-2),
#                                             np.log10(2.0),
#                                             50)
#         else:
#             self.redshiftGrid = redshiftGrid
#         if dustGrid is None:
#             self.dustGrid = np.logspace(np.log10(1e-2),
#                                         np.log10(100),
#                                         15)
#         else:
#             self.dustGrid = dustGrid
#
#         for filt in photometricBands:
#             fmodgrid = np.zeros((self.redshiftGrid.size, self.dustGrid.size))
#             for iz in range(self.redshiftGrid.size):
#                 opz = (self.redshiftGrid[iz] + 1)
#                 xf_z = filt.wavelengthGrid / opz
#                 yf_z = filt.tabulatedResponse
#                 ysed = self.sed_interp(xf_z)
#                 facz = opz**2. / (4*np.pi*self.DL(self.redshiftGrid[iz])**2.)
#                 for jd in range(self.dustGrid.size):
#                     ysedext = facz * ysed *\
#                         10**-0.4*self.DustModel(xf_z, self.dustGrid[jd])
#                     fmodgrid[iz, jd] =\
#                         np.trapz(ysedext * yf_z, x=xf_z) / filt.norm
#             self.fbinterps[filt.bandName] = RectBivariateSpline(
#                 self.redshiftGrid, self.dustGrid, fmodgrid)
#
#     def photometricFlux(self, redshifts, dusts, bandName, grid=False):
#         return self.fbinterps[bandName](redshifts, dusts, grid=grid).T
#
#     def flux(self, redshift, dust, wave):
#         opz = 1. + redshift
#         xf_z = wave / opz
#         facz = opz**2. / (4*np.pi*self.DL(redshift)**2.)
#         ysed = self.sed_interp(xf_z)
#         ysedext = facz * ysed *\
#             10**-0.4*self.DustModel(xf_z, dust)
#         return ysedext


class SpectralTemplate_z:
    """
    SED template, tabulated and to be interpolated  on a redshift grid
    """
    def __init__(self,
                 tabulatedWavelength, tabulatedSpectrum, photometricBands,
                 redshiftGrid=None, order=15):
        self.DL = approx_DL()
        self.photometricBands = photometricBands
        self.numBands = len(photometricBands)
        self.sed_interp = interp1d(tabulatedWavelength, tabulatedSpectrum,
                                   bounds_error=False,
                                   fill_value="extrapolate")
        if redshiftGrid is None:
            self.redshiftGrid = np.logspace(np.log10(1e-2),
                                            np.log10(2.0),
                                            350)
        else:
            self.redshiftGrid = redshiftGrid

        self.fbcoefs = {}
        self.fbinterps = {}
        self.logfbinterps = {}
        self.order = order
        self.fmodgrid = np.zeros((self.redshiftGrid.size,
                                  len(photometricBands)))
        self.bandNames = []
        for ib, filt in enumerate(photometricBands):
            self.bandNames.append(filt.bandName)
            for iz in range(self.redshiftGrid.size):
                opz = (self.redshiftGrid[iz] + 1)
                xf_z = filt.wavelengthGrid / opz
                yf_z = filt.tabulatedResponse
                ysed = self.sed_interp(xf_z)
                facz = opz**2. / (4*np.pi*self.DL(self.redshiftGrid[iz])**2.)
                ysedext = facz * ysed
                self.fmodgrid[iz, ib] =\
                    np.trapz(ysedext * yf_z, x=xf_z) / filt.norm
            self.fbinterps[filt.bandName] = UnivariateSpline(
                self.redshiftGrid, self.fmodgrid[:, ib], s=0)
            self.fbcoefs[filt.bandName] = np.polyfit(
                self.redshiftGrid, np.log(self.fmodgrid[:, ib]), self.order-1)
            self.logfbinterps[filt.bandName] =\
                np.poly1d(self.fbcoefs[filt.bandName])

    def photometricFlux_spline(self, redshifts, bandName):
        return self.fbinterps[bandName](redshifts)

    def photometricFlux(self, redshifts, bandName):
        return np.exp(self.logfbinterps[bandName](redshifts))

    def photometricFlux_bis(self, redshifts, bandName):
        xgg = redshifts[:, None] ** np.arange(self.order-1, -1, -1)[None, :]
        return np.exp(np.sum(xgg * self.fbcoefs[bandName][None, :], axis=1))

    def photometricFlux_gradz(self, redshifts, bandName):
        mod_der = np.poly1d(np.polyder(self.fbcoefs[bandName]))
        return mod_der(redshifts) * self.photometricFlux(redshifts, bandName)

    def photometricFlux_gradz_bis(self, redshifts, bandName):
        xgg = redshifts[:, None] ** np.arange(self.order-2, -1, -1)[None, :]
        der = np.arange(self.order-1, 0, -1)
        flux = self.photometricFlux_bis(redshifts, bandName)
        return np.sum(xgg * der * self.fbcoefs[bandName][None, :-1],
                      axis=1) * flux

    def flux(self, redshift, wave):
        opz = 1. + redshift
        xf_z = wave / opz
        facz = opz**2. / (4*np.pi*self.DL(redshift)**2.)
        ysed = self.sed_interp(xf_z)
        ysedext = facz * ysed
        return ysedext
