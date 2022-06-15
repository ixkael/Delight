import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from delight.io import *
from delight.utils import *

if len(sys.argv) < 2:
    raise Exception('Please provide a parameter file')

params = parseParamFile(sys.argv[1], verbose=False, catFilesNeeded=False)

dir_seds = params['templates_directory']
sed_names = params['templates_names']
redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)
numZ = redshiftGrid.size
numT = len(sed_names)
numB = len(params['bandNames'])
numObjects = params['numObjects']
noiseLevel = params['noiseLevel']
f_mod = np.zeros((numT, numB), dtype=object)
for it, sed_name in enumerate(sed_names):
    data = np.loadtxt(dir_seds + '/' + sed_name + '_fluxredshiftmod.txt')
    for jf in range(numB):
        f_mod[it, jf] = interp1d(redshiftGrid, data[:, jf], kind='linear')

# Generate training data
redshifts = np.random.uniform(low=redshiftGrid[0],
                              high=redshiftGrid[-1],
                              size=numObjects)
types = np.random.randint(0, high=numT, size=numObjects)

ell = 1e6
fluxes, fluxesVar = np.zeros((numObjects, numB)), np.zeros((numObjects, numB))
for k in range(numObjects):
    for i in range(numB):
        trueFlux = ell * f_mod[types[k], i](redshifts[k])
        noise = trueFlux * noiseLevel
        fluxes[k, i] = trueFlux + noise * np.random.randn()
        fluxesVar[k, i] = noise**2.
data = np.zeros((numObjects, 1 + len(params['training_bandOrder'])))
bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn,\
    refBandColumn = readColumnPositions(params, prefix="training_")
for ib, pf, pfv in zip(bandIndices, bandColumns, bandVarColumns):
    data[:, pf] = fluxes[:, ib]
    data[:, pfv] = fluxesVar[:, ib]
data[:, redshiftColumn] = redshifts
data[:, -1] = types
np.savetxt(params['trainingFile'], data)

# Generate Target data
redshifts = np.random.uniform(low=redshiftGrid[0],
                              high=redshiftGrid[-1],
                              size=numObjects)
types = np.random.randint(0, high=numT, size=numObjects)
fluxes, fluxesVar = np.zeros((numObjects, numB)), np.zeros((numObjects, numB))
for k in range(numObjects):
    for i in range(numB):
        trueFlux = f_mod[types[k], i](redshifts[k])
        noise = trueFlux * noiseLevel
        fluxes[k, i] = trueFlux + noise * np.random.randn()
        fluxesVar[k, i] = noise**2.

data = np.zeros((numObjects, 1 + len(params['target_bandOrder'])))
bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn,\
    refBandColumn = readColumnPositions(params, prefix="target_")
for ib, pf, pfv in zip(bandIndices, bandColumns, bandVarColumns):
    data[:, pf] = fluxes[:, ib]
    data[:, pfv] = fluxesVar[:, ib]
data[:, redshiftColumn] = redshifts
data[:, -1] = types
np.savetxt(params['targetFile'], data)
