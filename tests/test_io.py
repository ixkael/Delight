
from delight.io import *

paramFile = "tests/parametersTest.cfg"


def test_Parser():
    params = parseParamFile(paramFile, verbose=False)


def test_createGrids():
    out = createGrids(paramFile)


def test_readBandCoefficients():
    out = readBandCoefficients(paramFile)


def test_readColumnPositions():
    out = readColumnPositions(paramFile)
