# -*- coding: utf-8 -*-

from delight.io import *

paramFile = "tests/parametersTest.cfg"


def test_Parser():
    params = parseParamFile(paramFile, verbose=False)


def test_createGrids():
    params = parseParamFile(paramFile, verbose=False)
    out = createGrids(params)


def test_readBandCoefficients():
    params = parseParamFile(paramFile, verbose=False)
    out = readBandCoefficients(params)


def test_readColumnPositions():
    params = parseParamFile(paramFile, verbose=False)
    out = readColumnPositions(params)
