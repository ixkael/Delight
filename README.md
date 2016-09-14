# Delight
**Photometric redshift via Gaussian processes with physical kernels.**

*Warning: this code is still in active development and is not quite ready to be blindly applied to arbitrary photometric galaxy surveys. But this day will come.*

![alt tag](https://travis-ci.org/ixkael/Delight.svg?branch=master)
![alt tag](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)
[![Coverage Status](https://coveralls.io/repos/github/ixkael/Delight/badge.svg?branch=master)](https://coveralls.io/github/ixkael/Delight?branch=master)

**Tests**: pytest for unit tests, PEP8 for code style, coveralls for test coverage.

## Content

**./paper/**: journal paper describing the method </br>
**./delight/**: main code (Python/Cython) </br>
**./tests/**: test suite for the main code </br>
**./notebooks/**: demo notebooks using delight </br>
**./data/**: some useful inputs for tests/demos </br>
**./other/**: useful mathematica notebooks, etc </br>

## Requirements

python, cython, numpy, scipy, pytest, pylint, coveralls, matplotlib, astropy </br>

## Authors

Boris Leistedt (NYU) </br>
David W. Hogg (NYU) (MPIA) (SCDA)

## License

Copyright 2016 the authors. The code in this repository is released under the open-source MIT License. See the file LICENSE for more details.
