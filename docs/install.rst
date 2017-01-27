
Installation
==================

Short description
----------------

There are four python modules:

- **IO**: routines for parsing the parameter file and constructing useful quantities.
- **Photoz_GP**: the main classes for constructing a GP with the photo-z kernel, and fitting photometry.
- **Photoz_kernels**: the main classes for the mean function and kernel of the photo-z GP.
- **Utils**: various utilities, e.g., for likelihood functions.

There are two additional cython modules:

- **Photoz_kernels_cy**: the code computation of the photo-z kernel.
- **Utils_cy**: a faster multi-band likelihood function.

Installation
----------------

1. Install required packages (see `requirements.txt`).

Via pip or conda, e.g.,

.. code-block:: bash

    pip install -r requirements.txt

2. Compile cython code and install module.

.. code-block:: bash

    python setup.py build_ext --inplace
    python setup.py install

4. Run the tests. Nothing should fail!

.. code-block:: bash

    python scripts/processFilters.py tests/parametersTest.cfg
    python scripts/processSEDs.py tests/parametersTest.cfg
    python scripts/simulateWithSEDs.py tests/parametersTest.cfg
    coverage run --source delight -m py.test

Getting Started
----------------

- :ref:`Tutorial - getting started with Delight`
- :ref:`Example - filling missing bands`
- The ./notebooks directory contains some other notebooks.
- The code in the ./scripts directory is also a good start.
