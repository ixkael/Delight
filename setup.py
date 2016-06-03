from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules=[ Extension("delight.photoz_kernels_cy",
              ["delight/photoz_kernels_cy.pyx"],
              libraries=["m"],
              include_dirs=[numpy.get_include()],
              extra_link_args=['-fopenmp'],
              extra_compile_args = ["-ffast-math", "-march=native", "-fopenmp"])]

setup(
  name = "photoz_kernels_cy",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)
