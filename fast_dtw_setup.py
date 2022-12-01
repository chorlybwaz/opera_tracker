# python fast_dtw_setup.py build_ext --inplace

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
	ext_modules = cythonize('fast_dtw.pyx'),
	include_dirs=[numpy.get_include()],
	)

