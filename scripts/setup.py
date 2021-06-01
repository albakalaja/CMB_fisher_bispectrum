from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='Fisher',
    ext_modules=cythonize("compute_fisher.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
