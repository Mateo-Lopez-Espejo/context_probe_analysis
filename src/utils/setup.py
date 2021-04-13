from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Perfect Coverage',
    package_dir={'perfect_coverage': ''},
    ext_modules=cythonize("perfect_coverage.pyx"),
    zip_safe=False,
)

