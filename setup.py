#!/usr/bin/env python

from setuptools import setup
from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from Cython.Build import cythonize

def numpy_include():
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()
    return numpy_include


rank_cy_modules = [
    Extension(
        'rank_cy',
        ['fastreid/evaluation/rank_cylib/rank_cy.pyx'],
        include_dirs=[numpy_include()],
    )
]


setup(
    name="fastreid",
    version="0.1",
    author="FAIR",
    url="https://github.com/JDAI-CV/fast-reid",
    description="JDAI-CV",
    python_requires=">=3.6",
    ext_modules=cythonize(rank_cy_modules),
    # cmdclass={"build_ext": },
)
