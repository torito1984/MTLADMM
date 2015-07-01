# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:30:03 2015

@author: Michele
"""

from distutils.core import setup
from Cython.Build import cythonize
from numpy.distutils.misc_util import Configuration
from numpy.distutils.system_info import get_info, BlasNotFoundError
import numpy
import os
from os.path import join
import warnings

#setup(
#  ext_modules = cythonize("Cmultitask_admm.pyx"),
#  include_dirs=[numpy.get_include()]
#)

def configuration(parent_package='', top_path=None):

    config = Configuration('multitask_learning', parent_package, top_path)

    # some libs needs cblas, fortran-compiled BLAS will not be sufficient
    blas_info = get_info('blas_opt', 0)
    if (not blas_info) or (('NO_ATLAS_INFO', 1) in blas_info.get('define_macros', [])):
        config.add_library('cblas',sources=[join('src', 'cblas', '*.c')])
        warnings.warn(BlasNotFoundError.__doc__)

    return config


setup(ext_modules = cythonize("Cmultitask_admm.pyx"), include_dirs=[numpy.get_include()], **configuration(top_path='').todict())

# python setup.py build_ext --inplace
