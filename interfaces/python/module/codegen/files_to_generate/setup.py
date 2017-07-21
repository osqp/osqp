from setuptools import setup, Extension
from platform import system
from numpy import get_include
from glob import glob
import os


'''
Define macros
'''
define_macros = []

if system() == 'Windows':
    define_macros += [('IS_WINDOWS', None)]
else:
    if system() == 'Linux':
        define_macros += [('IS_LINUX', None)]
    elif system() == 'Darwin':
        define_macros += [('IS_MAC', None)]


define_macros += [('DLONG', None)]
define_macros += [('EMBEDDED', EMBEDDED_FLAG)]
define_macros += [('PYTHON', None)]

'''
Define compiler flags
'''
if system() != 'Windows':
    compile_args = ["-O3"]
else:
    compile_args = []


# Add additional libraries
libraries = []
if system() == 'Linux':
    libraries = ['rt']

'''
Include directory
'''
include_dirs = [get_include(),                  # Numpy includes
                os.path.join('..', 'include')]  # OSQP includes


'''
Source files
'''
sources_files = ['PYTHON_EXT_NAMEmodule.c']             # Python wrapper
sources_files += glob(os.path.join('osqp', '*.c'))      # OSQP files


PYTHON_EXT_NAME = Extension('PYTHON_EXT_NAME',
                          define_macros=define_macros,
                          libraries=libraries,
                          include_dirs=include_dirs,
                          sources=sources_files,
                          extra_compile_args=compile_args)


setup(name='PYTHON_EXT_NAME',
      version='0.1.2',
      author='Bartolomeo Stellato, Goran Banjac',
      description='This is the Python module for embedded OSQP: ' +
                  'Operator Splitting solver for Quadratic Programs.',
      install_requires=["numpy >= 1.7", "future"],
      license='Apache 2.0',
      ext_modules=[PYTHON_EXT_NAME])
