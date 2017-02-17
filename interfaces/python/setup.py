from __future__ import print_function
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from shutil import copyfile
from numpy import get_include
from glob import glob
from subprocess import call
from platform import system
import os


# PARAMETERS
PRINTING = True
PROFILING = True
DFLOAT = False
DLONG = True


# Add parameters to cmake_args and define_macros
cmake_args = []
define_macros = []
if PROFILING:
    cmake_args += ['-DPROFILING:BOOL=ON']
    define_macros += [('PROFILING', None)]
else:
    cmake_args += ['-DPROFILING:BOOL=OFF']

if PRINTING:
    cmake_args += ['-DPRINTING:BOOL=ON']
    define_macros += [('PRINTING', None)]
else:
    cmake_args += ['-DPRINTING:BOOL=OFF']

if DLONG:
    cmake_args += ['-DDLONG:BOOL=ON']
    define_macros += [('DLONG', None)]
else:
    cmake_args += ['-DDLONG:BOOL=OFF']

if DFLOAT:
    cmake_args += ['-DDFLOAT:BOOL=ON']
    define_macros += [('DFLOAT', None)]
else:
    cmake_args += ['-DDFLOAT:BOOL=OFF']

# Check if windows linux or mac to pass flag
if system() == 'Windows':
    define_macros += [('IS_WINDOWS', None)]
elif system() == 'Linux':
    define_macros += [('IS_LINUX', None)]
elif system() == 'Darwin':
    define_macros += [('IS_MAC', None)]




# Define osqp and suitesparse directories
osqp_dir = os.path.join('..','..')
osqp_build_dir = os.path.join(osqp_dir, 'build')

# Interface files
include_dirs = [get_include(),                          # Numpy directories
                os.path.join(osqp_dir, 'include')]      # osqp.h

sources_files = glob(os.path.join('src', '*.c'))

# Set optimizer flag
if system() != 'Windows':
    compile_args = ["-O3"]
else:
    compile_args = []


# External libraries
libraries = []
library_dirs = []
if system() == 'Linux':
    libraries += ['rt']


# Add OSQP compiled library
# if system() == 'Windows':
    # lib_ext = '.lib'
# else:
lib_ext = '.a'
extra_objects = [os.path.join('src', 'libosqpdirstatic%s' % lib_ext)]



# Get compiler command if windows or unix
if system() == 'Windows':
    make_cmd = 'mingw32-make.exe'
else:
    make_cmd = 'make'

class build_ext_osqp(build_ext):
    def build_extensions(self):
        # Compile OSQP using CMake

        # Create build directory
        if not os.path.exists(osqp_build_dir):
            os.makedirs(osqp_build_dir)
        os.chdir(osqp_build_dir)

        # Compile static library with CMake
        call(['cmake'] + cmake_args + ['..'])
        call([make_cmd, 'osqpdirstatic'])

        # Change directory back to the python interface
        os.chdir(os.path.join('..', 'interfaces', 'python'))

        # Copy static library to src folder
        lib_origin = os.path.join(osqp_dir, 'build', 'out',
                                  'libosqpdirstatic%s' % lib_ext)
        lib_name = os.path.split(lib_origin)[-1]
        copyfile(lib_origin, os.path.join('src', lib_name))

        # Run extension
        build_ext.build_extensions(self)



_osqp = Extension('_osqp',
                  define_macros=define_macros,
                  libraries=libraries,
                  library_dirs=library_dirs,
                  include_dirs=include_dirs,
                  extra_objects=extra_objects,
                  sources=sources_files,
                  extra_compile_args=compile_args)

setup(name='osqp',
      version='0.0.0',
      author='Bartolomeo Stellato, Goran Banjac',
      description='This is the Python package for OSQP: Operator Splitting solver for Quadratic Programs.',
      package_dir={'': 'src'},
      install_requires=["numpy >= 1.7", "scipy >= 0.13.2", "future"],
      license='Apache 2.0',
      cmdclass = {'build_ext': build_ext_osqp},
      py_modules=['osqp', 'osqppurepy', '_osqppurepy'],
      ext_modules=[_osqp])
