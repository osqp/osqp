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
# TODO: ADD



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
if system() == 'Windows':
    lib_ext = '.lib'
else:
    lib_ext = '.a'
extra_objects = [os.path.join('src', 'libosqpdirstatic%s' % lib_ext)]


class build_ext_osqp(build_ext):
    def build_extensions(self):
        # Compile OSQP using CMake

        # Create build directory
        if not os.path.exists(osqp_build_dir):
            os.makedirs(osqp_build_dir)
        os.chdir(osqp_build_dir)

        # Run cmake to create the static library
        call(['cmake',              #  Pass settings
              '-DPRINTING:BOOL=ON',
              '-DPROFILING:BOOL=ON',
              '-DDLONG:BOOL=ON',
              '-DDFLOAT:BOOL=OFF',
              '..'])
        call(['make', 'osqpdirstatic'])

        # Change directory back to the python interface
        os.chdir(os.path.join('..', 'interfaces', 'python'))

        # Copy static library to src folder
        lib_origin = os.path.join(osqp_dir, 'build', 'lib',
                                  'libosqpdirstatic%s' % lib_ext)
        lib_name = os.path.split(lib_origin)[-1]
        copyfile(lib_origin, os.path.join('src', lib_name))

        # Run extension
        build_ext.build_extensions(self)



_osqp = Extension('_osqp',
                  define_macros=[('PRINTING', None),
                                 ('PROFILING', None),
                                 ('DLONG', None),
                                 ('PYTHON', None)],
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
      install_requires=["numpy >= 1.7", "scipy >= 0.13.2"],
      license='Apache 2.0',
      cmdclass = {'build_ext': build_ext_osqp},
      py_modules=['osqp', 'osqppurepy', '_osqppurepy'],
      ext_modules=[_osqp])
