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

# Check if windows linux or mac to pass flag
if system() == 'Windows':
    define_macros += [('IS_WINDOWS', None)]
    cmake_args += ['-G', 'MinGW Makefiles']
else:
    cmake_args += ['-G', 'Unix Makefiles']
    if system() == 'Linux':
        define_macros += [('IS_LINUX', None)]
    elif system() == 'Darwin':
        define_macros += [('IS_MAC', None)]


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


# Define osqp and suitesparse directories
osqp_dir = os.path.join('..', '..')
osqp_build_dir = os.path.join(osqp_dir, 'build')
suitesparse_dir = os.path.join(osqp_dir, 'lin_sys', 'direct', 'suitesparse')

# Interface files
include_dirs = [
    get_include(),                                      # Numpy directories
    os.path.join(osqp_dir, 'include'),                  # osqp.h
    os.path.join(suitesparse_dir),                      # private.h
    os.path.join(suitesparse_dir, 'ldl', 'include'),    # ldl.h
    os.path.join(suitesparse_dir, 'amd', 'include'),    # amd.h
    os.path.join('extension', 'include')]               # auxiliary .h files

sources_files = glob(os.path.join('extension', 'src', '*.c'))


# Set optimizer flag
if system() != 'Windows':
    compile_args = ["-O3"]
else:
    compile_args = []

# External libraries
libraries = []
library_dirs = []

# Add OSQP compiled library
lib_ext = '.a'
extra_objects = [os.path.join('extension', 'src',
                 'libosqpdirstatic%s' % lib_ext)]


'''
Copy C sources for code generation
'''

# List with OSQP C files
cfiles = [os.path.join(osqp_dir, 'src', f)
          for f in os.listdir(os.path.join(osqp_dir, 'src'))
          if f.endswith('.c') and f != 'polish.c']
cfiles += [os.path.join(suitesparse_dir, f)
           for f in os.listdir(suitesparse_dir)
           if f.endswith('.c') and f != 'SuiteSparse_config.c']
cfiles += [os.path.join(suitesparse_dir, 'ldl', 'src', f)
           for f in os.listdir(os.path.join(suitesparse_dir, 'ldl', 'src'))
           if f.endswith('.c')]

# List with OSQP H files
hfiles = [os.path.join(osqp_dir, 'include', f)
          for f in os.listdir(os.path.join(osqp_dir, 'include'))
          if f.endswith('.h') and f != 'polish.h']
hfiles += [os.path.join(suitesparse_dir, f)
           for f in os.listdir(suitesparse_dir)
           if f.endswith('.h') and f != 'SuiteSparse_config.h']
hfiles += [os.path.join(suitesparse_dir, 'ldl', 'include', f)
           for f in os.listdir(os.path.join(suitesparse_dir, 'ldl', 'include'))
           if f.endswith('.h')]

# List Jinja files
jinjafiles_list = glob(os.path.join('osqp', 'codegen', 'jinja', '*.*'))


class build_ext_osqp(build_ext):
    def build_extensions(self):
        # Compile OSQP using CMake

        # Create build directory
        if not os.path.exists(osqp_build_dir):
            os.makedirs(osqp_build_dir)
        os.chdir(osqp_build_dir)

        # Compile static library with CMake
        call(['cmake'] + cmake_args + ['..'])
        call(['cmake', '--build', '.', '--target', 'osqpdirstatic'])

        # Change directory back to the python interface
        os.chdir(os.path.join('..', 'interfaces', 'python'))

        # Copy static library to src folder
        lib_name = 'libosqpdirstatic%s' % lib_ext
        lib_origin = os.path.join(osqp_build_dir, 'out', lib_name)
        copyfile(lib_origin, os.path.join('extension', 'src', lib_name))

        # Run extension
        build_ext.build_extensions(self)


_osqp = Extension('osqp._osqp',
                  define_macros=define_macros,
                  libraries=libraries,
                  library_dirs=library_dirs,
                  include_dirs=include_dirs,
                  extra_objects=extra_objects,
                  sources=sources_files,
                  extra_compile_args=compile_args)

packages = ['osqp',
            'osqp.codegen',
            'osqppurepy']

setup(name='osqp',
      version='0.0.0',
      author='Bartolomeo Stellato, Goran Banjac',
      description='This is the Python package for OSQP: ' +
                  'Operator Splitting solver for Quadratic Programs.',
      package_dir={'osqp': 'module',
                   'osqppurepy': 'modulepurepy'},
      data_files=[('osqp/codegen/sources/src', cfiles),
                  ('osqp/codegen/sources/include', hfiles),
                  ('osqp/codegen/jinja', jinjafiles_list)],
      install_requires=["numpy >= 1.7", "scipy >= 0.13.2", "future"],
      license='Apache 2.0',
      cmdclass={'build_ext': build_ext_osqp},
      packages=packages,
      ext_modules=[_osqp])
