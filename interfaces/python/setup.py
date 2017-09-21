from __future__ import print_function
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from shutil import copyfile
from numpy import get_include
from glob import glob
import shutil as sh
from subprocess import call
from platform import system
import os
import sys


# PARAMETERS
PRINTING = True
PROFILING = True
DFLOAT = False
DLONG = True
CTRLC = True


# Add parameters to cmake_args and define_macros
cmake_args = ["-DUNITTESTS=OFF"]
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

if CTRLC:
    cmake_args += ['-DCTRLC:BOOL=ON']
    define_macros += [('CTRLC', None)]
else:
    cmake_args += ['-DCTRLC:BOOL=OFF']

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


# Pass Python option to CMake and Python interface compilation
cmake_args += ['-DPYTHON=ON']

# Pass python version to cmake
py_version = "%i.%i" % sys.version_info[:2]
cmake_args += ['-DPYTHON_VER_NUM=%s' % py_version]

# Pass python to compiler
define_macros += [('PYTHON', None)]


# Define osqp and suitesparse directories
current_dir = os.getcwd()
osqp_dir = os.path.join('osqp')
osqp_build_dir = os.path.join(osqp_dir, 'build')
suitesparse_dir = os.path.join(osqp_dir, 'lin_sys', 'direct', 'suitesparse')

# Interface files
include_dirs = [
    get_include(),                                      # Numpy directories
    os.path.join(osqp_dir, 'include'),                  # osqp.h
    os.path.join(suitesparse_dir),                      # private.h
    os.path.join(suitesparse_dir, 'ldl', 'include'),    # ldl.h
    os.path.join(suitesparse_dir, 'amd', 'include'),    # amd.h
    os.path.join(current_dir, 'extension', 'include')]  # auxiliary .h files

sources_files = glob(os.path.join('extension', 'src', '*.c'))


# Set optimizer flag
if system() != 'Windows':
    compile_args = ["-O3"]
else:
    compile_args = []

# External libraries
library_dirs = []
libraries = []
if system() == 'Linux':
    libraries += ['rt']
if system() == 'Windows' and sys.version_info[0] == 3:
    # They moved the stdio library to another place.
    # We need to include this to fix the dependency
    libraries += ['legacy_stdio_definitions']

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
          if f.endswith('.c') and f not in ('cs.c', 'ctrlc.c', 'polish.c')]
cfiles += [os.path.join(suitesparse_dir, f)
           for f in os.listdir(suitesparse_dir)
           if f.endswith('.c') and f != 'SuiteSparse_config.c']
cfiles += [os.path.join(suitesparse_dir, 'ldl', 'src', f)
           for f in os.listdir(os.path.join(suitesparse_dir, 'ldl', 'src'))
           if f.endswith('.c')]

# List with OSQP H files
hfiles = [os.path.join(osqp_dir, 'include', f)
          for f in os.listdir(os.path.join(osqp_dir, 'include'))
          if f.endswith('.h') and f not in ('cs.h', 'ctrlc.h', 'polish.h')]
hfiles += [os.path.join(suitesparse_dir, f)
           for f in os.listdir(suitesparse_dir)
           if f.endswith('.h') and f != 'SuiteSparse_config.h']
hfiles += [os.path.join(suitesparse_dir, 'ldl', 'include', f)
           for f in os.listdir(os.path.join(suitesparse_dir, 'ldl', 'include'))
           if f.endswith('.h')]

# List of files to generate
files_to_generate = glob(os.path.join('module', 'codegen',
                                      'files_to_generate', '*.*'))


class build_ext_osqp(build_ext):
    def build_extensions(self):
        # Compile OSQP using CMake

        # Create build directory
        if os.path.exists(osqp_build_dir):
            sh.rmtree(osqp_build_dir)
        os.makedirs(osqp_build_dir)
        os.chdir(osqp_build_dir)

        try:
            call(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build OSQP")

        # Compile static library with CMake
        call(['cmake'] + cmake_args + ['..'])
        call(['cmake', '--build', '.', '--target', 'osqpdirstatic'])

        # Change directory back to the python interface
        #  import ipdb; ipdb.set_trace()
        os.chdir(current_dir)
        #  os.chdir(os.path.join('..', 'interfaces', 'python'))

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
      version='0.1.3',
      author='Bartolomeo Stellato, Goran Banjac',
      description='OSQP: The Operator Splitting QP Solver',
      package_dir={'osqp': 'module',
                   'osqppurepy': 'modulepurepy'},
      data_files=[('osqp/codegen/sources/src', cfiles),
                  ('osqp/codegen/sources/include', hfiles),
                  ('osqp/codegen/files_to_generate', files_to_generate)],
      install_requires=["numpy >= 1.7", "scipy >= 0.13.2", "future"],
      license='Apache 2.0',
      url="http://osqp.readthedocs.io/",
      cmdclass={'build_ext': build_ext_osqp},
      packages=packages,
      ext_modules=[_osqp])
