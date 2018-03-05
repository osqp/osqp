from __future__ import print_function
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from shutil import copyfile, copy
from numpy import get_include
from glob import glob
import shutil as sh
from subprocess import call, check_output
from platform import system, architecture
import os
import sys


# Add parameters to cmake_args and define_macros
cmake_args = ["-DUNITTESTS=OFF"]
define_macros = []

# Check if windows linux or mac to pass flag
if system() == 'Windows':
    cmake_args += ['-G', 'MinGW Makefiles']

else:  # Linux or Mac
    cmake_args += ['-G', 'Unix Makefiles']

# Pass Python option to CMake and Python interface compilation
cmake_args += ['-DPYTHON=ON']

# Pass python to compiler launched from setup.py
define_macros += [('PYTHON', None)]

# Pass python version to cmake
py_version = "%i.%i" % sys.version_info[:2]
cmake_args += ['-DPYTHON_VER_NUM=%s' % py_version]


# Define osqp and suitesparse directories
current_dir = os.getcwd()
osqp_dir = os.path.join('osqp_sources')
osqp_build_dir = os.path.join(osqp_dir, 'build')
suitesparse_dir = os.path.join(osqp_dir, 'lin_sys', 'direct', 'suitesparse')

# Interface files
include_dirs = [
    get_include(),                                      # Numpy directories
    os.path.join(osqp_dir, 'include'),                  # osqp.h
    # suitesparse_ldl headers to extract workspace for codegen
    os.path.join(suitesparse_dir),
    os.path.join('extension', 'include')]               # auxiliary .h files

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
                 'libosqpstatic%s' % lib_ext)]


'''
Copy C sources for code generation
'''

# Create codegen directory
osqp_codegen_sources_dir = os.path.join('module', 'codegen', 'sources')
if os.path.exists(osqp_codegen_sources_dir):
    sh.rmtree(osqp_codegen_sources_dir)
os.makedirs(osqp_codegen_sources_dir)

# OSQP C files
cfiles = [os.path.join(osqp_dir, 'src', f)
          for f in os.listdir(os.path.join(osqp_dir, 'src'))
          if f.endswith('.c') and f not in ('cs.c', 'ctrlc.c', 'polish.c',
          'lin_sys.c')]
cfiles += [os.path.join(suitesparse_dir, f)
           for f in os.listdir(suitesparse_dir)
           if f.endswith('.c') and f != 'SuiteSparse_config.c']
cfiles += [os.path.join(suitesparse_dir, 'ldl', 'src', f)
           for f in os.listdir(os.path.join(suitesparse_dir, 'ldl', 'src'))
           if f.endswith('.c')]
osqp_codegen_sources_c_dir = os.path.join(osqp_codegen_sources_dir, 'src')
if os.path.exists(osqp_codegen_sources_c_dir):  # Create destination directory
    sh.rmtree(osqp_codegen_sources_c_dir)
os.makedirs(osqp_codegen_sources_c_dir)
for f in cfiles:  # Copy C files
    copy(f, osqp_codegen_sources_c_dir)

# List with OSQP H files
hfiles = [os.path.join(osqp_dir, 'include', f)
          for f in os.listdir(os.path.join(osqp_dir, 'include'))
          if f.endswith('.h') and f not in ('glob_opts.h', 'cs.h',
                                            'ctrlc.h', 'polish.h',
                                            'lin_sys.h')]
hfiles += [os.path.join(suitesparse_dir, f)
           for f in os.listdir(suitesparse_dir)
           if f.endswith('.h') and f != 'SuiteSparse_config.h']
hfiles += [os.path.join(suitesparse_dir, 'ldl', 'include', f)
           for f in os.listdir(os.path.join(suitesparse_dir, 'ldl', 'include'))
           if f.endswith('.h')]
osqp_codegen_sources_h_dir = os.path.join(osqp_codegen_sources_dir, 'include')
if os.path.exists(osqp_codegen_sources_h_dir):  # Create destination directory
    sh.rmtree(osqp_codegen_sources_h_dir)
os.makedirs(osqp_codegen_sources_h_dir)
for f in hfiles:  # Copy header files
    copy(f, osqp_codegen_sources_h_dir)

# List with OSQP configure files
configure_files = [os.path.join(osqp_dir, 'configure', 'glob_opts.h.in')]
osqp_codegen_sources_configure_dir = os.path.join(osqp_codegen_sources_dir, 'configure')
if os.path.exists(osqp_codegen_sources_configure_dir):  # Create destination directory
    sh.rmtree(osqp_codegen_sources_configure_dir)
os.makedirs(osqp_codegen_sources_configure_dir)
for f in configure_files:  # Copy configure files
    copy(f, osqp_codegen_sources_configure_dir)

# List of files to generate  (No longer needed. It is in MANIFEST.in)
#  files_to_generate = glob(os.path.join('module', 'codegen',
                                      #  'files_to_generate', '*.*'))



class build_ext_osqp(build_ext):
    def build_extensions(self):
        # Compile OSQP using CMake

        # Create build directory
        if os.path.exists(osqp_build_dir):
            sh.rmtree(osqp_build_dir)
        os.makedirs(osqp_build_dir)
        os.chdir(osqp_build_dir)

        try:
            check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build OSQP")

        # Compile static library with CMake
        call(['cmake'] + cmake_args + ['..'])
        call(['cmake', '--build', '.', '--target', 'osqpstatic'])

        # Change directory back to the python interface
        os.chdir(current_dir)

        # Copy static library to src folder
        lib_name = 'libosqpstatic%s' % lib_ext
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
      version='0.3.0',
      author='Bartolomeo Stellato, Goran Banjac',
      author_email='bartolomeo.stellato@gmail.com',
      description='OSQP: The Operator Splitting QP Solver',
      package_dir={'osqp': 'module',
                   'osqppurepy': 'modulepurepy'},
      include_package_data=True,  # Include package data from MANIFEST.in
      install_requires=["numpy >= 1.7", "scipy >= 0.13.2", "future"],
      license='Apache 2.0',
      url="http://osqp.readthedocs.io/",
      cmdclass={'build_ext': build_ext_osqp},
      packages=packages,
      ext_modules=[_osqp])
