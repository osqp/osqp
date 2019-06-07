#!/bin/bash

if [[ ${OSQP_VERSION} == *"dev"* ]]; then
    OSQP_PACKAGE_NAME="${OSQP_PACKAGE_NAME}-dev";
fi

# Create deps dir
mkdir ${DEPS_DIR}
cd ${DEPS_DIR}


# Install Anaconda

# Use the miniconda installer for faster download / install of conda
# itself
if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh;
else
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
fi
chmod +x miniconda.sh && ./miniconda.sh -b -p ${DEPS_DIR}/miniconda
export PATH=${DEPS_DIR}/miniconda/bin:$PATH
hash -r
conda config --set always_yes yes --set changeps1 no
conda update --yes -q conda
conda create -n testenv --yes python=$PYTHON_VERSION numpy scipy future
source activate testenv

# Install cmake valgrind and lcov
conda install --yes -c conda-forge cmake valgrind lcov

# # Install CMake
# if [[ "${TRAVIS_OS_NAME}" == "linux" ]]; then
#     CMAKE_URL="http://www.cmake.org/files/v3.7/cmake-3.7.1-Linux-x86_64.tar.gz"
# else
#     CMAKE_URL="http://www.cmake.org/files/v3.7/cmake-3.7.1-Darwin-x86_64.tar.gz"
# fi
# mkdir cmake && wget --quiet -O - ${CMAKE_URL} | tar --strip-components=1 -xz -C cmake
# export PATH=${DEPS_DIR}/cmake/bin:${PATH}
# cmake --version

# # Install lcov and valgrind on linux
# if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
#     sudo apt-get update -y
#     sudo apt-get install -y lcov
#     sudo apt-get install -y valgrind
# else if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
#     brew update
#     brew install lcov
#     fi
# fi

gem install coveralls-lcov



# Add MKL shared libraries to the path
export MKL_SHARED_LIB_DIR=`python -c 'import numpy.distutils.system_info as sysinfo; print(sysinfo.get_info("mkl")["library_dirs"][0])'`

echo "MKL shared library path: ${MKL_SHARED_LIB_DIR}"

# if [[ "${TRAVIS_OS_NAME}" == "linux" ]]; then
#     export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MKL_SHARED_LIB_DIR}
# else if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
#     export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${MKL_SHARED_LIB_DIR}
# fi
# fi
