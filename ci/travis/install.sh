#!/bin/bash

# Create deps dir
mkdir ${DEPS_DIR}
cd ${DEPS_DIR}

# Install lcov and valgrind on linux
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
    sudo apt-get update -y
    sudo apt-get install -y lcov
    sudo apt-get install -y valgrind
else if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    brew update
    brew install lcov
    fi
fi

# Install Anaconda
# Use the miniconda installer for faster download / install of conda
# itself
if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh;
else
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
fi
export CONDA_ROOT=${DEPS_DIR}/miniconda
chmod +x miniconda.sh && ./miniconda.sh -b -p $CONDA_ROOT
export PATH=${DEPS_DIR}/miniconda/bin:$PATH
hash -r
conda config --set always_yes yes --set changeps1 no
conda update --yes -q conda
conda create -n testenv --yes python=$PYTHON_VERSION mkl numpy scipy future
source activate testenv

# Install cmake
conda install --yes -c conda-forge cmake

# Install coveralls lcov
gem install coveralls-lcov

cd ${TRAVIS_BUILD_DIR}
