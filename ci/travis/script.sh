#!/bin/bash
set -ev


# Update variables from install
# CMake
if [[ "${TRAVIS_OS_NAME}" == "linux" ]]; then
    export PATH=${DEPS_DIR}/cmake/bin:${PATH}
fi

# Anaconda
export PATH=${DEPS_DIR}/miniconda/bin:$PATH
hash -r
source activate testenv


# Test C interface
# ---------------------------------------------------

# Compile OSQP
echo "Change directory to Travis build ${TRAVIS_BUILD_DIR}"
cd ${TRAVIS_BUILD_DIR}
mkdir build
cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug ..
make


# Test OSQP C
${TRAVIS_BUILD_DIR}/build/out/osqp_tester_direct


# Test Python interface
# ---------------------------------------------------

# Install Python interface
cd ${TRAVIS_BUILD_DIR}/interfaces/python
python setup.py install

# Test OSQP Python
cd ${TRAVIS_BUILD_DIR}/interfaces/python
nosetests
