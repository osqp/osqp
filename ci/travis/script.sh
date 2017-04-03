#!/bin/bash
set -ev

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
