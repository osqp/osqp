#!/bin/bash
set -ev


# Test OSQP C
${TRAVIS_BUILD_DIR}/build/out/osqp_tester_direct

# Test OSQP Python
cd ${TRAVIS_BUILD_DIR}/interfaces/python
nosetests
