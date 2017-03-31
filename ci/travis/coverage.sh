#!/bin/bash
set -ev

cd ${TRAVIS_BUILD_DIR}/build
lcov --directory . --capture -o coverage.info # capture coverage info
lcov --remove coverage.info "${TRAVIS_BUILD_DIR}/tests/*" \
    "${TRAVIS_BUILD_DIR}/lin_sys/direct/suitesparse/amd/*" \
    "${TRAVIS_BUILD_DIR}/lin_sys/direct/suitesparse/ldl/*" \
    "${TRAVIS_BUILD_DIR}/lin_sys/direct/suitesparse/SuiteSparse_config*" \
    -o coverage.info # filter out tests and unnecessary files
lcov --list coverage.info # debug before upload
coveralls-lcov coverage.info # uploads to coveralls
