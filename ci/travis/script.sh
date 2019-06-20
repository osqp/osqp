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

# Add MKL shared libraries to the path
MKL_SHARED_LIB_DIR=`ls -d ${DEPS_DIR}/miniconda/pkgs/*/lib | grep mkl-2 | tail -1`
OPENMP_SHARED_LIB_DIR=`ls -d ${DEPS_DIR}/miniconda/pkgs/*/lib | grep openmp | tail -1`
if [[ "${TRAVIS_OS_NAME}" == "linux" ]]; then
    export LD_LIBRARY_PATH=${MKL_SHARED_LIB_DIR}:${OPENMP_SHARED_LIB_DIR}:${LD_LIBRARY_PATH}
else if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    export DYLD_LIBRARY_PATH=${MKL_SHARED_LIB_DIR}:${OPENMP_SHARED_LIB_DIR}:${DYLD_LIBRARY_PATH}
fi
fi



# Test C interface
# ---------------------------------------------------

# Compile and test OSQP
echo "Change directory to Travis build ${TRAVIS_BUILD_DIR}"
echo "Testing OSQP with standard configuration"
cd ${TRAVIS_BUILD_DIR}
mkdir build
cd build
cmake -G "Unix Makefiles" -DCOVERAGE=ON -DUNITTESTS=ON ..
make
${TRAVIS_BUILD_DIR}/build/out/osqp_tester

if [[ $TRAVIS_OS_NAME == "linux" ]]; then
    # Check memory with valgrind
    valgrind --suppressions=${TRAVIS_BUILD_DIR}/.valgrind-suppress.supp --leak-check=full --gen-suppressions=all --track-origins=yes --error-exitcode=42 ${TRAVIS_BUILD_DIR}/build/out/osqp_tester
    # Perform code coverage
    cd ${TRAVIS_BUILD_DIR}/build
    lcov --directory . --capture -o coverage.info # capture coverage info
    lcov --remove coverage.info "${TRAVIS_BUILD_DIR}/tests/*" \
        "${TRAVIS_BUILD_DIR}/lin_sys/direct/qdldl/amd/*" \
        "${TRAVIS_BUILD_DIR}/lin_sys/direct/qdldl/qdldl_sources/*" \
        "/usr/include/x86_64-linux-gnu/**/*" \
        -o coverage.info # filter out tests and unnecessary files
    lcov --list coverage.info # debug before upload
    coveralls-lcov coverage.info # uploads to coveralls
fi


echo "Testing OSQP with floats"
cd ${TRAVIS_BUILD_DIR}
rm -rf build
mkdir build
cd build
cmake -G "Unix Makefiles" -DDFLOAT=ON -DUNITTESTS=ON ..
make
${TRAVIS_BUILD_DIR}/build/out/osqp_tester

echo "Testing OSQP without long integers"
cd ${TRAVIS_BUILD_DIR}
rm -rf build
mkdir build
cd build
cmake -G "Unix Makefiles" -DDLONG=OFF -DUNITTESTS=ON ..
make
${TRAVIS_BUILD_DIR}/build/out/osqp_tester



echo "Testing OSQP without printing"
cd ${TRAVIS_BUILD_DIR}
rm -rf build
mkdir build
cd build
cmake -G "Unix Makefiles" -DPRINTING=OFF -DUNITTESTS=ON ..
make
${TRAVIS_BUILD_DIR}/build/out/osqp_tester


# Test custom memory management
# ---------------------------------------------------

echo "Test OSQP custom allocators"
cd ${TRAVIS_BUILD_DIR}
rm -rf build
mkdir build
cd build
cmake -DUNITTESTS=OFF \
    -DOSQP_CUSTOM_MEMORY=${TRAVIS_BUILD_DIR}/tests/custom_memory/custom_memory.h \
    ..
#make the static library first: it will be missing symbols
#for the custom memory functions
make osqpstatic
#compile the custom memory management functions
g++ -c ${TRAVIS_BUILD_DIR}/tests/custom_memory/custom_memory.c \
    -o ${TRAVIS_BUILD_DIR}/build/custom_memory.o
#for testing purposes, just add this to osqp
#static library.   For real applications, users
#may prefer to keep it separate and link both
#to their application
ar -r -cs ${TRAVIS_BUILD_DIR}/build/out/libosqp.a \
        ${TRAVIS_BUILD_DIR}/build/custom_memory.o

#now make the rest of the demos and test
make osqp_demo
${TRAVIS_BUILD_DIR}/build/out/osqp_demo
