#!/bin/bash
set -e

# Add MKL shared libraries to the path
# This unfortunately does not work in travis OSX if I put the export command
# in the install.sh (it works on linux though)
export MKL_SHARED_LIB_DIR=`ls -rd ${CONDA_ROOT}/pkgs/*/ | grep mkl-2 | head -n 1`lib:`ls -rd ${CONDA_ROOT}/pkgs/*/ | grep intel-openmp- | head -n 1`lib

echo "MKL shared library path: ${MKL_SHARED_LIB_DIR}"

if [[ "${TRAVIS_OS_NAME}" == "linux" ]]; then
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MKL_SHARED_LIB_DIR}
else if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${MKL_SHARED_LIB_DIR}
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
#cmake -G "Unix Makefiles" -DCOVERAGE=ON -DUNITTESTS=OFF ..
#make
#${TRAVIS_BUILD_DIR}/build/out/osqp_tester
# Pefrorm code coverage (only in Linux case)
if [[ $TRAVIS_OS_NAME == "linux" ]]; then
    if [[ $PLAT == "aarch64" ]]; then
       echo "Tanveen aarch"
       cmake -G "Unix Makefiles" -DCOVERAGE=ON -DUNITTESTS=ON -DENABLE_MKL_PARDISO=OFF ..
       make
     else
       echo "Tanveen non aarch"
       cmake -G "Unix Makefiles" -DCOVERAGE=ON -DUNITTESTS=ON ..
       make
    fi
    ${TRAVIS_BUILD_DIR}/build/out/osqp_tester
    cd ${TRAVIS_BUILD_DIR}/build
    lcov --directory . --capture -o coverage.info # capture coverage info
    lcov --remove coverage.info "${TRAVIS_BUILD_DIR}/tests/*" \
        "${TRAVIS_BUILD_DIR}/lin_sys/direct/qdldl/amd/*" \
        "${TRAVIS_BUILD_DIR}/lin_sys/direct/qdldl/qdldl_sources/*" \
        "/usr/include/x86_64-linux-gnu/**/*" \
        -o coverage.info # filter out tests and unnecessary files
     lcov --list coverage.info # debug before upload
     coveralls-lcov coverage.info # uploads to coveralls
else
   cmake -G "Unix Makefiles" -DCOVERAGE=ON -DUNITTESTS=ON -DENABLE_MKL_PARDISO=OFF ..
   make
   ${TRAVIS_BUILD_DIR}/build/out/osqp_tester
fi

if [[ $TRAVIS_OS_NAME == "linux" ]]; then
    echo "Testing OSQP with valgrind (disabling MKL pardiso for memory allocation issues)"
    cd ${TRAVIS_BUILD_DIR}
    rm -rf build
    mkdir build
    cd build
    #disable PARDISO since intel instructions in MKL
    #cause valgrind 3.11 to fail
    cmake -G "Unix Makefiles" -DENABLE_MKL_PARDISO=OFF -DUNITTESTS=ON ..
    make
    valgrind --suppressions=${TRAVIS_BUILD_DIR}/.valgrind-suppress.supp --leak-check=full --gen-suppressions=all --track-origins=yes --error-exitcode=42 ${TRAVIS_BUILD_DIR}/build/out/osqp_tester
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
if [[ $PLAT == 'aarch64' ]]; then
    echo "Tanveen aarch 2"
    cmake -G "Unix Makefiles" -DDLONG=OFF -DENABLE_MKL_PARDISO=OFF -DUNITTESTS=ON ..
else
    cmake -G "Unix Makefiles" -DDLONG=OFF -DUNITTESTS=ON ..
fi
make
${TRAVIS_BUILD_DIR}/build/out/osqp_tester

echo "Building OSQP with embedded=1"
cd ${TRAVIS_BUILD_DIR}
rm -rf build
mkdir build
cd build
cmake -G "Unix Makefiles" -DEMBEDDED=1 ..
make

echo "Building OSQP with embedded=2"
cd ${TRAVIS_BUILD_DIR}
rm -rf build
mkdir build
cd build
cmake -G "Unix Makefiles" -DEMBEDDED=2 ..
make

echo "Building OSQP without profiling"
cd ${TRAVIS_BUILD_DIR}
rm -rf build
mkdir build
cd build
cmake -G "Unix Makefiles" -DPROFILING=OFF ..
make

echo "Building OSQP without user interrupt"
cd ${TRAVIS_BUILD_DIR}
rm -rf build
mkdir build
cd build
cmake -G "Unix Makefiles" -DCTRLC=OFF ..
make

echo "Testing OSQP without printing"
cd ${TRAVIS_BUILD_DIR}
rm -rf build
mkdir build
cd build
if [[ $PLAT == 'aarch64' ]]; then
    echo "Tanveen aarch 3"
    cmake -G "Unix Makefiles" -DPRINTING=OFF -DUNITTESTS=ON -DENABLE_MKL_PARDISO=OFF ..
else
    cmake -G "Unix Makefiles" -DPRINTING=OFF -DUNITTESTS=ON ..
fi
make
${TRAVIS_BUILD_DIR}/build/out/osqp_tester


# Test custom memory management
# ---------------------------------------------------

echo "Test OSQP custom allocators"
cd ${TRAVIS_BUILD_DIR}
rm -rf build
mkdir build
cd build
if [[ $PLAT == 'aarch64' ]]; then
    echo "Tanveen aarch64 4"
    cmake -DUNITTESTS=ON -DENABLE_MKL_PARDISO=OFF -DOSQP_CUSTOM_MEMORY=${TRAVIS_BUILD_DIR}/tests/custom_memory/custom_memory.h ..
else
   cmake -DUNITTESTS=ON -DOSQP_CUSTOM_MEMORY=${TRAVIS_BUILD_DIR}/tests/custom_memory/custom_memory.h ..
fi
make osqp_tester_custom_memory
${TRAVIS_BUILD_DIR}/build/out/osqp_tester_custom_memory


cd ${TRAVIS_BUILD_DIR}

set +e
