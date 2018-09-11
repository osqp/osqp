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


echo "Creating Bintray package..."

# Compile OSQP
cd ${TRAVIS_BUILD_DIR}
rm -rf build
mkdir build
cd build
cmake -G "Unix Makefiles" ..
make

cd ${TRAVIS_BUILD_DIR}/build/out
if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    OS_NAME="mac"
    OS_SHARED_LIB_EXT="dylib"
else
    OS_NAME="linux"
    OS_SHARED_LIB_EXT="so"
fi
OSQP_DEPLOY_DIR=osqp-0.4.1-${OS_NAME}64
mkdir $OSQP_DEPLOY_DIR/
mkdir $OSQP_DEPLOY_DIR/lib
mkdir $OSQP_DEPLOY_DIR/include
# Copy license
cp ../../LICENSE $OSQP_DEPLOY_DIR/
# Copy includes
cp ../../include/*.h  $OSQP_DEPLOY_DIR/include
# Copy static library
cp libosqp.a $OSQP_DEPLOY_DIR/lib
# Copy shared library
cp libosqp.$OS_SHARED_LIB_EXT $OSQP_DEPLOY_DIR/lib
# Compress package
tar -czvf $OSQP_DEPLOY_DIR.tar.gz  $OSQP_DEPLOY_DIR


# Deploy package
curl -T $OSQP_DEPLOY_DIR.tar.gz -ubstellato:$BINTRAY_API_KEY -H "X-Bintray-Package:OSQP" -H "X-Bintray-Version:0.4.1" https://api.bintray.com/content/bstellato/generic/OSQP/0.4.1/

# Publish
curl -X POST -ubstellato:$BINTRAY_API_KEY https://api.bintray.com/content/bstellato/generic/OSQP/0.4.1/publish


exit 0
