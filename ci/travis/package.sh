#!/bin/bash
set -e
echo "Creating Bintray shared library package..."

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
OSQP_BIN=osqp-${OSQP_VERSION}-${OS_NAME}64
mkdir $OSQP_BIN/
mkdir $OSQP_BIN/lib
mkdir $OSQP_BIN/include
# Copy license
cp ../../LICENSE $OSQP_BIN/
# Copy includes
cp ../../include/*.h  $OSQP_BIN/include
# Copy static library
cp libosqp.a $OSQP_BIN/lib
# Copy shared library
cp libosqp.$OS_SHARED_LIB_EXT $OSQP_BIN/lib
# Compress package
tar -czvf ${TRAVIS_BUILD_DIR}/$OSQP_BIN.tar.gz  $OSQP_BIN


echo "Creating Bintray sources package..."

OSQP_SOURCES=osqp-${OSQP_VERSION}
# Clone OSQP repository
cd ${TRAVIS_BUILD_DIR}
mkdir sources/
cd sources/
git clone https://github.com/$TRAVIS_REPO_SLUG.git ${OSQP_SOURCES} --recursive
cd ${OSQP_SOURCES}
git checkout -qf $TRAVIS_COMMIT
git submodule update
cd ..
# Create archive ignoring hidden files
tar --exclude=".*" -czvf ${TRAVIS_BUILD_DIR}/${OSQP_SOURCES}.tar.gz ${OSQP_SOURCES}

# Create bintray.json file from bintray.json.in
BINTRAY_DEST_FILE="${TRAVIS_BUILD_DIR}/ci/travis/bintray.json"
BINTRAY_TEMPLATE_FILE="${BINTRAY_DEST_FILE}.in"
sed -e "s/@OSQP_PACKAGE_NAME@/${OSQP_PACKAGE_NAME}/g" \
    -e "s/@OSQP_VERSION@/${OSQP_VERSION}/g" \
    "${BINTRAY_TEMPLATE_FILE}" > "${BINTRAY_DEST_FILE}"

# Create dist folder and copy artifacts
DIST_DIR=${TRAVIS_BUILD_DIR}/dist
mkdir ${DIST_DIR}
cp ${TRAVIS_BUILD_DIR}/${OSQP_SOURCES}.tar.gz ${DIST_DIR}
cp ${TRAVIS_BUILD_DIR}/${OSQP_BIN}.tar.gz ${DIST_DIR}

cd ${TRAVIS_BUILD_DIR}

set +e
