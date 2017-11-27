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


# Create shared library archive for Bintray only if Python 3.6 
# NB: need to do it only once
if [[ "$PYTHON_VERSION" == "3.6" ]]; then
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
OSQP_DEPLOY_DIR=osqp-0.2.1-${OS_NAME}64
mkdir $OSQP_DEPLOY_DIR/
mkdir $OSQP_DEPLOY_DIR/lib
mkdir $OSQP_DEPLOY_DIR/include
# Copy license 
cp ../../LICENSE $OSQP_DEPLOY_DIR/
# Copy includes
cp ../../include/*.h  $OSQP_DEPLOY_DIR/include
# Copy static library
cp libosqpstatic.a $OSQP_DEPLOY_DIR/lib 
# Copy shared library
cp libosqp.$OS_SHARED_LIB_EXT $OSQP_DEPLOY_DIR/lib 
# Compress package
tar -czvf $OSQP_DEPLOY_DIR.tar.gz  $OSQP_DEPLOY_DIR


# Deploy package
curl -T $OSQP_DEPLOY_DIR.tar.gz -ubstellato:$BINTRAY_API_KEY -H "X-Bintray-Package:OSQP" -H "X-Bintray-Version:0.2.1" https://api.bintray.com/content/bstellato/generic/OSQP/0.2.1/

# Publish
curl -X POST -ubstellato:$BINTRAY_API_KEY https://api.bintray.com/content/bstellato/generic/OSQP/0.2.1/publish

fi


# Anaconda
cd ${TRAVIS_BUILD_DIR}/interfaces/python/conda_recipe

echo "Creating conda package..."
conda build --python ${PYTHON_VERSION} osqp --output-folder conda-bld/

echo "Deploying to Anaconda.org..."
anaconda -t $ANACONDA_TOKEN upload conda-bld/**/osqp-*.tar.bz2 --user oxfordcontrol

echo "Successfully deployed to Anaconda.org."


# NB: Binary Linux packages not supported on Pypi
cd ${TRAVIS_BUILD_DIR}/interfaces/python

if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then

	echo "Creating pip binary package..."
	python setup.py bdist_wheel


	echo "Deploying to Pypi..."
	if [[ "$TEST_PYPI" == "true" ]]; then
	    twine upload --repository testpypi --config-file ../../ci/pypirc -p $PYPI_PASSWORD dist/*     # Test pypi repo
	else
	    twine upload --repository pypi --config-file ../../ci/pypirc -p $PYPI_PASSWORD dist/*         # Main pypi repo
	fi
	echo "Successfully deployed to Pypi"

else if [[ "$TRAVIS_OS_NAME" == "linux" && "$PYTHON_VERSION" == "3.6" ]]; then
	# Choose one python version to upload source distribution (3.6)
	echo "Creating pip source package..."
	python setup.py sdist


	echo "Deploying to Pypi..."
	if [[ "$TEST_PYPI" == "true" ]]; then
		twine upload --repository testpypi --config-file ../../ci/pypirc -p $PYPI_PASSWORD dist/*     # Test pypi repo
	else
		twine upload --repository pypi --config-file ../../ci/pypirc -p $PYPI_PASSWORD dist/*         # Main pypi repo
	fi
	echo "Successfully deployed to Pypi"

fi
fi

exit 0
