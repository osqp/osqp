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
	twine upload --repository pypi --config-file ../../ci/pypirc -p $PYPI_PASSWORD dist/*         # Main pypi repo
	# twine upload --repository testpypi --config-file ../../ci/pypirc -p $PYPI_PASSWORD dist/*     # Test pypi repo
	echo "Successfully deployed to Pypi"

else if [[ "$TRAVIS_OS_NAME" == "linux" && "$PYTHON_VERSION" == "3.6" ]]; then
	# Choose one python version to upload source distribution (3.6)
	echo "Creating pip source package..."
	python setup.py sdist


	echo "Deploying to Pypi..."
	twine upload --repository pypi --config-file ../../ci/pypirc -p $PYPI_PASSWORD dist/*         # Main pypi repo
	# twine upload --repository testpypi --config-file ../../ci/pypirc -p $PYPI_PASSWORD dist/*     # Test pypi repo
	echo "Successfully deployed to Pypi"

fi
fi

exit 0
