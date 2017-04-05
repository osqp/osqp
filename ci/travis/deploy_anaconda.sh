set -e

# Make sure we have the right anaconda path
export PATH=${DEPS_DIR}/miniconda/bin:$PATH
cd ${TRAVIS_BUILD_DIR}/interfaces/python/conda_recipe

echo "Creating conda package..."
conda build osqp --output-folder conda-bld/

echo "Deploying to Anaconda.org..."
anaconda -t $ANACONDA_TOKEN upload conda-bld/**/osqp-*.tar.bz2

echo "Successfully deployed to Anaconda.org."
exit 0
