Python Wrappers for OSQP
========================

The Python wrappers for OSQP allow you to interact with the core objects in OSQP
in an intuitive way, with easy switching of algebra backends.

We recommend that you use an environment manager like ``conda`` to create a dedicated
Python environment for the purpose of building the Python wrappers for OSQP.

While building the wrappers, we will be using the following command periodically to
check what algebra backends we have successfully built.

.. code-block:: bash

        $ python -c "from osqp import algebras_available; print(algebras_available())"

Wrappers for ``builtin`` algebra
------------------------------

Building wrappers for the ``builtin`` algebra demonstrates some of the basic steps
you would need to take for all algebra backends.

.. code-block:: bash

	$ conda create --name osqp_build python=3.9
	$ conda activate osqp_build

	# "build" is the python package that is capable of generating binary "wheels"
	# there are other ways of doing this but this way is the most standards compliant
	$ pip install build

	$ git clone git@github.com:osqp/osqp-python.git
	$ cd osqp-python/

	# Build the python wheels for "builtin" algebra
	$ python -m build .

	# Built artifacts are in the "dist" folder, we can "pip install" any of these files
	# installing the wheel entails no (further) compilation so its fastest
	$ tree dist
	dist
	├── osqp-1.0.0b1-cp39-cp39-linux_x86_64.whl    <- compiled wheel
	└── osqp-1.0.0b1.tar.gz                        <- source gz

	$ pip install dist/osqp-1.0.0b1-cp39-cp39-linux_x86_64.whl

	$ python -c "from osqp import algebras_available; print(algebras_available())"
	['builtin']

Wrappers for ``mkl`` algebra
--------------------------

For building Python wrappers for the ``MKL`` algebra backend, you need to have the `Intel OneAPI<https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#base-kit>`_.
library installed and accessible for your platform.

Once installed, set the environment variable ``MKL_ROOT`` to the path of the installed toolkit. For example:

.. code-block:: bash

        $ export MKL_ROOT=/opt/intel/oneapi/mkl/latest

The process of building the ``MKL`` OSQP wrappers is the same as above, except that the environment variable
``OSQP_ALGEBRA_BACKEND`` set to ``mkl`` tells the build process that we wish to build MKL wrappers.

.. code-block:: bash

        $ export OSQP_ALGEBRA_BACKEND=mkl 

	$ python -m build .

	$ tree dist
	dist
	├── osqp-1.0.0b1-cp39-cp39-linux_x86_64.whl
	├── osqp-1.0.0b1.tar.gz
	├── osqp_mkl-1.0.0b1-cp39-cp39-linux_x86_64.whl
	└── osqp_mkl-1.0.0b1.tar.gz

	$ pip install dist/osqp_mkl-1.0.0b1-cp39-cp39-linux_x86_64.whl
	$ python -c "from osqp import algebras_available; print(algebras_available())"
	['mkl', 'builtin']

Wrappers for ``cuda`` algebra
--------------------------

For building Python wrappers for the ``CUDA`` algebra backend, you need to have the `CUDA Tooklit<https://developer.nvidia.com/cuda-toolkit-archive>`_.
installed and accessible for your platform. The build process has been tested out for CUDA Toolkit versions 11.2 through 11.7.

If you install the CUDA Toolkit using the default paths and options (which makes the toolkit available at ``/usr/local/cuda`` for \*nix platforms),
there is no need for any further configuration.

The process of building the ``CUDA`` OSQP wrappers is the same as above, except that the environment variable
``OSQP_ALGEBRA_BACKEND`` set to ``cuda`` tells the build process that we wish to build CUDA wrappers.

.. code-block:: bash

        $ export OSQP_ALGEBRA_BACKEND=cuda

	$ python -m build .

        $ tree dist
	dist
	├── osqp-1.0.0b1-cp39-cp39-linux_x86_64.whl
	├── osqp-1.0.0b1.tar.gz
	├── osqp_cuda-1.0.0b1-cp39-cp39-linux_x86_64.whl
	├── osqp_cuda-1.0.0b1.tar.gz
	├── osqp_mkl-1.0.0b1-cp39-cp39-linux_x86_64.whl
	└── osqp_mkl-1.0.0b1.tar.gz

	$ pip install dist/osqp_cuda-1.0.0b1-cp39-cp39-linux_x86_64.whl
	$ python -c "from osqp import algebras_available; print(algebras_available())"
	['cuda', 'mkl', 'builtin']


Building wrappers against an experimental branch of OSQP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The file ``CMakeLists.txt`` in the root folder of the ``osqp-python`` repository lists the ``osqp`` branch that
is fetched and compiled as part of the build process. The relevant section of this file looks like (at the time
of this writing):

.. code-block::

	FetchContent_Declare(
	  osqp
	  GIT_REPOSITORY https://github.com/osqp/osqp.git
	  GIT_TAG v1.0.0.beta1)

If you wish to build the wrappers against a particular branch or commit of ``osqp`` (for example when adding
wrappers for experimental features not available in the main/master branch of ``osqp``), you can modify the
``GIT_TAG`` property to point to the relevant commit/branch, and run the steps in this document to build the
wrappers against the algebra of your choice.

