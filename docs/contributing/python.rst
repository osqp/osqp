Python Wrappers for OSQP
========================

The Python wrappers for OSQP allow you to interact with the core objects in OSQP
in an intuitive way, with easy switching of algebra backends.

We recommend that you use an environment manager to create a dedicated
Python environment for the purpose of building the Python wrappers for OSQP.

While building the wrappers, we will be using the following command periodically to
check what algebra backends we have successfully built.

.. code-block:: bash

        $ python -c "from osqp import algebras_available; print(algebras_available())"

See :ref:`Building alternative algebra backends <python_build_algebras>` on how to build and test
different OSQP algebra backends.

Building from source
--------------------
You need to install the following (see :ref:`build_from_sources` for more details):

- `GCC compiler <https://gcc.gnu.org/>`_
- A Python 3.8 interpreter or newer, with a recent `pip` library.

.. note::

   **Windows**: You need to install **also** the Visual Studio C++ compiler:

   * Python 3: `Build Tools for Visual Studio <https://visualstudio.microsoft.com/downloads/>`_

Cloning the repo and :code:`pip` installing it directly should work. Just make sure you have a recent version of :code:`pip` (:code:`pip install pip --upgrade`).

.. code:: bash

   git clone https://github.com/osqp/osqp-python.git
   cd osqp-python
   pip install .

.. _python_build_algebras :

Building alternative algebra backends (advanced)
------------------------------------------------

The following instructions illustrate how you can use the :code:`osqp` :code:`git` source tree to build :code:`osqp` wheels that support different algebra backends. These are probably not needed for most users.

The procedure has been tested out on Linux. The process should be identical on MacOS and Windows, though of course you will likely not be able to to build :code:`osqp-cuda` wheels on MacOS, due to missing :code:`cuda-toolkit` on Mac platforms.

Create a dedicated environment to build :code:`osqp`. Checkout and :code:`cd` into the code.

.. code:: bash

   git clone https://github.com/osqp/osqp-python.git
   cd osqp-python

Build wheels for `builtin` backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    python -m pip wheel . --no-deps --wheel-dir dist

The resulting wheel is available in the :code:`dist` folder, and can be installed using :code:`pip install dist/osqp-1.*.whl`

.. code:: bash

    dist
    ├── osqp-1.0.0b2.dev68+gcaa6446-cp39-cp39-linux_x86_64.whl


The default wheel we built provides us with the :code:`osqp` module, and we can check that we have the :code:`built-in` algebra available:

.. code:: bash

    python -c "from osqp import algebras_available; print(algebras_available())"
    ['builtin']


Build wheels for `mkl` backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The process assumes that the :code:`mkl` development library has been installed, and environment variables set so that the build process can discover these libraries (in particular, you will likely need to set :code:`MKL_ROOT`, which on our machine is set to :code:`opt/intel/oneapi/mkl/latest`). When working on a cluster, you may want to look for a `module <https://hpc-wiki.info/hpc/Modules>`__ that populates the necessary environment variables (on our clusters, we do a :code:`module load intel-mkl/2024.0`, for example).

.. note::

    If you're using a `conda` environment to build `osqp`, one way to get `MKL` is to try (`osqp` uses this in its CI):

    .. code::

        conda install -c https://software.repos.intel.com/python/conda/ mkl-devel

    On Windows:

    ..code::

        conda install -c https://software.repos.intel.com/python/conda/ dpcpp_impl_win-64


However you decide to get :code:`MKL` for your platform, the next step is to build the wrappers for the :code:`mkl` backend:

.. code:: bash

    python -m pip wheel backend/mkl --no-deps --wheel-dir dist

The resulting wheel is available in the :code:`dist` folder, and can be installed using :code:`pip install dist/osqp_mkl-1.*.whl`

.. code:: bash

    dist
    ├── osqp-1.0.0b2.dev68+gcaa6446-cp39-cp39-linux_x86_64.whl
    └── osqp_mkl-1.0.0b2.dev68+gcaa6446-cp39-cp39-linux_x86_64.whl

The mkl wheel we built provides us with the :code:`osqp_mkl` module. When we try to do this:

.. code:: bash

    $ python -c "import osqp_mkl"
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
    ImportError: libmkl_rt.so.2: cannot open shared object file: No such file or directory


This is because we need the :code:`mkl` libraries available to python at runtime to be able to use :code:`osqp-mkl`. There are many ways for users to do this, so we don't enforce an :code:`mkl` dependency in :code:`pip` to install the :code:`osqp_mkl` wheels. An easy way to do this in conda would be :code:`conda install anaconda::mkl`, for example.

**Note that while its possible to do an `import osqp_mkl` in python, we'll never import that module directly in our code**, and just use `import osqp`. We can check that we have the `mkl` algebra in `osqp` available:

.. code:: bash

    python -c "from osqp import algebras_available; print(algebras_available())"
    ['mkl', 'builtin']

Note that :code:`mkl` appears *before* :code:`builtin`, and will be the preferred backend for all :code:`osqp` operations. We can verify this by running:

.. code:: bash

    python -c "from osqp import default_algebra; print(default_algebra())"
    mkl

This behavior can be overridden by setting the :code:`OSQP_ALGEBRA_BACKEND` environment variable (which can take the values :code:`builtin`, :code:`mkl`, or :code:`cuda`).

.. code:: bash

    OSQP_ALGEBRA_BACKEND=builtin python -c "from osqp import default_algebra; print(default_algebra())"
    builtin

Build wheels for `cuda` backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This assumes you have the :code:`cuda-toolkit` installed, and available at :code:`/usr/local/cuda`. When working on a cluster, you may want to look for a `module <https://hpc-wiki.info/hpc/Modules>`__ that populates the necessary environment variables (on our clusters, we do a :code:`module load cudatoolkit/12.4`, for example).

.. code:: bash

    python -m pip wheel backend/cuda --no-deps --wheel-dir dist


The resulting wheel is available in the `dist` folder, and can be installed using `pip install dist/osqp_cuda-1.*.whl`

.. code:: bash

    dist
    ├── osqp-1.0.0b2.dev68+gcaa6446-cp39-cp39-linux_x86_64.whl
    ├── osqp_cuda-1.0.0b2.dev68+gcaa6446-cp39-cp39-linux_x86_64.whl
    └── osqp_mkl-1.0.0b2.dev68+gcaa6446-cp39-cp39-linux_x86_64.whl

The cuda wheel we built provides us with the :code:`osqp_cuda` module, and we can check that we have the :code:`cuda` algebra available:

.. code:: bash

    python -c "from osqp import algebras_available; print(algebras_available())"
    ['cuda', 'mkl', 'builtin']

Again, the default algebra can be overridden with the :code:`OSQP_ALGEBRA_BACKEND` environment variable.

Install developer dependencies and run tests
--------------------------------------------

Finally, to test that :code:`osqp` is installed/working correct with all available algebras, run the tests.

.. code:: bash

    pip install .[dev]
    pytest

The tests run across all available algebras. The :code:`OSQP_ALGEBRA_BACKEND` environment variable does not need to be set, and has no effect for the tests. If the :code:`mkl` backend is available, then the tests are run for both the :code:`"direct"` and :code:`"indirect"` *modes* of :code:`mkl`.

To pick exactly what algebras are tested, read on.

.. note::

    Tests that use the :code:`mkl` backend and the :code:`indirect` mode are slow to run on head nodes of clusters, where cpu cores are a shared resource and thus cannot be monopolized. Tests involving the :code:`cuda` algebra may not be possible to run on head nodes of clusters anyway (because of lack of GPUs there). Also, :code:`codegen` tests in :code:`osqp` require an internet connection to work properly, which compute nodes may or may not have (because the generated code gets compiled using :code:`python+cmake`, which wants to fetch certain modules..)

    For these and other unforeseen scenarios, fine-tuning of test parametrization is supported using the :code:`OSQP_TEST_ALGEBRA_INCLUDE` and :code:`OSQP_TEST_ALGEBRA_SKIP` environment variables (both optional). These variables can take space-delimited values that include :code:`builtin`, :code:`mkl-direct`, :code:`mkl-indirect`, and :code:`cuda`. Of course, this can be combined with the :code:`-k` pattern selection that :code:`pytest` itself supports.

    For example:

    - Run all tests for all available algebras:

    .. code:: bash

        pytest

    - Run all tests, but only for the :code:`builtin` and :code:`mkl-direct` algebras (if available):

    .. code:: bash

        OSQP_TEST_ALGEBRA_INCLUDE="builtin mkl-direct" pytest

    - Run all tests, but skip the :code:`cuda` algebra:

    .. code:: bash

        OSQP_TEST_ALGEBRA_SKIP="cuda" pytest

    - Run all tests, but only for the :code:`builtin` algebra, and skip the :code:`codegen` tests:

    .. code:: bash

        OSQP_TEST_ALGEBRA_INCLUDE="builtin" pytest -k "not codegen"

Building wrappers against an experimental branch of OSQP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The file ``CMakeLists.txt`` in the root folder of the ``osqp-python`` repository lists the ``osqp`` branch that
is fetched and compiled as part of the build process. The relevant section of this file looks like (at the time
of this writing):

.. code-block::

	FetchContent_Declare(
	  osqp
	  GIT_REPOSITORY https://github.com/osqp/osqp.git
	  GIT_TAG v1.0.0)

If you wish to build the wrappers against a particular branch or commit of ``osqp`` (for example when adding
wrappers for experimental features not available in the main/master branch of ``osqp``), you can modify the
``GIT_TAG`` property to point to the relevant commit/branch, and run the steps in this document to build the
wrappers against the algebra of your choice.

