.. _backends_python:

Python
------

Checking which backends are installed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the :code:`algebras_available` function to check what algebra backends you have available locally:

.. code-block:: bash

        $ python -c "from osqp import algebras_available; print(algebras_available())"
        ['cuda', 'mkl', 'builtin']

.. note::

   Installing alternative algebras like *mkl* or *cu12* **does not automatically** install the runtime MKL or Cuda libraries for your system.
   If you have installed osqp with the *mkl* or *cu12* extras but do not see the expected algebra(s) available, you may need to install these libraries separately, and ensure that they are available to the Python interpreter at runtime.

   For example, to use *mkl*, you might need to run:

   .. code-block:: bash

       $ pip install mkl-devel && LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.venv/lib

   .. code-block:: bash

       $ conda install conda-forge::mkl

   To use *cuda*, you might need to install the `Cuda toolkit <https://developer.nvidia.com/cuda-toolkit-archive>`_
   for your system, or (more easily) for your python environment:

   .. code-block:: bash

       pip install nvidia-cublas-cu12 nvidia-cusparse-cu12 nvidia-cuda-runtime-cu12
       LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.venv/lib/nvidia/cusparse/lib

   .. code-block:: bash

       conda install conda-forge::cudatoolkit

   The exact path to the libraries will vary depending on your system. If unsure, you can try to directly import the
   extension modules :code:`osqp_mkl` or :code:`osqp_cuda` to see what additional runtime libraries are needed:

    .. code-block:: bash

         $ python -c "import osqp_mkl"
         Traceback (most recent call last):
           File "<string>", line 1, in <module>
         ImportError: libmkl_rt.so.2: cannot open shared object file: No such file or directory

         $ python -c "import osqp_cuda"
         Traceback (most recent call last):
           File "<string>", line 1, in <module>
         ImportError: libcublas.so.2: cannot open shared object file: No such file or directory

   If you're using osqp in a cluster environment, it might be sufficient to just activate a module that provides the necessary libraries. For example, on our clusters, we do a :code:`module load intel-mkl/2024.0` or :code:`module load cudatoolkit/12.6` to get the necessary libraries.

Switching algebra backends
^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the :code:`default_algebra` function to check what algebra backend is currently set as the default:

.. code-block:: bash

        $ python -c "from osqp import default_algebra; print(default_algebra())"
        cuda

By default, :code:`osqp` uses the *best* algebra that is available (:code:`cuda` being preferred over :code:`mkl`, which is preferred over :code:`builtin`).
The default algebra can be overridden by setting the :code:`OSQP_ALGEBRA_BACKEND` environment variable to one of :code:`builtin`, :code:`mkl`, or :code:`cuda`.

.. code-block:: bash

        $ OSQP_ALGEBRA_BACKEND=builtin python -c "from osqp import default_algebra; print(default_algebra())"
        builtin

You can set this environment variable in your shell, or in your python script, before importing :code:`osqp`.
Alternatively, you can specify the :code:`algebra` argument to :code:`osqp.OSQP` to use a specific algebra backend for a particular problem.

.. code-block:: bash

        $ python -c "from osqp import OSQP; problem = OSQP(algebra='mkl'); print(problem.algebra)"
        mkl

        $ python -c "from osqp import OSQP; problem = OSQP(algebra='builtin'); print(problem.algebra)"
        builtin
