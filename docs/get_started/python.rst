Python
======

Python interface supports Python 3.8 or newer.

Pip
----

Install :code:`osqp` with the *default algebra* backend using :code:`pip`:

.. code:: bash

   pip install osqp

The :code:`builtin` algebra backend is always available for use. Alternative :code:`osqp` algebra backends - :code:`mkl` or :code:`cu12` as of the time of this writing, can also be installed:

To install :code:`osqp` with the *mkl* backend:

.. code:: bash

   pip install osqp[mkl]

To install :code:`osqp` with the *cu12* (Cuda 12.x) backend:

.. code:: bash

   pip install osqp[cu12]

To install :code:`osqp` with the *mkl* and *cu12* backends:

.. code:: bash

   pip install osqp[mkl,cu12]

.. note::

   These commands install osqp with the *mkl* or *cu12* "extras", which provide the :code:`osqp-mkl` or :code:`osqp-cuda` packages respectively.
   These extension modules are directly importable using :code:`import osqp_mkl` or :code:`import osqp_cuda`, though you will never directly need to do this.

   See :ref:`Algebra Backends -> Python <backends_python>` for more information on how to use these backends.