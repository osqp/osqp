.. _c_interface:

C
=====


.. _C_main_API:

Main solver API
---------------

The main C API is imported from the header :code:`osqp.h` and provides the following functions


.. doxygenfunction:: osqp_setup

.. doxygenfunction:: osqp_solve

.. doxygenfunction:: osqp_cleanup


.. _C_sublevel_API:

Sublevel API
------------
Sublevel C API is also imported from the header :code:`osqp.h` and provides the following functions

Warm start
^^^^^^^^^^
OSQP automatically warm starts primal and dual variables from the previous QP solution. If you would like to warm start their values manually, you can use

.. doxygenfunction:: osqp_warm_start


.. _C_update_data :

Update problem data
^^^^^^^^^^^^^^^^^^^
Problem data can be updated without executing the setup again using the following functions.

.. doxygenfunction:: osqp_update_data_vec

.. doxygenfunction:: osqp_update_data_mat



.. _C_data_types :

Data types
----------

The most basic used datatypes are

* :code:`c_int`: can be :code:`long` or :code:`int` if the compiler flag :code:`DLONG` is set or not
* :code:`c_float`: can be a :code:`float` or a :code:`double` if the compiler flag :code:`DFLOAT` is set or not.


The matrices are defined in `Compressed Sparse Column (CSC) format <https://people.sc.fsu.edu/~jburkardt/data/cc/cc.html>`_ using zero-based indexing.

.. doxygenstruct:: csc
   :members:


The relevant structures used in the API are

Solver
^^^^^^^^

.. doxygenstruct:: OSQPSolver
  :members:

Settings
^^^^^^^^

.. doxygenstruct:: OSQPSettings
  :members:

Solution
^^^^^^^^

.. doxygenstruct:: OSQPSolution
   :members:

Info
^^^^^

.. doxygenstruct:: OSQPInfo
   :members:



.. TODO: Add sublevel API
.. TODO: Add using your own linear system solver
