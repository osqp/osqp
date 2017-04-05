.. _c_cpp_interface:

C/C++
=====

Once the sources are built as in :ref:`build_from_sources`, the generated static :code:`build/out/libosqpdirstatic.a` and shared :code:`build/out/libosqpdir.ext` libraries can be used to interface any C/C++ software to OSQP. Simply compile with the linker option with :code:`-L(PATH_TO_OSQP)/build/out` and :code:`-losqpdir` or :code:`-losqpdirstatic`. Note that the :code:`osqp_demo_direct` example already performs the required linking using the CMake directives. See the file :code:`CMakeLists.txt` in the root folder for more details.



Main solver API
---------------

The main C/C++ API is imported from the header :code:`osqp.h` and provides the following functions


.. doxygenfunction:: osqp_setup

.. doxygenfunction:: osqp_solve

.. doxygenfunction:: osqp_cleanup


Data types
----------

The most basic used datatypes are

.. doxygentypedef:: c_int
.. doxygentypedef:: c_float

:code:`c_int` can be :code:`long` or :code:`int` if the compiler flag :code:`DLONG` is set or not. :code:`c_float` can be a :code:`float` or a :code:`double` if the compiler flag :code:`DFLOAT` is set or not.



The relevant structures used in the API are

Data
^^^^

.. doxygenstruct:: OSQPData
   :members:

The matrices are defined in `Compressed Sparse Column (CSC) format <https://people.sc.fsu.edu/~jburkardt/data/cc/cc.html>`_.

.. doxygenstruct:: csc
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

Workspace
^^^^^^^^^

.. doxygenstruct:: OSQPWorkspace
   :members:


Scaling
^^^^^^^

.. doxygenstruct:: OSQPScaling
   :members:

Polish
^^^^^^
.. doxygenstruct:: OSQPPolish
  :members:



.. TODO: Add sublevel API
.. TODO: Add using your own linear system solver
