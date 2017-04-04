.. _c_cpp_interface:

C/C++
=====

Once the sources are built as in :ref:`build_from_sources`, the generated static :code:`build/out/libosqpdirstatic.a` and shared :code:`build/out/libosqpdir.ext` libraries can be used to interface any C/C++ software to OSQP. Simply compile with the linker option with :code:`-L(PATH_TO_OSQP)/build/out` and :code:`-losqpdir` or :code:`-losqpdirstatic`. Note that the :code:`osqp_demo_direct` example already performs the required linking using the CMake directives. See the file :code:`CMakeLists.txt` in the root folder for more details.



Main solver API
---------------

The main C/C++ API is imported from the header :code:`osqp.h`.


.. doxygenfunction:: osqp_setup

.. doxygenfunction:: osqp_solve

.. doxygenfunction:: osqp_cleanup


OSQP Structures
---------------

.. doxygenstruct:: OSQPWorkspace
   :project: osqp
   :members:



Sublevel API
------------
