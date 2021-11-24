.. _install_osqp_libs:

CC++
=====

Binaries
--------

Precompiled platform-dependent shared and static libraries are available on `GitHub <https://github.com/osqp/osqp/releases>`_.
We here assume that the user uncompressed each archive to :code:`OSQP_FOLDER`.

Each archive contains static :code:`OSQP_FOLDER/lib/libosqp.a` and shared :code:`OSQP_FOLDER/lib/libosqp.ext` libraries to be used to interface OSQP to any C/C++ software.
The extension :code:`.ext` is platform dependent and is :code:`.so` for Linux, :code:`.dylib` for Mac and :code:`.dll` for Windows.
The required include files can be found in :code:`OSQP_FOLDER/include`.

Simply compile with the linker option with :code:`-LOSQP_FOLDER/lib` and :code:`-losqp`.

If you are interested in development builds, you can find them on `GitHub <https://github.com/osqp/osqp/releases>`__.

Sources
-------

The OSQP libraries can also be compiled from sources. For more details see :ref:`build_from_sources`.


Including OSQP in CMake projects
--------------------------------
If you compiled OSQP from sources and followed the CMake installation instructions in :ref:`install_the_binaries` section, you can include the package in another CMake project with the following lines depending on whether you need a shared or a static library

.. code::

   # Find OSQP library and headers
   find_package(osqp REQUIRED)

   # Link the OSQP shared library
   target_link_libraries(yourTarget PRIVATE osqp::osqp)

   # or...

   # Link the OSQP static library
   target_link_libraries(yourTarget PRIVATE osqp::osqpstatic)


