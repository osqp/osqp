.. _install_osqp_libs:

CC++
=====

Binaries
--------

Precompiled platform-dependent shared and static libraries are available on `Bintray <https://bintray.com/bstellato/generic/OSQP/0.2.1>`_.
We here assume that the user uncompressed each archive to :code:`OSQP_FOLDER`.

Each archive contains static :code:`OSQP_FOLDER/lib/libosqpstatic.a` and shared :code:`OSQP_FOLER/lib/libosqp.ext` libraries to be used to interface OSQP to any C/C++ software. 
The extension :code:`.ext` is platform dependent and is :code:`.so` for Linux, :code:`.dylib` for Mac and :code:`.dll` for Windows.
The required include files can be found in :code:`OSQP_FOLDER/include`.

Simply compile with the linker option with :code:`-LOSQP_FOLDER/lib` and :code:`-losqp` or :code:`-losqpstatic`. 



Sources
-------

The OSQP libraries can also be compiled from sources. For more details see :ref:`build_from_sources`.


