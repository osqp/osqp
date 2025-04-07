.. _linear_system_solvers_installation :

Linear System Solvers
======================

The linear system solver is a core part of the OSQP algorithm.
Depending on the problem instance, different linear system solvers can greatly speedup or reduce the computation time of OSQP.
To set the preferred linear system solver, see :ref:`linear_system_solvers_setting`.

Dynamic shared library loading
------------------------------
OSQP dynamically loads the shared libraries related to the selected external solver. Thus, there is no need to link it at compile time.
The only requirement is that the shared library related to the solver is in the library path of the operating system

+------------------+---------------------------+----------------+
| Operating system | Path variable             | Extension      |
+==================+===========================+================+
| Linux            | :code:`LD_LIBRARY_PATH`   | :code:`.so`    |
+------------------+---------------------------+----------------+
| Mac              | :code:`DYLD_LIBRARY_PATH` | :code:`.dylib` |
+------------------+---------------------------+----------------+
| Windows          | :code:`PATH`              | :code:`.dll`   |
+------------------+---------------------------+----------------+





QDLDL
---------------
OSQP comes with `QDLDL <https://github.com/osqp/qdldl>`_ internally installed.
It does not require any external shared library.
QDLDL is a sparse direct solver that works well for most small to medium sized problems.
However, it becomes not really efficient for large scale problems since it is not multi-threaded.


oneMKL Pardiso
--------------
`oneMKL Pardiso <https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/onemkl-pardiso-parallel-direct-sparse-solver-iface.html>`_ is an efficient multi-threaded linear system solver that works well for large scale problems part of the Intel Math Kernel Library.
Intel offers `free licenses <https://www.intel.com/content/www/us/en/developer/articles/tool/onemkl-license-faq.html>`_ for oneMKL for most non-commercial applications.

Install with MKL
^^^^^^^^^^^^^^^^
We can install MKL Pardiso by using the standard `MKL installer <https://software.intel.com/en-us/mkl>`_.
The main library to be loaded is called :code:`libmkl_rt`.
To add it, together with its dependencies, to your path, just execute the automatic MKL script.

+------------------+------------------------------------------------+
| Operating system | Script                                         |
+==================+================================================+
| Linux            | :code:`source $MKLROOT/bin/mklvars.sh intel64` |
+------------------+------------------------------------------------+
| Mac              | :code:`source $MKLROOT/bin/mklvars.sh intel64` |
+------------------+------------------------------------------------+
| Windows          | :code:`%MKLROOT%\mklvars.bat intel64`          |
+------------------+------------------------------------------------+

where :code:`MKLROOT` is the MKL installation directory.

Install with Anaconda
^^^^^^^^^^^^^^^^^^^^^
`Anaconda Python distribution <https://www.anaconda.com/download/>`_ comes with the intel MKL libraries preinstalled including MKL Pardiso.
To use this version, the Anaconda libraries folders have to be in your system path.
Anaconda environments should add them automatically so in most cases you do not have to do anything. If you get an error where OSQP cannot find MKL, you can add the right path by adding the output from the following command to your path variable:

.. code::

   echo "`ls -rd ${CONDA_ROOT}/pkgs/*/ | grep mkl-2 | head -n 1`lib:`ls -rd ${CONDA_ROOT}/pkgs/*/ | grep intel-openmp- | head -n 1`lib"


where :code:`CONDA_ROOT` is the root of your anaconda installation.

