Matlab
======
OSQP Matlab interface requires Matlab 2015b or newer.


Binaries
--------

Precompiled platform-dependent Matlab binaries are available on `GitHub <https://github.com/osqp/osqp-matlab/releases>`_.

To install the interface, just run the following commands:

.. code:: matlab

    websave('install_osqp.m','https://raw.githubusercontent.com/osqp/osqp-matlab/master/package/install_osqp.m');
    install_osqp


Sources
-------

You need to install the following (see :ref:`build_from_sources` for more details):

- A supported 64bit `C/C++ compiler <https://www.mathworks.com/support/requirements/supported-compilers.html>`_
- `CMake <https://cmake.org/>`_



After you install both, check that your compiler is selected by executing

.. code:: matlab

   mex -setup

.. note::

   **Windows**: If Matlab does not find TDM-GCC, you need to set the environment variable :code:`MW_MINGW64_LOC` as follows

   .. code:: matlab

      setenv('MW_MINGW64_LOC', 'C:\TDM-GCC-64')


   where :code:`C:\\TDM-GCC-64` is the installation folder for TDM-GCC.

You can now build the interface by running inside Matlab

.. code:: matlab

   !git clone --recurse-submodules https://github.com/osqp/osqp-matlab
   osqp.build('osqp_mex')


You are now ready to start using the Matlab interface.
