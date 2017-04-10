Matlab
======
OSQP Matlab interface required Matlab 2015b or newer.


You can also build OSQP from source as described below.

Binaries
--------

Precompiled platform-dependent Matlab binaries are available on Github:

* Windows: `osqp-0.0.0-matlab-win.tar.gz <https://github.com/oxfordcontrol/osqp/releases/download/v0.0.0/osqp-0.0.0-matlab-win.tar.gz>`_

* Linux `osqp-0.0.0-matlab-linux.tar.gz <https://github.com/oxfordcontrol/osqp/releases/download/v0.0.0/osqp-0.0.0-matlab-linux.tar.gz>`_

* Mac `osqp-0.0.0-matlab-mac.tar.gz <https://github.com/oxfordcontrol/osqp/releases/download/v0.0.0/osqp-0.0.0-matlab-mac.tar.gz>`_

To install the OSQP Matlab binaries, just run the following commands:

.. code:: matlab

    urlwrite('https://github.com/oxfordcontrol/osqp/releases/download/v0.0.0/install_osqp.m','install_osqp.m');
    install_osqp


Sources
-------

You need to install the following (see :ref:`build_from_sources` for more details):

- A supported 64bit `C/C++ compiler <https://www.mathworks.com/support/compilers.html>`_
- `CMake <https://cmake.org/>`_



After you install both, check that your compiler is selected by executing

.. code:: matlab

   mex -setup

.. note::

   **Windows**: If Matlab does not find TDM-GCC, you need to set the environment variable :code:`MW_MINGW64_LOC` as follows

   .. code:: matlab

      setenv('MW_MINGW64_LOC', 'C:\TDM-GCC-64')


   where :code:`C:\TDM-GCC-64` is the installation folder for TDM-GCC.

You can now build the interface by running inside Matlab

.. code:: matlab

   !git clone https://github.com/oxfordcontrol/osqp
   cd osqp/interfaces/matlab
   make_osqp


Then you can add the interface to the search path by executing from the same directory

.. code:: matlab

   addpath(pwd)
   savepath
