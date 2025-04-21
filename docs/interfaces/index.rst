.. _interfaces:

Interfaces
============

OSQP has several interfaces. The information about settings, status values and how to assign different linear system solvers appear in the following links

* :ref:`Solver settings <solver_settings>`
* :ref:`Linear system solvers <linear_system_solvers_setting>`
* :ref:`Status values <status_values>`



.. toctree::
   :maxdepth: 1
   :glob:
   :hidden:

   solver_settings.rst
   linear_systems_solvers.rst
   status_values.rst


Official Interfaces
-------------------

+------------------------------------+----------------------------------------------------------+------------------------------------------------------------------------------------------+
| Language                           | Maintainers                                              | Repository                                                                               |
+====================================+==========================================================+==========================================================================================+
| :ref:`C <c_interface>`             | | `Bartolomeo Stellato <bartolomeo.stellato@gmail.com>`_ | `github.com/osqp/osqp <https://github.com/osqp/osqp>`_                                   |
|                                    | | `Goran Banjac <gbanjac@control.ee.ethz.ch>`_           |                                                                                          |
|                                    | | `Paul Goulart <paul.goulart@eng.ox.ac.uk>`_            |                                                                                          |
+------------------------------------+----------------------------------------------------------+------------------------------------------------------------------------------------------+
| :ref:`Python <python_interface>`   | | `Bartolomeo Stellato <bartolomeo.stellato@gmail.com>`_ | `github.com/osqp/osqp-python <https://github.com/osqp/osqp-python>`_                     |
|                                    | | `Goran Banjac <gbanjac@control.ee.ethz.ch>`_           |                                                                                          |
+------------------------------------+----------------------------------------------------------+------------------------------------------------------------------------------------------+
| :ref:`Matlab <matlab_interface>`   | | `Bartolomeo Stellato <bartolomeo.stellato@gmail.com>`_ | `github.com/osqp/osqp-matlab <https://github.com/osqp/osqp-matlab>`_                     |
|                                    | | `Goran Banjac <gbanjac@control.ee.ethz.ch>`_           |                                                                                          |
|                                    | | `Paul Goulart <paul.goulart@eng.ox.ac.uk>`_            |                                                                                          |
+------------------------------------+----------------------------------------------------------+------------------------------------------------------------------------------------------+
| :ref:`Julia <julia_interface>`     | | `Twan Koolen <tkoolen@mit.edu>`_                       | `github.com/osqp/OSQP.jl <https://github.com/osqp/OSQP.jl>`_                             |
|                                    | | `Benoît Legat <benoit.legat@uclouvain.be>`_            |                                                                                          |
|                                    | | `Bartolomeo Stellato <bartolomeo.stellato@gmail.com>`_ |                                                                                          |
+------------------------------------+----------------------------------------------------------+------------------------------------------------------------------------------------------+
| :ref:`R <rlang_interface>`         | | `Bartolomeo Stellato <bartolomeo.stellato@gmail.com>`_ | `github.com/osqp/osqp-r <https://github.com/osqp/osqp-r>`_                               |
|                                    | | `Paul Goulart <paul.goulart@eng.ox.ac.uk>`_            |                                                                                          |
+------------------------------------+----------------------------------------------------------+------------------------------------------------------------------------------------------+



.. toctree::
   :maxdepth: 1
   :glob:
   :hidden:

   C.rst
   python.rst
   matlab.rst
   julia.rst
   rlang.rst




Community Maintained Interfaces
-------------------------------


+------------------------------------------------+----------------------------------------------------------+------------------------------------------------------------------------------------------+
| Language                                       | Maintainers                                              | Repository                                                                               |
+================================================+==========================================================+==========================================================================================+
| :ref:`C++/Eigen Google <eigen_google>`         | | `Miles Lubin <miles.lubin@gmail.com>`_                 | `github.com/google/osqp-cpp <https://github.com/google/osqp-cpp>`_                       |
+------------------------------------------------+----------------------------------------------------------+------------------------------------------------------------------------------------------+
| :ref:`C++/Eigen Robotology <eigen_robotology>` | | `Giulio Romualdi <giulio.romualdi@gmail.com>`_         | `github.com/robotology/osqp-eigen <https://github.com/robotology/osqp-eigen>`_           |
+------------------------------------------------+----------------------------------------------------------+------------------------------------------------------------------------------------------+
| :ref:`Rust <rust_interface>`                   | | `Ed Barnard <eabarnard@gmail.com>`_                    | `github.com/osqp/osqp.rs <https://github.com/osqp/osqp.rs>`_                             |
+------------------------------------------------+----------------------------------------------------------+------------------------------------------------------------------------------------------+
| :ref:`Ruby <ruby_interface>`                   | | `Andrew Kane <andrew@chartkick.com>`_                  | `https://github.com/ankane/osqp <https://github.com/ankane/osqp>`_                       |
+------------------------------------------------+----------------------------------------------------------+------------------------------------------------------------------------------------------+
| :ref:`Fortran <fortran_interface>`             | | `Nick Gould <nick.gould@stfc.ac.uk>`_                  | `github.com/osqp/osqp-fortran <https://github.com/osqp/osqp-fortran>`_                   |
|                                                | | `Bartolomeo Stellato <bartolomeo.stellato@gmail.com>`_ |                                                                                          |
|                                                | | `Paul Goulart <paul.goulart@eng.ox.ac.uk>`_            |                                                                                          |
+------------------------------------------------+----------------------------------------------------------+------------------------------------------------------------------------------------------+
| :ref:`Cutest <cutest_interface>`               | | `Nick Gould <nick.gould@stfc.ac.uk>`_                  | `github.com/ralna/CUTEst <https://github.com/ralna/CUTEst/tree/master/src/osqp>`_        |
+------------------------------------------------+----------------------------------------------------------+------------------------------------------------------------------------------------------+



.. toctree::
   :maxdepth: 1
   :glob:
   :hidden:

   eigen_google.rst
   eigen_robotology.rst
   rust.rst
   ruby.rst
   fortran.rst
   cutest.rst


