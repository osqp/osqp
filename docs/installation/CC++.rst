C/C++
=====
The main compilation directives are specified using `CMake <https://cmake.org/>`_. The software is supposed to be compiled using `gcc <https://gcc.gnu.org/>`_.

Linux
-----

Almost all linux distributions come with :code:`gcc` and :code:`cmake` commands. Thus, to generate the :code:`Makefiles` and to compile the project it is just necessary to run

.. code-block:: bash

   mkdir build
   cd build
   cmake -G "Unix Makefiles" ..
   make


Mac
---








Windows
-------
