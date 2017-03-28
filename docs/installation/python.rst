Python
======

Python interface supports Python 2.7 and 3.4 or newer on Linux and Mac. Windows version is supports only Python 2.7 at the moment.

Anaconda
----------

.. code:: bash

   conda install -c oxfordcontrol osqp


Pip
----

.. code:: bash

   pip install osqp


Sources
---------
You need to install the following (see :ref:`build_from_sources` for more details):

- `GCC compiler <https://gcc.gnu.org/>`_
- `CMake <https://cmake.org/>`_

.. note::

   **Windows**: You need to install **also** the `Visual Studio C++ compiler for Python <https://www.microsoft.com/en-us/download/details.aspx?id=44266>`_


Now you are ready to build OSQP python interface from sources. Run the following in your terminal

.. code:: bash

   git clone https://github.com/oxfordcontrol/osqp
   cd osqp/interfaces/python
   python setup.py install
