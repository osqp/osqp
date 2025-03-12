Python
======

Python interface supports Python 3.8 or newer.

Pip
----

.. code:: bash

   pip install osqp


Sources
---------
You need to install the following (see :ref:`build_from_sources` for more details):

- `GCC compiler <https://gcc.gnu.org/>`_
- A Python 3.8 interpreter or newer, with a recent `pip` library.

.. note::

   **Windows**: You need to install **also** the Visual Studio C++ compiler:

   * Python 3: `Build Tools for Visual Studio <https://visualstudio.microsoft.com/downloads/>`_


Now you are ready to build OSQP python interface from sources. Run the following in your terminal

.. code:: bash

   git clone --recurse-submodules https://github.com/osqp/osqp-python
   cd osqp-python
   pip install .
