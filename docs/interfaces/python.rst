Python
======

Import
------
The OSQP module is can be imported with

.. code:: python

    import osqp

The solver is initialized by creating an OSQP object

.. code:: python

    m = osqp.OSQP()


Setup
-----
The problem is specified in the setup phase by running

.. code:: python

    m.setup(P=P, q=q, A=A, l=l, u=u, **settings)


The arguments :code:`q`, :code:`l` and :code:`u` are numpy arrays. The elements of :code:`l` and :code:`u` can be :math:`\pm \infty` (:code:`+numpy.inf` or :code:`-numpy.inf`).

The arguments :code:`P` and :code:`A` are scipy sparse matrices in CSC format. If they are sparse matrices are in another format, the interface will attemp to convert them. There is no need to specify all the arguments.


The keyword arguments


Solve
-----

Update
------
