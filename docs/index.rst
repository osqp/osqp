OSQP solver documentation
==========================
**Join our** `forum <https://groups.google.com/forum/#!forum/osqp>`_ **for any
questions related to the solver!**

The OSQP (Operator Splitting Quadratic Program) solver is a numerical
optimization package for solving convex quadratic programs in the form

.. math::
  \begin{array}{ll}
    \mbox{minimize} & \frac{1}{2} x^T P x + q^T x \\
    \mbox{subject to} & l \leq A x \leq u
  \end{array}

where :math:`x` is the optimization variable and
:math:`P \in \mathbf{S}^{n}_{+}` a positive semidefinite matrix.


Features
--------

.. glossary::

    Efficient
        It uses a custom ADMM-based first-order method requiring only a single matrix factorization in the setup phase. All the other operations are extremely cheap. It also implements custom sparse linear algebra routines exploiting structures in the problem data.

    Robust
        The algorithm is absolutely division free after the setup and it requires no assumptions on problem data (the problem only needs to be convex). It just works!

    Detects primal / dual infeasible problems
        When the problem is primal or dual infeasible, OSQP detects it. It is the first available QP solver based on first-order methods able to do so.

    Embeddable
        It has an easy interface to generate customized embeddable C code with no memory manager required.

    Library-free
        It requires no external library to run. Only the setup phase requires the AMD and SparseLDL from Timothy A. Davis that are already included in the sources.

    Efficiently warm started
        It can be easily warm-started and the matrix factorization can be cached to solve parametrized problems extremely efficiently.

    Interfaces
        It can be interfaced to C, C++, Python and Matlab.



License
-------
OSQP is distributed under the `Apache 2.0 License <https://www.apache.org/licenses/LICENSE-2.0>`_



Credits
-------

The following people have been involved in the development of OSQP:


* `Bartolomeo Stellato <https://bstellato.github.io/>`_ (University of Oxford): main development
* `Goran Banjac <http://users.ox.ac.uk/~sedm4978/>`_ (University of Oxford): main development
* `Paul Goulart <http://users.ox.ac.uk/~engs1373/>`_ (University of Oxford): methods, maths and Matlab interface
* `Alberto Bemporad <http://cse.lab.imtlucca.it/~bemporad/>`_ (IMT Lucca): methods and maths
* `Stephen Boyd <http://web.stanford.edu/~boyd/>`_ (Stanford University): methods and maths



.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Documentation

   solver/index
   installation/index
   interfaces/index
   parsers/index
   codegen/index
   examples/index
   citing/index
