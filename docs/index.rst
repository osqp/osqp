OSQP solver documentation
==========================

`Visit our GitHub Discussions page <https://github.com/orgs/osqp/discussions>`_
for any questions related to the solver!

The OSQP (Operator Splitting Quadratic Program) solver is a numerical
optimization package for solving convex quadratic programs in the form

.. math::
  \begin{array}{ll}
    \mbox{minimize} & \frac{1}{2} x^T P x + q^T x \\
    \mbox{subject to} & l \le A x \le u
  \end{array}

where :math:`x \in \mathbf{R}^n` is the optimization variable and
:math:`P \in \mathbf{S}^{n}_{+}` a positive semidefinite matrix.

**Code available on** `GitHub <https://github.com/osqp/osqp>`_.

.. rubric:: Citing OSQP

If you are using OSQP for your work, we encourage you to

* :ref:`Cite the related papers <citing>`
* Put a star on GitHub |github-star|


.. |github-star| image:: https://img.shields.io/github/stars/osqp/osqp.svg?style=social&label=Star
  :target: https://github.com/osqp/osqp


**We are looking forward to hearing your success stories with OSQP!** Please `share them with us <bartolomeo.stellato@gmail.com>`_.

.. rubric:: Features


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
        It requires no external library to run.

    Efficiently warm started
        It can be easily warm-started and the matrix factorization can be cached to solve parametrized problems extremely efficiently.

    Interfaces
        It provides interfaces to C, C++, Fortran, Julia, Matlab, Python, R, Ruby, and Rust.



.. rubric:: License

OSQP is distributed under the `Apache 2.0 License <https://www.apache.org/licenses/LICENSE-2.0>`_



.. rubric:: Credits

The following people have been involved in the development of OSQP:

* `Bartolomeo Stellato <https://stellato.io/>`_ (Princeton University): main development
* `Goran Banjac <https://github.com/gbanjac>`_ (ETH Zürich): main development
* `Nicholas Moehle <https://www.nicholasmoehle.com/>`_ (Stanford University): methods, maths, and code generation
* `Paul Goulart <https://users.ox.ac.uk/~engs1373/>`_ (University of Oxford): methods, maths, and Matlab interface
* `Alberto Bemporad <http://cse.lab.imtlucca.it/~bemporad/>`_ (IMT Lucca): methods and maths
* `Stephen Boyd <https://web.stanford.edu/~boyd/>`_ (Stanford University): methods and maths
* `Ian McInerney <https://ism.engineer>`_ (Imperial College London): software engineering, code generation
* `Vineet Bansal <https://researchcomputing.princeton.edu/about/people-directory/vineet-bansal>`_ (Princeton University): software engineering
* `Michel Schubiger <mailto:michel.schubiger@bluewin.ch>`_ (Schindler R&D): GPU implementation
* `John Lygeros <https://control.ee.ethz.ch/people/profile.john-lygeros.html>`_ (ETH Zurich): methods and maths
* `Amit Solomon <mailto:as3993@princeton.edu>`_ (Princeton University): software engineering

Interfaces development

* `Nick Gould <https://www.numerical.rl.ac.uk/people/nick-gould>`_ (Rutherford Appleton Laboratory): Fortran and CUTEst interfaces
* `Ed Barnard <eabarnard@gmail.com>`_ (University of Oxford): Rust interface


.. rubric:: Bug reports and support

Please report any issues via the `Github issue tracker <https://github.com/osqp/osqp/issues>`_. All types of issues are welcome including bug reports, documentation typos, feature requests and so on.


.. rubric:: Numerical benchmarks

Numerical benchmarks against other solvers are available `here <https://github.com/osqp/osqp_benchmarks>`_.


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Documentation

   solver/index
   get_started/index
   interfaces/index
   parsers/index
   codegen/index
   examples/index
   advanced/index
   get_started/migration_guide
   contributing/index
   citing/index
