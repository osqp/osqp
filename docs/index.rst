.. osqp documentation master file, created by
   sphinx-quickstart on Sat Feb 18 15:49:00 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to OSQP's documentation
================================

OSQP (Operator Splitting Quadratic Program) solver is a numerical optimization package for solving convex quadratic problems in the form


.. math::
  \begin{array}{ll}
    \mbox{minimize} & x^\top P x + q^\top x \\
    \mbox{subject to} & l \leq A x \leq u
  \end{array}

OSQP is written in C and can be used in C, C++, Python and Matlab.

.. todo::

    Add Support for Julia, Java and Scala.

    Add support for CVX, Yalmip, CVXPY, Convex.jl





.. toctree::
   :maxdepth: 2
   :caption: User Documentation
   
   installation/index
