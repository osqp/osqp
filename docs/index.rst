Welcome to OSQP's documentation
================================

The OSQP (Operator Splitting Quadratic Program) solver is a numerical optimization package for solving convex quadratic programs in the form


.. math::
  \begin{array}{ll}
    \mbox{minimize} & \frac{1}{2} x^\top P x + q^\top x \\
    \mbox{subject to} & l \leq A x \leq u
  \end{array}

where :math:`x\in\mathbf{R}^{n}` is the optimization variable. The objective function is defined by a positive semidefinite matrix :math:`P \in \mathbf{S}^{n}_{+}` and vector :math:`q\in \mathbf{R}^{n}`. The linear constraints are defined by matrix :math:`A\in\mathbf{R}^{m \times n}` and vectors :math:`l \in \mathbf{R}^{m} \cup \{-\infty\}^{m}`, :math:`u \in \mathbf{R}^{m} \cup \{+\infty\}^{m}`.


OSQP is written in C and can be used in C, C++, Python and Matlab.


.. todo::

    Add Support for Julia, Java and Scala.

    Add support for CVX, Yalmip, CVXPY, Convex.jl



.. todo::

    .. Citing OSQP
    .. -----------
    .. If you are using OSQP for your work, you can cite the following article:

    .. .. code:: latex

        @article{aurhors:17,
        }


    Add link to article and complete bibtex entry


License
-------
OSQP is distributed under the `Apache 2.0 License <https://www.apache.org/licenses/LICENSE-2.0>`_


.. toctree::
   :maxdepth: 2
   :caption: User Documentation

   installation/index
   interfaces/index
   parsers/index
   codegen/index
   examples/index
