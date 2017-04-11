The solver
===========

.. Cite the paper

Problem statement
-----------------

OSQP solves convex quadratic programs (QPs) of the form

.. math::
  \begin{array}{ll}
    \mbox{minimize} & \frac{1}{2} x^T P x + q^T x \\
    \mbox{subject to} & l \leq A x \leq u
  \end{array}

where :math:`x\in\mathbf{R}^{n}` is the optimization variable.
The objective function is defined by a positive semidefinite matrix
:math:`P \in \mathbf{S}^{n}_{+}` and vector :math:`q\in \mathbf{R}^{n}`.
The linear constraints are defined by matrix :math:`A\in\mathbf{R}^{m \times n}`
and vectors :math:`l \in \mathbf{R}^{m} \cup \{-\infty\}^{m}`,
:math:`u \in \mathbf{R}^{m} \cup \{+\infty\}^{m}`.


Solution and convergence
-------------------------

The solver runs an `ADMM algorithm <http://web.stanford.edu/~boyd/papers/admm_distr_stats.html>`_  producing at each iteration :math:`k` a tuple :math:`(x^{k}, z^{k}, y^{k})` with :math:`x^{k} \in \mathbf{R}^{n}` and :math:`z^{k}, y^{k} \in \mathbf{R}^{m}`.

The primal and and dual residuals associated to :math:`(x^{k}, z^{k}, y^{k})` are

.. math::

   \begin{align}
   r_{\rm prim}^{k} &= Ax^{k} - z^{k}\\
   r_{\rm dual}^{k} &= Px^{k} + q + A^T y^{k}
   \end{align}


If the problem is feasible, the residuals converge to zero as :math:`k\to\infty`. The algorithm stops when the norms of :math:`r_{\rm prim}^{k}` and :math:`r_{\rm dual}^{k}` are within the specified tolerances.


Primal/dual infeasible problems
-------------------------------

OSQP is able to detect if the problem is primal or dual infeasible.

When the problem is primal infeasible, the algorithm generates a certificate of infeasibility, *i.e.*, a vector :math:`v\in\mathbf{R}^{m}` such that

.. math::

   A^T v = 0, \quad u^T v_{+} + l^T v_{-} < 0,

where :math:`v_+=\max(v,0)` and :math:`v_-=\min(v,0)`.

When the problem is dual infeasible, OSQP generates a vector :math:`s\in\mathbf{R}^{n}` being a certificate of dual infeasibility, *i.e.*,

.. math::

   P s = 0, \quad q^T s < 0, \quad (As)_i = \begin{cases} 0 & l_i \in \mathbf{R}, u_i\in\mathbf{R} \\ \ge 0 & l_i\in\mathbf{R}, u_i=+\infty \\ \le 0 & u_i\in\mathbf{R}, l_i=-\infty \end{cases}
