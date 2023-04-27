Huber fitting
=============

*Huber fitting* or the *robust least-squares problem* performs linear regression under the assumption that there are outliers in the data.
The fitting problem is written as

.. math::
  \begin{array}{ll}
    \mbox{minimize} & \sum_{i=1}^{m} \phi_{\rm hub}(a_i^T x - b_i),
  \end{array}

with the Huber penalty function :math:`\phi_{\rm hub}:\mathbf{R}\to\mathbf{R}` defined as

.. math::
  \phi_{\rm hub}(u) =
  \begin{cases}
      u^2         & |u| \le 1 \\
      (2|u| - 1)  & |u| > 1
  \end{cases}

The problem has the following equivalent form (see `here <https://doi.org/10.1109/34.877518>`_ for more details)

.. math::
  \begin{array}{ll}
    \mbox{minimize}   & u^T u + 2\,\boldsymbol{1}^T (r+s) \\
    \mbox{subject to} & Ax - b - u = r - s \\
                      & r \ge 0 \\
                      & s \ge 0
  \end{array}



Python
------

.. code:: python

    import osqp
    import numpy as np
    import scipy as sp
    from scipy import sparse

    # Generate problem data
    sp.random.seed(1)
    n = 10
    m = 100
    Ad = sparse.random(m, n, density=0.5, format='csc')
    x_true = np.random.randn(n) / np.sqrt(n)
    ind95 = (np.random.rand(m) < 0.95).astype(float)
    b = Ad.dot(x_true) + np.multiply(0.5*np.random.randn(m), ind95) \
        + np.multiply(10.*np.random.rand(m), 1. - ind95)

    # OSQP data
    Im = sparse.eye(m)
    P = sparse.block_diag([sparse.csc_matrix((n, n)), 2*Im,
                           sparse.csc_matrix((2*m, 2*m))],
                           format='csc')
    q = np.append(np.zeros(m+n), 2*np.ones(2*m))
    A = sparse.bmat([[Ad,   -Im,   -Im,   Im],
                     [None,  None,  Im,   None],
                     [None,  None,  None, Im]], format='csc')
    l = np.hstack([b, np.zeros(2*m)])
    u = np.hstack([b, np.inf*np.ones(2*m)])

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace
    prob.setup(P, q, A, l, u)

    # Solve problem
    res = prob.solve()



Matlab
------

.. code:: matlab

    % Generate problem data
    rng(1)
    n = 10;
    m = 100;
    Ad = sprandn(m, n, 0.5);
    x_true = randn(n, 1) / sqrt(n);
    ind95 = rand(m, 1) > 0.95;
    b = Ad*x_true + 10*rand(m, 1).*ind95 + 0.5*randn(m, 1).*(1-ind95);

    % OSQP data
    Im = speye(m);
    Om = sparse(m, m);
    Omn = sparse(m, n);
    P = blkdiag(sparse(n, n), 2*Im, sparse(2*m, 2*m));
    q = [zeros(m + n, 1); 2*ones(2*m, 1)];
    A = [Ad,  -Im, -Im, Im;
         Omn,  Om,  Im, Om;
         Omn,  Om,  Om, Im];
    l = [b; zeros(2*m, 1)];
    u = [b; inf*ones(2*m, 1)];

    % Create an OSQP object
    prob = osqp;

    % Setup workspace
    prob.setup(P, q, A, l, u);

    % Solve problem
    res = prob.solve();



CVXPY
-----

.. code:: python

    from cvxpy import *
    import numpy as np
    import scipy as sp
    from scipy import sparse

    # Generate problem data
    sp.random.seed(1)
    n = 10
    m = 100
    A = sparse.random(m, n, density=0.5, format='csc')
    x_true = np.random.randn(n) / np.sqrt(n)
    ind95 = (np.random.rand(m) < 0.95).astype(float)
    b = A.dot(x_true) + np.multiply(0.5*np.random.randn(m), ind95) \
        + np.multiply(10.*np.random.rand(m), 1. - ind95)

    # Define problem
    x = Variable(n)
    objective = sum(huber(A*x - b))

    # Solve with OSQP
    Problem(Minimize(objective)).solve(solver=OSQP)



YALMIP
------

.. code:: matlab

    % Generate problem data
    rng(1)
    n = 10;
    m = 100;
    A = sprandn(m, n, 0.5);
    x_true = randn(n, 1) / sqrt(n);
    ind95 = rand(m, 1) > 0.95;
    b = A*x_true + 10*rand(m, 1).*ind95 + 0.5*randn(m, 1).*(1-ind95);

    % Define problem
    x = sdpvar(n, 1);
    objective = huber(A*x - b);

    % Solve with OSQP
    options = sdpsettings('solver', 'osqp');
    optimize([], objective, options);
