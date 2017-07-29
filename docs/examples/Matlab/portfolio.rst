Portfolio optimization
----------------------


Portfolio optimization seeks to allocate assets in a way that maximizes the risk adjusted return,


.. math::
  \begin{array}{ll}
    \mbox{maximize} & \mu^T x - \gamma \left( x^T \Sigma x \right) \\
    \mbox{subject to} & \boldsymbol{1}^T x = 1 \\
                      & x \ge 0
  \end{array}


where :math:`x \in \mathbf{R}^{n}` represents the portfolio, :math:`\mu \in \mathbf{R}^{n}` the vector of expected returns, :math:`\gamma > 0` the risk aversion parameter, and :math:`\Sigma \in \mathbf{S}^{n}_{+}` the risk model covariance matrix.
The risk model is usually assumed to be the sum of a diagonal and a rank :math:`k < n` matrix,


.. math::
  \Sigma = F F^T + D,


where :math:`F \in \mathbf{R}^{n \times k}` is the factor loading matrix and :math:`D \in \mathbf{S}^{n}_{+}` is a diagonal matrix describing the asset-specific risk.
The resulting problem has the following equivalent form,


.. math::
  \begin{array}{ll}
    \mbox{minimize} & \frac{1}{2} x^T D x + \frac{1}{2} y^T y - \frac{1}{2\gamma}\mu^T x \\
    \mbox{subject to} & y = F^T x \\
                      & \boldsymbol{1}^T x = 1 \\
                      & x \ge 0
  \end{array}



.. code:: matlab

    % Define problem data
    P = blkdiag(D, speye(k));
    q = [-mu/(2*gamma); zeros(k, 1)];
    A = [F', -speye(k);
         ones(1, n), zeros(1, k);
         speye(n), sparse(n, k)];
    l = [zeros(k, 1); 1; zeros(n, 1)];
    u = [zeros(k, 1); 1; ones(n, 1)];

    % Create an OSQP object
    prob = osqp;

    % Setup workspace
    prob.setup(P, q, A, l, u);

    % Solve problem
    res = prob.solve();
