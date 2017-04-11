YALMIP
=======

Least-squares problem
---------------------

You can easily define and solve the following least-squares program  in YALMIP

.. math::
  \begin{array}{ll}
    \mbox{minimize} & \|Ax - b\|_2 \\
    \mbox{subject to} & 0 \leq x \leq 1
  \end{array}

The code is the following

.. code:: matlab

   % Problem data.
   rng(1)
   m = 30;
   n = 20;
   A = sprandn(m, n, 0.7);
   b = randn(m, 1);


   % Define problem
   x = sdpvar(n, 1);
   objective = norm(A*x - b)^2;
   constraints = [ 0 <= x <= 1];

   % Solve with OSQP
   options = sdpsettings('solver','osqp');
   optimize(constraints, objective, options);

   % Get optimal primal and dual solution
   x_opt = value(x);
   y_opt = dual(constraints(1));
