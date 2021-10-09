Setup and solve
===============


Consider the following QP


.. math::
  \begin{array}{ll}
    \mbox{minimize} & \frac{1}{2} x^T \begin{bmatrix}4 & 1\\ 1 & 2 \end{bmatrix} x + \begin{bmatrix}1 \\ 1\end{bmatrix}^T x \\
    \mbox{subject to} & \begin{bmatrix}1 \\ 0 \\ 0\end{bmatrix} \leq \begin{bmatrix} 1 & 1\\ 1 & 0\\ 0 & 1\end{bmatrix} x \leq  \begin{bmatrix}1 \\ 0.7 \\ 0.7\end{bmatrix}
  \end{array}



We show below how to solve the problem in Python, Matlab, Julia and C.



Python
------

.. code:: python

    import osqp
    import numpy as np
    from scipy import sparse

    # Define problem data
    P = sparse.csc_matrix([[4, 1], [1, 2]])
    q = np.array([1, 1])
    A = sparse.csc_matrix([[1, 1], [1, 0], [0, 1]])
    l = np.array([1, 0, 0])
    u = np.array([1, 0.7, 0.7])

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace and change alpha parameter
    prob.setup(P, q, A, l, u, alpha=1.0)

    # Solve problem
    res = prob.solve()



Matlab
------

.. code:: matlab

    % Define problem data
    P = sparse([4, 1; 1, 2]);
    q = [1; 1];
    A = sparse([1, 1; 1, 0; 0, 1]);
    l = [1; 0; 0];
    u = [1; 0.7; 0.7];

    % Create an OSQP object
    prob = osqp;

    % Setup workspace and change alpha parameter
    prob.setup(P, q, A, l, u, 'alpha', 1);

    % Solve problem
    res = prob.solve();



Julia
------

.. code:: julia

    using OSQP
    using Compat.SparseArrays

    # Define problem data
    P = sparse([4. 1.; 1. 2.])
    q = [1.; 1.]
    A = sparse([1. 1.; 1. 0.; 0. 1.])
    l = [1.; 0.; 0.]
    u = [1.; 0.7; 0.7]

    # Crate OSQP object
    prob = OSQP.Model()

    # Setup workspace and change alpha parameter
    OSQP.setup!(prob; P=P, q=q, A=A, l=l, u=u, alpha=1)

    # Solve problem
    results = OSQP.solve!(prob)



R
-

.. code:: r

    library(osqp)
    library(Matrix)

    # Define problem data
    P <- Matrix(c(4., 1.,
                  1., 2.), 2, 2, sparse = TRUE)
    q <- c(1., 1.)
    A <- Matrix(c(1., 1., 0.,
                  1., 0., 1.), 3, 2, sparse = TRUE)
    l <- c(1., 0., 0.)
    u <- c(1., 0.7, 0.7)

    # Change alpha parameter and setup workspace
    settings <- osqpSettings(alpha = 1.0)
    model <- osqp(P, q, A, l, u, settings)

    # Solve problem
    res <- model$Solve()



C
-

.. code:: c

    #include <stdlib.h>
    #include "osqp.h"

    int main(int argc, char **argv) {
        /* Load problem data */
        c_float P_x[3] = {4.0, 1.0, 2.0, };
        c_int P_nnz = 3;
        c_int P_i[3] = {0, 0, 1, };
        c_int P_p[3] = {0, 1, 3, };
        c_float q[2] = {1.0, 1.0, };
        c_float A_x[4] = {1.0, 1.0, 1.0, 1.0, };
        c_int A_nnz = 4;
        c_int A_i[4] = {0, 1, 0, 2, };
        c_int A_p[3] = {0, 2, 4, };
        c_float l[3] = {1.0, 0.0, 0.0, };
        c_float u[3] = {1.0, 0.7, 0.7, };
        c_int n = 2;
        c_int m = 3;

        /* Exitflag */
        c_int exitflag = 0;

        /* Solver, settings, matrices */
        OSQPSolver   *solver;
        OSQPSettings *settings;
        csc *P = malloc(sizeof(csc));
        csc *A = malloc(sizeof(csc));

        /* Populate matrices */
        csc_set_data(A, m, n, A_nnz, A_x, A_i, A_p);
        csc_set_data(P, n, n, P_nnz, P_x, P_i, P_p);

        /* Set default settings */
        settings = (OSQPSettings *)malloc(sizeof(OSQPSettings));
        if (settings) {
            osqp_set_default_settings(settings);
            settings->alpha = 1.0; /* Change alpha parameter */
        }

        /* Setup solver */
        exitflag = osqp_setup(&solver, P, q, A, l, u, m, n, settings);

        /* Solve problem */
        if (!exitflag) exitflag = osqp_solve(solver);

        /* Cleanup */
        osqp_cleanup(solver);
        if (A) free(A);
        if (P) free(P);
        if (settings) free(settings);

        return (int)exitflag;
    };
