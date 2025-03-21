Update vectors
==============


Consider the following QP


.. math::
  \begin{array}{ll}
    \mbox{minimize} & \frac{1}{2} x^T \begin{bmatrix}4 & 1\\ 1 & 2 \end{bmatrix} x + \begin{bmatrix}1 \\ 1\end{bmatrix}^T x \\
    \mbox{subject to} & \begin{bmatrix}1 \\ 0 \\ 0\end{bmatrix} \leq \begin{bmatrix} 1 & 1\\ 1 & 0\\ 0 & 1\end{bmatrix} x \leq \begin{bmatrix}1 \\ 0.7 \\ 0.7\end{bmatrix}
  \end{array}



We show below how to setup and solve the problem.
Then we update the vectors :math:`q`, :math:`l`, and :math:`u` and solve the updated problem


.. math::
  \begin{array}{ll}
    \mbox{minimize} & \frac{1}{2} x^T \begin{bmatrix}4 & 1\\ 1 & 2 \end{bmatrix} x + \begin{bmatrix}2 \\ 3\end{bmatrix}^T x \\
    \mbox{subject to} & \begin{bmatrix}2 \\ -1 \\ -1\end{bmatrix} \leq \begin{bmatrix} 1 & 1\\ 1 & 0\\ 0 & 1\end{bmatrix} x \leq \begin{bmatrix}2 \\ 2.5 \\ 2.5\end{bmatrix}
  \end{array}
  


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

    # Setup workspace
    prob.setup(P, q, A, l, u)

    # Solve problem
    res = prob.solve()

    # Update problem
    q_new = np.array([2, 3])
    l_new = np.array([2, -1, -1])
    u_new = np.array([2, 2.5, 2.5])
    prob.update(q=q_new, l=l_new, u=u_new)

    # Solve updated problem
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

    % Setup workspace
    prob.setup(P, q, A, l, u);

    % Solve problem
    res = prob.solve();

    % Update problem
    q_new = [2; 3];
    l_new = [2; -1; -1];
    u_new = [2; 2.5; 2.5];
    prob.update('q', q_new, 'l', l_new, 'u', u_new);

    % Solve updated problem
    res = prob.solve();



Julia
------

.. code:: julia

    using OSQP
    using SparseArrays

    # Define problem data
    P = sparse([4. 1.; 1. 2.])
    q = [1.; 1.]
    A = sparse([1. 1.; 1. 0.; 0. 1.])
    l = [1.; 0.; 0.]
    u = [1.; 0.7; 0.7]

    # Crate OSQP object
    prob = OSQP.Model()

    # Setup workspace
    OSQP.setup!(prob; P=P, q=q, A=A, l=l, u=u)

    # Solve problem
    results = OSQP.solve!(prob)

    # Update problem
    q_new = [2.; 3.]
    l_new = [2.; -1.; -1.]
    u_new = [2.; 2.5; 2.5]
    OSQP.update!(prob, q=q_new, l=l_new, u=u_new)

    # Solve updated problem
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

    # Setup workspace
    model <- osqp(P, q, A, l, u)

    # Solve problem
    res <- model$Solve()

    # Update problem
    q_new <- c(2., 3.)
    l_new <- c(2., -1., -1.)
    u_new <- c(2., 2.5, 2.5)
    model$Update(q = q_new, l = l_new, u = u_new)

    # Solve updated problem
    res <- model$Solve()



C
-

.. code:: c

    #include <stdlib.h>
    #include "osqp.h"

    int main(int argc, char **argv) {
        /* Load problem data */
        OSQPFloat P_x[3] = {4.0, 1.0, 2.0, };
        OSQPInt P_nnz = 3;
        OSQPInt P_i[3] = {0, 0, 1, };
        OSQPInt P_p[3] = {0, 1, 3, };
        OSQPFloat q[2] = {1.0, 1.0, };
        OSQPFloat q_new[2] = {2.0, 3.0, };
        OSQPFloat A_x[4] = {1.0, 1.0, 1.0, 1.0, };
        OSQPInt A_nnz = 4;
        OSQPInt A_i[4] = {0, 1, 0, 2, };
        OSQPInt A_p[3] = {0, 2, 4, };
        OSQPFloat l[3] = {1.0, 0.0, 0.0, };
        OSQPFloat l_new[3] = {2.0, -1.0, -1.0, };
        OSQPFloat u[3] = {1.0, 0.7, 0.7, };
        OSQPFloat u_new[3] = {2.0, 2.5, 2.5, };
        OSQPInt n = 2;
        OSQPInt m = 3;

        /* Exitflag */
        OSQPInt exitflag = 0;

        /* Solver */
        OSQPSolver *solver;

        /* Create CSC matrices that are backed by the above data arrays. */
        OSQPCscMatrix* P = OSQPCscMatrix_new(n, n, P_nnz, P_x, P_i, P_p);
        OSQPCscMatrix* A = OSQPCscMatrix_new(m, n, A_nnz, A_x, A_i, A_p);

        /* Set default settings */
        OSQPSettings *settings = OSQPSettings_new();

        /* Setup solver */
        exitflag = osqp_setup(&solver, P, q, A, l, u, m, n, settings);

        /* Solve problem */
        if (!exitflag) exitflag = osqp_solve(solver);

        /* Update problem */
        if (!exitflag) exitflag = osqp_update_data_vec(solver, q_new, l_new, u_new);

        /* Solve updated problem */
        if (!exitflag) exitflag = osqp_solve(solver);

        /* Cleanup */
        osqp_cleanup(solver);
        OSQPCscMatrix_free(A);
        OSQPCscMatrix_free(P);
        OSQPSettings_free(settings);

        return (int)exitflag;
    };



Rust
----

.. code:: rust

    use osqp::{CscMatrix, Problem, Settings};

    fn main() {
        // Define problem data
        let P = &[[4.0, 1.0],
            [1.0, 2.0]];
        let q = &[1.0, 1.0];
        let A = &[[1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0]];
        let l = &[1.0, 0.0, 0.0];
        let u = &[1.0, 0.7, 0.7];

        // Extract the upper triangular elements of `P`
        let P = CscMatrix::from(P).into_upper_tri();

        let settings = Settings::default();

        // Create an OSQP problem
        let mut prob = Problem::new(P, q, A, l, u, &settings).expect("failed to setup problem");

        // Solve problem
        let result = prob.solve();

        // Update problem
        prob.update_lin_cost(&[2.0, 3.0]);
        prob.update_bounds(&[2.0, -1.0, -1.0], &[2.0, 2.5, 2.5]);

        // Solve updated problem
        let result = prob.solve();

        // Print the solution
        println!("{:?}", result.x().expect("failed to solve problem"));
    }