#include "derivative.h"
#include "lin_alg.h"
#include "util.h"
#include "auxil.h"
#include "lin_sys.h"
#include "proj.h"
#include "error.h"
#include "csc_utils.h"


c_int adjoint_derivative(OSQPSolver *solver, c_float *dx, c_float *dy_l, c_float *dy_u, const csc* check1, const c_float* check2) {

    c_int m = solver->work->data->m;
    c_int n = solver->work->data->n;

    OSQPInfo*      info      = solver->info;
    OSQPSettings*  settings  = solver->settings;
    OSQPWorkspace* work      = solver->work;
    OSQPSolution*  solution  = solver->solution;

    OSQPMatrix *P = solver->work->data->P;
    OSQPVectorf *q = solver->work->data->q;
    OSQPMatrix *A = solver->work->data->A;
    OSQPVectorf *l = solver->work->data->l;
    OSQPVectorf *u = solver->work->data->u;
    OSQPVectorf *x = solver->work->x;
    OSQPVectorf *y = solver->work->y;

    c_float *l_data = OSQPVectorf_data(l);
    c_float *u_data = OSQPVectorf_data(u);
    c_float *y_data = OSQPVectorf_data(y);

    c_int *A_ineq_l_vec = (c_int *) c_malloc(m * sizeof(c_int));
    c_int *A_ineq_u_vec = (c_int *) c_malloc(m * sizeof(c_int));
    c_int *A_eq_vec = (c_int *) c_malloc(m * sizeof(c_int));
    c_int *nu_vec = (c_int *) c_malloc(m * sizeof(c_int));

    // TODO: We could use constr_type in OSQPWorkspace but it only tells us whether a constraint is 'loose'
    // not 'upper loose' or 'lower loose', which we seem to need here.
    c_float infval = OSQP_INFTY;  // TODO: Should we be multiplying this by OSQP_MIN_SCALING ?

    c_int n_ineq = 0;
    c_int n_eq = 0;
    c_int j;
    for (j = 0; j < m; j++) {
        c_float _l = l_data[j];
        c_float _u = u_data[j];
        if (_l < _u) {
            A_eq_vec[j] = 0;
            if (_l > -infval)
                A_ineq_l_vec[j] = 1;
            else
                A_ineq_l_vec[j] = 0;
            if (_u < infval)
                A_ineq_u_vec[j] = 1;
            else
                A_ineq_u_vec[j] = 0;
            n_ineq = n_ineq + 2;  // we add two constraints, (one lower, one upper) for every inequality constraint
            nu_vec[j] = -1;
        } else {
            A_eq_vec[j] = 1;
            A_ineq_l_vec[j] = 0;
            A_ineq_u_vec[j] = 0;
            n_eq++;
            if (y_data[j] >= 0) {
                nu_vec[j] = 1;
            } else {
                nu_vec[j] = 0;
            }
        }
    }


    OSQPVectori *A_ineq_l_i = OSQPVectori_malloc(m);
    OSQPVectori_from_raw(A_ineq_l_i, A_ineq_l_vec);
    c_free(A_ineq_l_vec);
    OSQPMatrix *A_ineq_l = OSQPMatrix_submatrix_byrows(A, A_ineq_l_i);
    OSQPMatrix_mult_scalar(A_ineq_l, -1);

    OSQPVectori *A_ineq_u_i = OSQPVectori_malloc(m);
    OSQPVectori_from_raw(A_ineq_u_i, A_ineq_u_vec);
    c_free(A_ineq_u_vec);
    OSQPMatrix *A_ineq_u = OSQPMatrix_submatrix_byrows(A, A_ineq_u_i);

    c_int A_ineq_l_nnz = OSQPMatrix_get_nz(A_ineq_l);
    c_int A_ineq_u_nnz = OSQPMatrix_get_nz(A_ineq_u);
    OSQPMatrix *G = OSQPMatrix_vstack(A_ineq_l, A_ineq_u);

    OSQPMatrix_free(A_ineq_l);
    OSQPMatrix_free(A_ineq_u);

    OSQPVectori *A_eq_i = OSQPVectori_malloc(m);
    OSQPVectori_from_raw(A_eq_i, A_eq_vec);
    c_free(A_eq_vec);
    OSQPMatrix *A_eq = OSQPMatrix_submatrix_byrows(A, A_eq_i);

    OSQPVectorf *zero = OSQPVectorf_malloc(m);
    OSQPVectorf_set_scalar(zero, 0);

    // --------- lambda
    OSQPVectorf *_y_l_ineq = OSQPVectorf_subvector_byrows(y, A_ineq_l_i);
    OSQPVectorf *y_l_ineq = OSQPVectorf_malloc(OSQPVectorf_length(_y_l_ineq));
    OSQPVectorf_ew_min_vec(y_l_ineq, _y_l_ineq, zero);
    OSQPVectorf_free(_y_l_ineq);
    OSQPVectorf_mult_scalar(y_l_ineq, -1);

    OSQPVectorf *_y_u_ineq = OSQPVectorf_subvector_byrows(y, A_ineq_u_i);
    OSQPVectorf *y_u_ineq = OSQPVectorf_malloc(OSQPVectorf_length(_y_u_ineq));
    OSQPVectorf_ew_max_vec(y_u_ineq, _y_u_ineq, zero);
    OSQPVectorf_free(_y_u_ineq);

    OSQPVectorf *lambda = OSQPVectorf_concat(y_l_ineq, y_u_ineq);

    OSQPVectorf_free(y_l_ineq);
    OSQPVectorf_free(y_u_ineq);
    // ---------- lambda

    // --------- h
    OSQPVectorf *l_ineq = OSQPVectorf_subvector_byrows(l, A_ineq_l_i);
    OSQPVectorf_mult_scalar(l_ineq, -1);
    OSQPVectorf *u_ineq = OSQPVectorf_subvector_byrows(u, A_ineq_u_i);

    OSQPVectorf *h = OSQPVectorf_concat(l_ineq, u_ineq);

    OSQPVectorf_free(l_ineq);
    OSQPVectorf_free(u_ineq);
    // ---------- h

    OSQPVectorf_free(zero);

    // ---------- GDiagLambda
    OSQPMatrix *GDiagLambda = OSQPMatrix_copy_new(G);
    OSQPMatrix_lmult_diag(GDiagLambda, lambda);

    // ---------- Slacks
    OSQPVectorf* slacks = OSQPVectorf_copy_new(h);
    OSQPMatrix_Axpy(G, x, slacks, 1, -1);

    OSQPMatrix *P_full = OSQPMatrix_triu_to_symm(P);

    // ---------- RHS
    OSQPVectorf *dxx = OSQPVectorf_malloc(n);
    OSQPVectorf_from_raw(dxx, dx);
    OSQPVectorf *dy_l_vec = OSQPVectorf_malloc(m);
    OSQPVectorf_from_raw(dy_l_vec, dy_l);
    OSQPVectorf *dy_u_vec = OSQPVectorf_malloc(m);
    OSQPVectorf_from_raw(dy_u_vec, dy_u);

    OSQPVectorf *dy_l_ineq = OSQPVectorf_subvector_byrows(dy_l_vec, A_ineq_l_i);
    OSQPVectorf_free(dy_l_vec);
    OSQPVectorf *dy_u_ineq = OSQPVectorf_subvector_byrows(dy_u_vec, A_ineq_u_i);
    OSQPVectorf_free(dy_u_vec);
    OSQPVectorf *dlambd = OSQPVectorf_concat(dy_l_ineq, dy_u_ineq);

    c_float *d_nu_vec = (c_int *) c_malloc(n_eq * sizeof(c_int));
    for (j=0; j<n_eq; j++) {
        if (nu_vec[j]==0) {
            d_nu_vec[j] = dy_u[j];
        } else if (nu_vec[j]==1) {
            d_nu_vec[j] = -dy_l[j];
        } else {}
    }
    c_free(nu_vec);
    OSQPVectorf *d_nu = OSQPVectorf_malloc(n_eq);
    OSQPVectorf_from_raw(d_nu, d_nu_vec);
    c_free(d_nu_vec);

    OSQPVectorf *rhs_temp = OSQPVectorf_concat(dxx, dlambd);
    OSQPVectorf *rhs = OSQPVectorf_concat(rhs_temp, d_nu);
    OSQPVectorf_mult_scalar(rhs, -1);
    OSQPVectorf_free(rhs_temp);
    OSQPVectorf *zeros = OSQPVectorf_malloc(n + n_ineq + n_eq);
    OSQPVectorf_set_scalar(zeros, 0);
    rhs = OSQPVectorf_concat(rhs, zeros);
    OSQPVectorf_free(zeros);

    // ----------- Check
    OSQPMatrix *checkmat1 = OSQPMatrix_new_from_csc(check1, 1);
    OSQPVectorf *checkvec2 = OSQPVectorf_new(check2, 2*(n+n_ineq+n_eq));
    c_int status = adjoint_derivative_linsys_solver(solver, settings, P_full, G, A_eq, GDiagLambda, slacks, rhs, checkmat1, checkvec2);

    c_int status2 = OSQPVectorf_is_eq(rhs, checkvec2, 0.0001);
    status = status && status2;

    // TODO: Make sure we're freeing everything we should!
    OSQPMatrix_free(G);
    OSQPMatrix_free(A_eq);
    OSQPMatrix_free(P_full);
    OSQPMatrix_free(GDiagLambda);

    OSQPVectorf_free(lambda);
    OSQPVectorf_free(slacks);

    OSQPVectori_free(A_ineq_l_i);
    OSQPVectori_free(A_ineq_u_i);
    OSQPVectori_free(A_eq_i);

    OSQPVectorf_free(dxx);
    OSQPVectorf_free(dlambd);
    OSQPVectorf_free(d_nu);

    OSQPVectorf_free(rhs);


    return status;
}