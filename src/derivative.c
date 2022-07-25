#include "derivative.h"
#include "lin_alg.h"
#include "util.h"
#include "auxil.h"
#include "lin_sys.h"
#include "proj.h"
#include "error.h"

#ifdef ALGEBRA_DEFAULT
#include "csc_utils.h"
#endif


c_int scale_dxdy(OSQPSolver *solver, OSQPVectorf *dx, OSQPVectorf *dy_l, OSQPVectorf *dy_u) {
    OSQPVectorf_ew_prod(dx, dx, solver->work->scaling->Dinv);
    OSQPVectorf_ew_prod(dy_l, dy_l, solver->work->scaling->Einv);
    OSQPVectorf_mult_scalar(dy_l, solver->work->scaling->c);
    OSQPVectorf_ew_prod(dy_u, dy_u, solver->work->scaling->Einv);
    OSQPVectorf_mult_scalar(dy_u, solver->work->scaling->c);

    return 0;
}

c_int unscale_derivatives_PqAlu(OSQPSolver *solver, csc *dP, OSQPVectorf *dq, csc *dA, OSQPVectorf *dl, OSQPVectorf *du) {

    csc_scale(dP, solver->work->scaling->cinv);
    csc_lmult_diag(dP, OSQPVectorf_data(solver->work->scaling->Dinv));
    csc_rmult_diag(dP, OSQPVectorf_data(solver->work->scaling->Dinv));

    OSQPVectorf_mult_scalar(dq, solver->work->scaling->cinv);
    OSQPVectorf_ew_prod(dq, dq, solver->work->scaling->Dinv);

    csc_lmult_diag(dA, OSQPVectorf_data(solver->work->scaling->Einv));
    csc_rmult_diag(dA, OSQPVectorf_data(solver->work->scaling->Dinv));

    OSQPVectorf_ew_prod(dl, dl, solver->work->scaling->Einv);
    OSQPVectorf_ew_prod(du, du, solver->work->scaling->Einv);

    return 0;
}

c_int adjoint_derivative(OSQPSolver *solver, c_float *dx, c_float *dy_l, c_float *dy_u, csc* dP, c_float* dq, csc* dA, c_float* dl, c_float* du) {
#ifdef ALGEBRA_DEFAULT

    c_int m = solver->work->data->m;
    c_int n = solver->work->data->n;

    OSQPSettings*  settings  = solver->settings;

    OSQPMatrix *P = solver->work->data->P;
    OSQPMatrix *A = solver->work->data->A;
    OSQPVectorf *l = solver->work->data->l;
    OSQPVectorf *u = solver->work->data->u;
    OSQPVectorf *x = OSQPVectorf_new(solver->solution->x, n);  // Note: x/y are unscaled solutions
    OSQPVectorf *y = OSQPVectorf_new(solver->solution->y, m);

    c_float *l_data = OSQPVectorf_data(l);
    c_float *u_data = OSQPVectorf_data(u);
    c_float *y_data = OSQPVectorf_data(y);

    c_int *A_ineq_l_vec = (c_int *) c_malloc(m * sizeof(c_int));
    c_int *A_ineq_u_vec = (c_int *) c_malloc(m * sizeof(c_int));
    c_int *A_eq_vec = (c_int *) c_malloc(m * sizeof(c_int));

    c_int *eq_indices_vec = (c_int *) c_malloc(m * sizeof(c_int));
    c_int *ineq_indices_vec = (c_int *) c_malloc(m * sizeof(c_int));
    c_int *l_noninf_indices_vec = (c_int *) c_malloc(m * sizeof(c_int));
    c_int *u_noninf_indices_vec = (c_int *) c_malloc(m * sizeof(c_int));
    c_int *nu_indices_vec = (c_int *) c_malloc(m * sizeof(c_int));
    c_int *nu_sign_vec = (c_int *) c_malloc(m * sizeof(c_int));

    OSQPVectorf *dxx = OSQPVectorf_new(dx, n);
    OSQPVectorf *dy_l_vec = OSQPVectorf_new(dy_l, m);
    OSQPVectorf *dy_u_vec = OSQPVectorf_new(dy_u, m);

    // TODO: Find out why scaling/unscaling is not working
    //if (solver->settings->scaling) scale_dxdy(solver, dxx, dy_l_vec, dy_u_vec);

    // TODO: We could use constr_type in OSQPWorkspace but it only tells us whether a constraint is 'loose'
    // not 'upper loose' or 'lower loose', which we seem to need here.
    c_float infval = OSQP_INFTY * OSQP_MIN_SCALING;

    c_int n_ineq_l = 0;
    c_int n_ineq_u = 0;
    c_int n_ineq = 0;
    c_int n_eq = 0;

    c_int j;
    for (j = 0; j < m; j++) {
        c_float _l = l_data[j];
        c_float _u = u_data[j];
        if (_l < _u) {
            ineq_indices_vec[n_ineq++] = j;
            A_eq_vec[j] = 0;
            if (_l > -infval) {
                l_noninf_indices_vec[n_ineq_l] = j;
                A_ineq_l_vec[j] = 1;
                n_ineq_l++;
            } else {
                A_ineq_l_vec[j] = 0;
            }
            if (_u < infval) {
                u_noninf_indices_vec[n_ineq_u] = j;
                A_ineq_u_vec[j] = 1;
                n_ineq_u++;
            } else {
                A_ineq_u_vec[j] = 0;
            }
        } else {
            eq_indices_vec[n_eq] = j;
            nu_indices_vec[n_eq] = j;
            A_eq_vec[j] = 1;
            A_ineq_l_vec[j] = 0;
            A_ineq_u_vec[j] = 0;
            if (y_data[j] >= 0) {
                nu_sign_vec[n_eq] = 1;
            } else {
                nu_sign_vec[n_eq] = -1;
            }
            n_eq++;
        }
    }

    OSQPVectori *A_ineq_l_i = OSQPVectori_new(A_ineq_l_vec, m);
    OSQPMatrix *A_ineq_l = OSQPMatrix_submatrix_byrows(A, A_ineq_l_i);
    OSQPMatrix_mult_scalar(A_ineq_l, -1);

    OSQPVectori *A_ineq_u_i = OSQPVectori_new(A_ineq_u_vec, m);
    OSQPMatrix *A_ineq_u = OSQPMatrix_submatrix_byrows(A, A_ineq_u_i);

    OSQPMatrix *G = OSQPMatrix_vstack(A_ineq_l, A_ineq_u);

    OSQPMatrix_free(A_ineq_l);
    OSQPMatrix_free(A_ineq_u);

    OSQPVectori *A_eq_i = OSQPVectori_new(A_eq_vec, m);
    OSQPMatrix *A_eq = OSQPMatrix_submatrix_byrows(A, A_eq_i);

    // --------- lambda
    OSQPVectorf *m_zeros = OSQPVectorf_malloc(m);
    OSQPVectorf_set_scalar(m_zeros, 0);
    OSQPVectorf *y_u = OSQPVectorf_malloc(m);
    OSQPVectorf_ew_max_vec(y_u, y, m_zeros);
    OSQPVectorf *y_l = OSQPVectorf_malloc(m);
    OSQPVectorf_ew_min_vec(y_l, y, m_zeros);
    OSQPVectorf_mult_scalar(y_l, -1);
    OSQPVectorf_free(m_zeros);

    OSQPVectorf *y_l_ineq = OSQPVectorf_subvector_byrows(y_l, A_ineq_l_i);
    OSQPVectorf *y_u_ineq = OSQPVectorf_subvector_byrows(y_u, A_ineq_u_i);
    OSQPVectorf *lambda = OSQPVectorf_concat(y_l_ineq, y_u_ineq);

    OSQPVectorf_free(y_l_ineq);
    OSQPVectorf_free(y_u_ineq);
    // ---------- lambda

    // --------- slacks
    OSQPVectorf *l_ineq = OSQPVectorf_subvector_byrows(l, A_ineq_l_i);
    OSQPVectorf_mult_scalar(l_ineq, -1);
    OSQPVectorf *u_ineq = OSQPVectorf_subvector_byrows(u, A_ineq_u_i);

    OSQPVectorf *h = OSQPVectorf_concat(l_ineq, u_ineq);

    OSQPVectorf_free(l_ineq);
    OSQPVectorf_free(u_ineq);

    OSQPVectorf* slacks = OSQPVectorf_copy_new(h);
    OSQPMatrix_Axpy(G, x, slacks, 1, -1);
    OSQPVectorf_free(h);

    // ---------- GDiagLambda
    OSQPMatrix *GDiagLambda = OSQPMatrix_copy_new(G);
    OSQPMatrix_lmult_diag(GDiagLambda, lambda);

    // ---------- P_full
    OSQPMatrix *P_full = OSQPMatrix_triu_to_symm(P);

    // ---------- RHS
    OSQPVectorf *dy_l_ineq = OSQPVectorf_subvector_byrows(dy_l_vec, A_ineq_l_i);
    OSQPVectorf_free(dy_l_vec);
    OSQPVectorf *dy_u_ineq = OSQPVectorf_subvector_byrows(dy_u_vec, A_ineq_u_i);
    OSQPVectorf_free(dy_u_vec);
    OSQPVectorf *dlambd = OSQPVectorf_concat(dy_l_ineq, dy_u_ineq);
    OSQPVectorf_free(dy_l_ineq);
    OSQPVectorf_free(dy_u_ineq);

    c_float *d_nu_vec = (c_float *) c_malloc(n_eq * sizeof(c_float));
    for (j=0; j<n_eq; j++) {
        if (nu_sign_vec[j]==1) {
            d_nu_vec[j] = dy_u[nu_indices_vec[j]];
        } else if (nu_sign_vec[j]==-1) {
            d_nu_vec[j] = -dy_l[nu_indices_vec[j]];
        } else {}  // should never happen
    }
    OSQPVectorf *d_nu = OSQPVectorf_new(d_nu_vec, n_eq);
    c_free(d_nu_vec);

    OSQPVectorf *rhs_temp1 = OSQPVectorf_concat(dxx, dlambd);
    OSQPVectorf_free(dxx);
    OSQPVectorf_free(dlambd);
    OSQPVectorf *rhs_temp2 = OSQPVectorf_concat(rhs_temp1, d_nu);
    OSQPVectorf_free(rhs_temp1);
    OSQPVectorf_free(d_nu);
    OSQPVectorf_mult_scalar(rhs_temp2, -1);
    OSQPVectorf *zeros = OSQPVectorf_malloc(n + n_ineq_l + n_ineq_u + n_eq);
    OSQPVectorf_set_scalar(zeros, 0);
    OSQPVectorf *rhs = OSQPVectorf_concat(rhs_temp2, zeros);
    OSQPVectorf_free(rhs_temp2);
    OSQPVectorf_free(zeros);

    // ----------- Check
    adjoint_derivative_linsys_solver(solver, settings, P_full, G, A_eq, GDiagLambda, slacks, rhs);

    c_float *rhs_data = OSQPVectorf_data(rhs);

    c_float *r_yl = (c_float *) c_malloc(m * sizeof(c_float));
    c_float *r_yu = (c_float *) c_malloc(m * sizeof(c_float));
    // TODO: We shouldn't have to do this if we assemble r_yl/r_yu judiciously
    for (j=0; j<m; j++) r_yl[j] = 0;
    for (j=0; j<m; j++) r_yu[j] = 0;

    c_int pos = n + n_ineq_l + n_ineq_u + n_eq;
    OSQPVectorf* rx = OSQPVectorf_view(rhs, pos, n);

    pos += n;
    for (j=0; j<n_ineq_l; j++) {
        r_yl[l_noninf_indices_vec[j]] = -rhs_data[pos+j];
    }
    pos += n_ineq_l;
    for (j=0; j<n_ineq_u; j++) {
        r_yu[u_noninf_indices_vec[j]] = rhs_data[pos+j];
    }
    pos += n_ineq_u;
    for (j=0; j<n_eq; j++) {
        if (nu_sign_vec[j]==1) {
            r_yl[eq_indices_vec[j]] = 0;
            r_yu[eq_indices_vec[j]] = rhs_data[pos+j] / y_data[eq_indices_vec[j]];
        } else {
            r_yl[eq_indices_vec[j]] = -rhs_data[pos+j] / y_data[eq_indices_vec[j]];
            r_yu[eq_indices_vec[j]] = 0;
        }

    }

    OSQPVectorf *ryl = OSQPVectorf_new(r_yl, m);
    c_free(r_yl);
    OSQPVectorf_mult(ryl, ryl, y_l);
    OSQPVectorf_mult_scalar(ryl, -1);
    OSQPVectorf *ryu = OSQPVectorf_new(r_yu, m);
    c_free(r_yu);
    OSQPVectorf_mult(ryu, ryu, y_u);

    // Assemble dP/dA
    // TODO: Check for incoming m/n/nzmax compatibility unless we're allocating
    c_float *rx_data = OSQPVectorf_data(rx);
    c_float *x_data = OSQPVectorf_data(x);
    c_float *y_u_data = OSQPVectorf_data(y_u);
    c_float *y_l_data = OSQPVectorf_data(y_l);
    c_float *ryu_data = OSQPVectorf_data(ryu);
    c_float *ryl_data = OSQPVectorf_data(ryl);

    c_int col;
    for (col=0; col<n; col++) {
        c_int p, i;
        for (p=dP->p[col]; p<dP->p[col+1]; p++) {
            i = dP->i[p];
            dP->x[p] = 0.5 * ((rx_data[i] * x_data[col]) + (rx_data[col] * x_data[i]));
        }
        for (p=dA->p[col]; p<dA->p[col+1]; p++) {
            i = dA->i[p];
            dA->x[p] = ((y_u_data[i] - y_l_data[i]) * rx_data[col]) + ((ryu_data[i] - ryl_data[i]) * x_data[col]);
        }
    }

    OSQPVectorf_mult_scalar(ryu, -1);

    // TODO: Find out why scaling/unscaling is not working
    // if (solver->settings->scaling) unscale_derivatives_PqAlu(solver, dP, rx, dA, ryl, ryu);

    // Assign vector derivatives to function arguments
    OSQPVectorf_to_raw(dq, rx);
    OSQPVectorf_to_raw(dl, ryl);
    OSQPVectorf_to_raw(du, ryu);

    // Free up remaining stuff
    OSQPVectorf_free(y_l);
    OSQPVectorf_free(y_u);

    OSQPVectorf_view_free(rx);
    OSQPVectorf_free(ryu);
    OSQPVectorf_free(ryl);

    c_free(l_noninf_indices_vec);
    c_free(u_noninf_indices_vec);
    c_free(nu_indices_vec);
    c_free(nu_sign_vec);

    OSQPMatrix_free(G);
    OSQPMatrix_free(A_eq);
    OSQPMatrix_free(P_full);
    OSQPMatrix_free(GDiagLambda);

    OSQPVectorf_free(lambda);
    OSQPVectorf_free(slacks);

    c_free(A_ineq_l_vec);
    c_free(A_ineq_u_vec);
    c_free(A_eq_vec);

    OSQPVectori_free(A_ineq_l_i);
    OSQPVectori_free(A_ineq_u_i);
    OSQPVectori_free(A_eq_i);

    OSQPVectorf_free(rhs);
    OSQPVectorf_free(x);
    OSQPVectorf_free(y);

    return 0;

#else

    c_eprint("Not implemented");
    return 1;

#endif

}