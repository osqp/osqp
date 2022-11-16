#include "derivative.h"
#include "lin_alg.h"
#include "util.h"
#include "auxil.h"
#include "lin_alg.h"
#include "error.h"
#include "printing.h"

#include "csc_utils.h"
#include "csc_math.h"

OSQPInt unscale_PAlu(OSQPSolver*  solver,
                     OSQPMatrix*  P,
                     OSQPMatrix*  A,
                     OSQPVectorf* l,
                     OSQPVectorf* u) {

  OSQPMatrix_mult_scalar(P, solver->work->scaling->cinv);
  OSQPMatrix_lmult_diag(P, solver->work->scaling->Dinv);
  OSQPMatrix_rmult_diag(P, solver->work->scaling->Dinv);

  OSQPMatrix_lmult_diag(A, solver->work->scaling->Einv);
  OSQPMatrix_rmult_diag(A, solver->work->scaling->Dinv);

  OSQPVectorf_ew_prod(l, l, solver->work->scaling->Einv);
  OSQPVectorf_ew_prod(u, u, solver->work->scaling->Einv);

  return 0;
}

OSQPInt scale_dxdy(OSQPSolver*  solver,
                   OSQPVectorf* dx,
                   OSQPVectorf* dy_l,
                   OSQPVectorf* dy_u) {
    OSQPVectorf_ew_prod(dx, dx, solver->work->scaling->Dinv);
    OSQPVectorf_ew_prod(dy_l, dy_l, solver->work->scaling->Einv);
    OSQPVectorf_mult_scalar(dy_l, solver->work->scaling->c);
    OSQPVectorf_ew_prod(dy_u, dy_u, solver->work->scaling->Einv);
    OSQPVectorf_mult_scalar(dy_u, solver->work->scaling->c);

    return 0;
}

OSQPInt unscale_derivatives_PqAlu(OSQPSolver*    solver,
                                  OSQPCscMatrix* dP,
                                  OSQPVectorf*   dq,
                                  OSQPCscMatrix* dA,
                                  OSQPVectorf*   dl,
                                  OSQPVectorf*   du) {

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

OSQPInt adjoint_derivative(OSQPSolver*    solver,
                           OSQPFloat*     dx,
                           OSQPFloat*     dy_l,
                           OSQPFloat*     dy_u,
                           OSQPCscMatrix* dP,
                           OSQPFloat*     dq,
                           OSQPCscMatrix* dA,
                           OSQPFloat*     dl,
                           OSQPFloat*     du) {

    OSQPInt m = solver->work->data->m;
    OSQPInt n = solver->work->data->n;

    OSQPSettings*  settings  = solver->settings;

    OSQPMatrix*  P = OSQPMatrix_copy_new(solver->work->data->P);
    OSQPMatrix*  A = OSQPMatrix_copy_new(solver->work->data->A);
    OSQPVectorf* l = OSQPVectorf_copy_new(solver->work->data->l);
    OSQPVectorf* u = OSQPVectorf_copy_new(solver->work->data->u);
    OSQPVectorf* x = OSQPVectorf_new(solver->solution->x, n);  // Note: x/y are unscaled solutions
    OSQPVectorf* y = OSQPVectorf_new(solver->solution->y, m);

    // TODO: If we didn't have to unscale P/A/l/u we would not have to copy these
    if (solver->settings->scaling) unscale_PAlu(solver, P, A, l, u);

    OSQPFloat* l_data = OSQPVectorf_data(l);
    OSQPFloat* u_data = OSQPVectorf_data(u);
    OSQPFloat* y_data = OSQPVectorf_data(y);

    OSQPInt* A_ineq_l_vec = (OSQPInt *) c_malloc(m * sizeof(OSQPInt));
    OSQPInt* A_ineq_u_vec = (OSQPInt *) c_malloc(m * sizeof(OSQPInt));
    OSQPInt* A_eq_vec = (OSQPInt *) c_malloc(m * sizeof(OSQPInt));

    OSQPInt* eq_indices_vec = (OSQPInt *) c_malloc(m * sizeof(OSQPInt));
    OSQPInt* ineq_indices_vec = (OSQPInt *) c_malloc(m * sizeof(OSQPInt));
    OSQPInt* l_noninf_indices_vec = (OSQPInt *) c_malloc(m * sizeof(OSQPInt));
    OSQPInt* u_noninf_indices_vec = (OSQPInt *) c_malloc(m * sizeof(OSQPInt));
    OSQPInt* nu_indices_vec = (OSQPInt *) c_malloc(m * sizeof(OSQPInt));
    OSQPInt* nu_sign_vec = (OSQPInt *) c_malloc(m * sizeof(OSQPInt));

    OSQPVectorf* dxx = OSQPVectorf_new(dx, n);
    OSQPVectorf* dy_l_vec = OSQPVectorf_new(dy_l, m);
    OSQPVectorf* dy_u_vec = OSQPVectorf_new(dy_u, m);

    // TODO: Find out why scaling/unscaling is not working
    //if (solver->settings->scaling) scale_dxdy(solver, dxx, dy_l_vec, dy_u_vec);

    // TODO: We could use constr_type in OSQPWorkspace but it only tells us whether a constraint is 'loose'
    // not 'upper loose' or 'lower loose', which we seem to need here.
    OSQPFloat infval = OSQP_INFTY * OSQP_MIN_SCALING;

    OSQPInt n_ineq_l = 0;
    OSQPInt n_ineq_u = 0;
    OSQPInt n_ineq = 0;
    OSQPInt n_eq = 0;

    OSQPInt j;
    for (j = 0; j < m; j++) {
        OSQPFloat _l = l_data[j];
        OSQPFloat _u = u_data[j];
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

    OSQPVectori* A_ineq_l_i = OSQPVectori_new(A_ineq_l_vec, m);
    OSQPMatrix*  A_ineq_l = OSQPMatrix_submatrix_byrows(A, A_ineq_l_i);
    OSQPMatrix_mult_scalar(A_ineq_l, -1);

    OSQPVectori* A_ineq_u_i = OSQPVectori_new(A_ineq_u_vec, m);
    OSQPMatrix*  A_ineq_u = OSQPMatrix_submatrix_byrows(A, A_ineq_u_i);

    OSQPMatrix* G = OSQPMatrix_vstack(A_ineq_l, A_ineq_u);

    OSQPMatrix_free(A_ineq_l);
    OSQPMatrix_free(A_ineq_u);

    OSQPVectori* A_eq_i = OSQPVectori_new(A_eq_vec, m);
    OSQPMatrix*  A_eq = OSQPMatrix_submatrix_byrows(A, A_eq_i);

    // --------- lambda
    OSQPVectorf* m_zeros = OSQPVectorf_malloc(m);
    OSQPVectorf_set_scalar(m_zeros, 0);
    OSQPVectorf* y_u = OSQPVectorf_malloc(m);
    OSQPVectorf_ew_max_vec(y_u, y, m_zeros);
    OSQPVectorf* y_l = OSQPVectorf_malloc(m);
    OSQPVectorf_ew_min_vec(y_l, y, m_zeros);
    OSQPVectorf_mult_scalar(y_l, -1);
    OSQPVectorf_free(m_zeros);

    OSQPVectorf* y_l_ineq = OSQPVectorf_subvector_byrows(y_l, A_ineq_l_i);
    OSQPVectorf* y_u_ineq = OSQPVectorf_subvector_byrows(y_u, A_ineq_u_i);
    OSQPVectorf* lambda = OSQPVectorf_concat(y_l_ineq, y_u_ineq);

    OSQPVectorf_free(y_l_ineq);
    OSQPVectorf_free(y_u_ineq);
    // ---------- lambda

    // --------- slacks
    OSQPVectorf* l_ineq = OSQPVectorf_subvector_byrows(l, A_ineq_l_i);
    OSQPVectorf_mult_scalar(l_ineq, -1);
    OSQPVectorf* u_ineq = OSQPVectorf_subvector_byrows(u, A_ineq_u_i);

    OSQPVectorf* h = OSQPVectorf_concat(l_ineq, u_ineq);

    OSQPVectorf_free(l_ineq);
    OSQPVectorf_free(u_ineq);

    OSQPVectorf* slacks = OSQPVectorf_copy_new(h);
    OSQPMatrix_Axpy(G, x, slacks, 1, -1);
    OSQPVectorf_free(h);

    // ---------- GDiagLambda
    OSQPMatrix* GDiagLambda = OSQPMatrix_copy_new(G);
    OSQPMatrix_lmult_diag(GDiagLambda, lambda);

    // ---------- P_full
    OSQPMatrix* P_full = OSQPMatrix_triu_to_symm(P);

    // ---------- RHS
    OSQPVectorf* dy_l_ineq = OSQPVectorf_subvector_byrows(dy_l_vec, A_ineq_l_i);
    OSQPVectorf_free(dy_l_vec);
    OSQPVectorf* dy_u_ineq = OSQPVectorf_subvector_byrows(dy_u_vec, A_ineq_u_i);
    OSQPVectorf_free(dy_u_vec);
    OSQPVectorf* dlambd = OSQPVectorf_concat(dy_l_ineq, dy_u_ineq);
    OSQPVectorf_free(dy_l_ineq);
    OSQPVectorf_free(dy_u_ineq);

    OSQPFloat* d_nu_vec = (OSQPFloat *) c_malloc(n_eq * sizeof(OSQPFloat));
    for (j=0; j<n_eq; j++) {
        if (nu_sign_vec[j]==1) {
            d_nu_vec[j] = dy_u[nu_indices_vec[j]];
        } else if (nu_sign_vec[j]==-1) {
            d_nu_vec[j] = -dy_l[nu_indices_vec[j]];
        } else {}  // should never happen
    }
    OSQPVectorf* d_nu = OSQPVectorf_new(d_nu_vec, n_eq);
    c_free(d_nu_vec);

    OSQPVectorf* rhs_temp1 = OSQPVectorf_concat(dxx, dlambd);
    OSQPVectorf_free(dxx);
    OSQPVectorf_free(dlambd);
    OSQPVectorf* rhs_temp2 = OSQPVectorf_concat(rhs_temp1, d_nu);
    OSQPVectorf_free(rhs_temp1);
    OSQPVectorf_free(d_nu);
    OSQPVectorf_mult_scalar(rhs_temp2, -1);
    OSQPVectorf* zeros = OSQPVectorf_malloc(n + n_ineq_l + n_ineq_u + n_eq);
    OSQPVectorf_set_scalar(zeros, 0);
    OSQPVectorf* rhs = OSQPVectorf_concat(rhs_temp2, zeros);
    OSQPVectorf_free(rhs_temp2);
    OSQPVectorf_free(zeros);

    // ----------- Check
    adjoint_derivative_linsys_solver(solver, settings, P_full, G, A_eq, GDiagLambda, slacks, rhs);

    OSQPFloat* rhs_data = OSQPVectorf_data(rhs);

    OSQPFloat* r_yl = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));
    OSQPFloat* r_yu = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));
    // TODO: We shouldn't have to do this if we assemble r_yl/r_yu judiciously
    for (j=0; j<m; j++) r_yl[j] = 0;
    for (j=0; j<m; j++) r_yu[j] = 0;

    OSQPInt pos = n + n_ineq_l + n_ineq_u + n_eq;
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
    OSQPVectorf_ew_prod(ryl, ryl, y_l);
    OSQPVectorf_mult_scalar(ryl, -1);
    OSQPVectorf *ryu = OSQPVectorf_new(r_yu, m);
    c_free(r_yu);
    OSQPVectorf_ew_prod(ryu, ryu, y_u);

    // Assemble dP/dA
    // TODO: Check for incoming m/n/nzmax compatibility unless we're allocating
    OSQPFloat* rx_data  = OSQPVectorf_data(rx);
    OSQPFloat* x_data   = OSQPVectorf_data(x);
    OSQPFloat* y_u_data = OSQPVectorf_data(y_u);
    OSQPFloat* y_l_data = OSQPVectorf_data(y_l);
    OSQPFloat* ryu_data = OSQPVectorf_data(ryu);
    OSQPFloat* ryl_data = OSQPVectorf_data(ryl);

    OSQPInt col;
    for (col=0; col<n; col++) {
        OSQPInt p, i;
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

    OSQPMatrix_free(P);
    OSQPMatrix_free(A);
    OSQPVectorf_free(l);
    OSQPVectorf_free(u);
    OSQPVectorf_free(x);
    OSQPVectorf_free(y);

    return 0;
}
