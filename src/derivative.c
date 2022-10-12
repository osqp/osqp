#include "derivative.h"
#include "lin_alg.h"
#include "util.h"
#include "auxil.h"
#include "lin_alg.h"
#include "proj.h"
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

OSQPInt adjoint_derivative_get_mat(OSQPSolver *solver,
                                        OSQPCscMatrix* dP,
                                        OSQPCscMatrix* dA) {

    OSQPInt n = solver->work->data->n;
    OSQPDerivativeData *derivative_data = solver->work->derivative_data;
    OSQPVectorf* x = OSQPVectorf_new(solver->solution->x, n);  // unscaled solution
    OSQPFloat* x_data = OSQPVectorf_data(x);

    OSQPFloat* y_u_data = OSQPVectorf_data(derivative_data->y_u);
    OSQPFloat* y_l_data = OSQPVectorf_data(derivative_data->y_l);
    OSQPFloat* ryu_data = OSQPVectorf_data(derivative_data->ryu);
    OSQPFloat* ryl_data = OSQPVectorf_data(derivative_data->ryl);

    OSQPInt pos = n + derivative_data->n_ineq_l + derivative_data->n_ineq_u + derivative_data->n_eq;
    OSQPVectorf* rx = OSQPVectorf_view(derivative_data->rhs, pos, n);
    OSQPFloat* rx_data  = OSQPVectorf_data(rx);

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

    OSQPVectorf_view_free(rx);
    OSQPVectorf_free(x);

    return 0;
}

OSQPInt adjoint_derivative_get_vec(OSQPSolver *solver,
                                        OSQPFloat*     dq,
                                        OSQPFloat*     dl,
                                        OSQPFloat*     du) {

    OSQPInt n = solver->work->data->n;
    OSQPDerivativeData *derivative_data = solver->work->derivative_data;

    OSQPInt pos = n + derivative_data->n_ineq_l + derivative_data->n_ineq_u + derivative_data->n_eq;
    OSQPVectorf* rx = OSQPVectorf_view(derivative_data->rhs, pos, n);

    // Assign vector derivatives to function arguments
    OSQPVectorf_to_raw(dq, rx);
    OSQPVectorf_to_raw(dl, derivative_data->ryl);
    OSQPVectorf_to_raw(du, derivative_data->ryu);
    for (OSQPInt i=0; i<OSQPVectorf_length(derivative_data->ryu); i++) {
        du[i] = -du[i];
    }

    OSQPVectorf_view_free(rx);
    return 0;
}

OSQPInt adjoint_derivative_compute(OSQPSolver *solver,
                                        const OSQPMatrix*   P,
                                        const OSQPMatrix*   G,
                                        const OSQPMatrix*   A_eq,
                                        OSQPMatrix*         GDiagLambda,
                                        OSQPVectorf*        slacks) {

    OSQPInt m = solver->work->data->m;
    OSQPInt n = solver->work->data->n;
    OSQPDerivativeData *derivative_data = solver->work->derivative_data;

    OSQPVectorf* y = OSQPVectorf_new(solver->solution->y, m);
    OSQPFloat* y_data = OSQPVectorf_data(y);

    adjoint_derivative_linsys_solver(solver, solver->settings, P, G, A_eq, GDiagLambda, slacks, derivative_data->rhs);

    OSQPFloat* rhs_data = OSQPVectorf_data(derivative_data->rhs);

    OSQPFloat* r_yl = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));
    OSQPFloat* r_yu = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));
    // TODO: We shouldn't have to do this if we assemble r_yl/r_yu judiciously
    OSQPInt j;
    for (j=0; j<m; j++) r_yl[j] = 0;
    for (j=0; j<m; j++) r_yu[j] = 0;

    OSQPInt pos = n + derivative_data->n_ineq_l + derivative_data->n_ineq_u + derivative_data->n_eq;

    pos += n;
    for (j=0; j<derivative_data->n_ineq_l; j++) {
        r_yl[OSQPVectori_data(derivative_data->l_noninf_indices_vec)[j]] = -rhs_data[pos+j];
    }
    pos += derivative_data->n_ineq_l;
    for (j=0; j<derivative_data->n_ineq_u; j++) {
        r_yu[OSQPVectori_data(derivative_data->u_noninf_indices_vec)[j]] = rhs_data[pos+j];
    }
    pos += derivative_data->n_ineq_u;
    for (j=0; j<derivative_data->n_eq; j++) {
        if (OSQPVectori_data(derivative_data->nu_sign_vec)[j]==1) {
            r_yl[OSQPVectori_data(derivative_data->eq_indices_vec)[j]] = 0;
            r_yu[OSQPVectori_data(derivative_data->eq_indices_vec)[j]] = rhs_data[pos+j] / y_data[OSQPVectori_data(derivative_data->eq_indices_vec)[j]];
        } else {
            r_yl[OSQPVectori_data(derivative_data->eq_indices_vec)[j]] = -rhs_data[pos+j] / y_data[OSQPVectori_data(derivative_data->eq_indices_vec)[j]];
            r_yu[OSQPVectori_data(derivative_data->eq_indices_vec)[j]] = 0;
        }
    }

    derivative_data->ryl = OSQPVectorf_new(r_yl, m);
    c_free(r_yl);
    OSQPVectorf_ew_prod(derivative_data->ryl, derivative_data->ryl, derivative_data->y_l);
    OSQPVectorf_mult_scalar(derivative_data->ryl, -1);
    derivative_data->ryu = OSQPVectorf_new(r_yu, m);
    c_free(r_yu);
    OSQPVectorf_ew_prod(derivative_data->ryu, derivative_data->ryu, derivative_data->y_u);
}

OSQPInt adjoint_derivative(OSQPSolver*    solver,
                           OSQPFloat*     dx,
                           OSQPFloat*     dy_l,
                           OSQPFloat*     dy_u) {

    OSQPSettings*  settings  = solver->settings;
    OSQPWorkspace* work = solver->work;
    OSQPDerivativeData* derivative_data = work->derivative_data;

    OSQPInt m = work->data->m;
    OSQPInt n = work->data->n;

    OSQPMatrix*  P = OSQPMatrix_copy_new(work->data->P);
    OSQPMatrix*  A = OSQPMatrix_copy_new(work->data->A);
    OSQPVectorf* l = OSQPVectorf_copy_new(work->data->l);
    OSQPVectorf* u = OSQPVectorf_copy_new(work->data->u);
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
    OSQPInt* nu_sign_vec = (OSQPInt *) c_malloc(m * sizeof(OSQPInt));

    OSQPVectorf* dxx = OSQPVectorf_new(dx, n);
    OSQPVectorf* dy_l_vec = OSQPVectorf_new(dy_l, m);
    OSQPVectorf* dy_u_vec = OSQPVectorf_new(dy_u, m);

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

    derivative_data->n_ineq_l = n_ineq_l;
    derivative_data->n_ineq_u = n_ineq_u;
    derivative_data->n_eq = n_eq;

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
    derivative_data->y_u = OSQPVectorf_malloc(m);
    OSQPVectorf_ew_max_vec(derivative_data->y_u, y, m_zeros);
    derivative_data->y_l = OSQPVectorf_malloc(m);
    OSQPVectorf_ew_min_vec(derivative_data->y_l, y, m_zeros);
    OSQPVectorf_mult_scalar(derivative_data->y_l, -1);
    OSQPVectorf_free(m_zeros);

    OSQPVectorf* y_l_ineq = OSQPVectorf_subvector_byrows(derivative_data->y_l, A_ineq_l_i);
    OSQPVectorf* y_u_ineq = OSQPVectorf_subvector_byrows(derivative_data->y_u, A_ineq_u_i);
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
            d_nu_vec[j] = dy_u[eq_indices_vec[j]];
        } else if (nu_sign_vec[j]==-1) {
            d_nu_vec[j] = -dy_l[eq_indices_vec[j]];
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

    // --------- stuff things into derivative_data
    // All this should go inside _compute
    derivative_data->rhs = OSQPVectorf_copy_new(rhs);
    OSQPVectorf_free(rhs);
    derivative_data->l_noninf_indices_vec = OSQPVectori_new(l_noninf_indices_vec, m);
    c_free(l_noninf_indices_vec);
    derivative_data->u_noninf_indices_vec = OSQPVectori_new(u_noninf_indices_vec, m);
    c_free(u_noninf_indices_vec);
    derivative_data->nu_sign_vec = OSQPVectori_new(nu_sign_vec, m);
    c_free(nu_sign_vec);
    derivative_data->eq_indices_vec = OSQPVectori_new(eq_indices_vec, m);
    c_free(eq_indices_vec);

    adjoint_derivative_compute(solver, P_full, G, A_eq, GDiagLambda, slacks);

    // --------------- ASSEMBLE ----------------- //

    // Free up remaining stuff
    c_free(ineq_indices_vec);

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

    OSQPMatrix_free(P);
    OSQPMatrix_free(A);
    OSQPVectorf_free(l);
    OSQPVectorf_free(u);
    OSQPVectorf_free(x);
    OSQPVectorf_free(y);

    return 0;
}