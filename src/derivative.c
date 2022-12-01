#include "derivative.h"
#include "lin_alg.h"
#include "error.h"
#include "csc_utils.h"
#include "csc_math.h"

OSQPInt unscale_PA(OSQPSolver*  solver,
                     OSQPMatrix*  P,
                     OSQPMatrix*  A) {

  OSQPMatrix_mult_scalar(P, solver->work->scaling->cinv);
  OSQPMatrix_lmult_diag(P, solver->work->scaling->Dinv);
  OSQPMatrix_rmult_diag(P, solver->work->scaling->Dinv);

  OSQPMatrix_lmult_diag(A, solver->work->scaling->Einv);
  OSQPMatrix_rmult_diag(A, solver->work->scaling->Dinv);

  return 0;
}

OSQPInt unscale_lu(OSQPSolver*  solver,
                     OSQPVectorf* l,
                     OSQPVectorf* u) {

    OSQPVectorf_ew_prod(l, l, solver->work->scaling->Einv);
    OSQPVectorf_ew_prod(u, u, solver->work->scaling->Einv);

    return 0;
}

OSQPInt adjoint_derivative_get_mat(OSQPSolver *solver,
                                        OSQPCscMatrix* dP,
                                        OSQPCscMatrix* dA) {

    // Check if solver has been initialized
    if (!solver || !solver->work || !solver->work->derivative_data)
      return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

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

    // Check if solver has been initialized
    if (!solver || !solver->work || !solver->work->derivative_data)
      return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

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
                                   OSQPFloat*     dx,
                                   OSQPFloat*     dy_l,
                                   OSQPFloat*     dy_u) {

    // Check if solver has been initialized
    if (!solver || !solver->work || !solver->work->derivative_data)
      return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

    OSQPInt m = solver->work->data->m;
    OSQPInt n = solver->work->data->n;
    OSQPDerivativeData *derivative_data = solver->work->derivative_data;

    OSQPMatrix*  P = OSQPMatrix_copy_new(solver->work->data->P);
    OSQPMatrix*  A = OSQPMatrix_copy_new(solver->work->data->A);
    OSQPVectorf* l = OSQPVectorf_copy_new(solver->work->data->l);
    OSQPVectorf* u = OSQPVectorf_copy_new(solver->work->data->u);
    OSQPVectorf* x = OSQPVectorf_new(solver->solution->x, n);  // Note: x/y are unscaled solutions
    OSQPVectorf* y = OSQPVectorf_new(solver->solution->y, m);

    // TODO: If we didn't have to unscale P/A/l/u we would not have to copy these
    if (solver->settings->scaling) unscale_PA(solver, P, A);
    if (solver->settings->scaling) unscale_lu(solver, l, u);

    OSQPFloat* l_data = OSQPVectorf_data(l);
    OSQPFloat* u_data = OSQPVectorf_data(u);
    OSQPFloat* y_data = OSQPVectorf_data(y);

    OSQPInt* A_ineq_l_vec = (OSQPInt *) c_malloc(m * sizeof(OSQPInt));
    OSQPInt* A_ineq_u_vec = (OSQPInt *) c_malloc(m * sizeof(OSQPInt));
    OSQPInt* A_eq_vec = (OSQPInt *) c_malloc(m * sizeof(OSQPInt));

    OSQPInt* eq_indices_vec = (OSQPInt *) c_malloc(m * sizeof(OSQPInt));
    OSQPInt* l_noninf_indices_vec = (OSQPInt *) c_malloc(m * sizeof(OSQPInt));
    OSQPInt* u_noninf_indices_vec = (OSQPInt *) c_malloc(m * sizeof(OSQPInt));
    OSQPInt* nu_sign_vec = (OSQPInt *) c_malloc(m * sizeof(OSQPInt));

    OSQPVectorf* dy_l_vec = OSQPVectorf_new(dy_l, m);
    OSQPVectorf* dy_u_vec = OSQPVectorf_new(dy_u, m);

    // TODO: We could use constr_type in OSQPWorkspace but it only tells us whether a constraint is 'loose'
    // not 'upper loose' or 'lower loose', which we seem to need here.
    OSQPFloat infval = OSQP_INFTY * OSQP_MIN_SCALING;

    OSQPInt n_ineq_l = 0;
    OSQPInt n_ineq_u = 0;
    OSQPInt n_eq = 0;

    OSQPInt j;
    for (j = 0; j < m; j++) {
        OSQPFloat _l = l_data[j];
        OSQPFloat _u = u_data[j];
        if (_l < _u) {
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
    c_free(A_ineq_l_vec);
    OSQPMatrix*  A_ineq_l = OSQPMatrix_submatrix_byrows(A, A_ineq_l_i);
    OSQPMatrix_mult_scalar(A_ineq_l, -1);

    OSQPVectori* A_ineq_u_i = OSQPVectori_new(A_ineq_u_vec, m);
    c_free(A_ineq_u_vec);
    OSQPMatrix*  A_ineq_u = OSQPMatrix_submatrix_byrows(A, A_ineq_u_i);

    OSQPMatrix* G = OSQPMatrix_vstack(A_ineq_l, A_ineq_u);

    OSQPMatrix_free(A_ineq_l);
    OSQPMatrix_free(A_ineq_u);

    OSQPVectori* A_eq_i = OSQPVectori_new(A_eq_vec, m);
    c_free(A_eq_vec);
    OSQPMatrix*  A_eq = OSQPMatrix_submatrix_byrows(A, A_eq_i);
    OSQPVectori_free(A_eq_i);
    OSQPMatrix_free(A);

    // --------- lambda
    OSQPVectorf* m_zeros = OSQPVectorf_malloc(m);
    OSQPVectorf_set_scalar(m_zeros, 0);
    OSQPVectorf_ew_max_vec(derivative_data->y_u, y, m_zeros);
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
    OSQPVectorf_free(l);
    OSQPVectorf_mult_scalar(l_ineq, -1);
    OSQPVectorf* u_ineq = OSQPVectorf_subvector_byrows(u, A_ineq_u_i);
    OSQPVectorf_free(u);
    OSQPVectorf* h = OSQPVectorf_concat(l_ineq, u_ineq);

    OSQPVectorf_free(l_ineq);
    OSQPVectorf_free(u_ineq);

    OSQPVectorf* slacks = OSQPVectorf_copy_new(h);
    OSQPVectorf_free(h);
    OSQPMatrix_Axpy(G, x, slacks, 1, -1);
    OSQPVectorf_free(x);

    // ---------- GDiagLambda
    OSQPMatrix* GDiagLambda = OSQPMatrix_copy_new(G);
    OSQPMatrix_lmult_diag(GDiagLambda, lambda);
    OSQPVectorf_free(lambda);

    // ---------- Assemble RHS of the linear system
    OSQPVectorf* dy_l_ineq = OSQPVectorf_subvector_byrows(dy_l_vec, A_ineq_l_i);
    OSQPVectorf_free(dy_l_vec);
    OSQPVectori_free(A_ineq_l_i);
    OSQPVectorf* dy_u_ineq = OSQPVectorf_subvector_byrows(dy_u_vec, A_ineq_u_i);
    OSQPVectorf_free(dy_u_vec);
    OSQPVectori_free(A_ineq_u_i);

    OSQPVectorf *rhs = derivative_data->rhs;
    OSQPInt pos = 0;
    OSQPVectorf_subvector_assign(rhs, dx, pos, n, -1);
    pos += n;
    OSQPVectorf_subvector_assign(rhs, OSQPVectorf_data(dy_l_ineq), pos, n_ineq_l, -1);
    OSQPVectorf_free(dy_l_ineq);
    pos += n_ineq_l;
    OSQPVectorf_subvector_assign(rhs, OSQPVectorf_data(dy_u_ineq), pos, n_ineq_u, -1);
    OSQPVectorf_free(dy_u_ineq);
    pos += n_ineq_u;

    OSQPFloat* d_nu_vec = (OSQPFloat *) c_malloc(n_eq * sizeof(OSQPFloat));
    for (j=0; j<n_eq; j++) {
      if (nu_sign_vec[j]==1) {
        d_nu_vec[j] = dy_u[eq_indices_vec[j]];
      } else if (nu_sign_vec[j]==-1) {
        d_nu_vec[j] = -dy_l[eq_indices_vec[j]];
      } else {}  // should never happen
    }
    OSQPVectorf_subvector_assign(rhs, d_nu_vec, pos, n_eq, -1);
    c_free(d_nu_vec);
    pos += n_eq;

    OSQPVectorf_subvector_assign_scalar(rhs, 0, pos, n + n_ineq_l + n_ineq_u + n_eq);
    // ---------- Assemble RHS of the linear system

    OSQPMatrix* P_full = OSQPMatrix_triu_to_symm(P);
    OSQPMatrix_free(P);
    adjoint_derivative_linsys_solver(solver, solver->settings, P_full, G, A_eq, GDiagLambda, slacks, rhs);
    OSQPMatrix_free(P_full);
    OSQPMatrix_free(G);
    OSQPMatrix_free(A_eq);
    OSQPMatrix_free(GDiagLambda);
    OSQPVectorf_free(slacks);

    OSQPFloat* rhs_data = OSQPVectorf_data(rhs);

    OSQPFloat* r_yl = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));
    OSQPFloat* r_yu = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));
    // TODO: We shouldn't have to do this if we assemble r_yl/r_yu judiciously
    for (j=0; j<m; j++) r_yl[j] = 0;
    for (j=0; j<m; j++) r_yu[j] = 0;

    pos += n;
    for (j=0; j<derivative_data->n_ineq_l; j++) {
        r_yl[l_noninf_indices_vec[j]] = -rhs_data[pos+j];
    }
    c_free(l_noninf_indices_vec);
    pos += derivative_data->n_ineq_l;
    for (j=0; j<derivative_data->n_ineq_u; j++) {
        r_yu[u_noninf_indices_vec[j]] = rhs_data[pos+j];
    }
    c_free(u_noninf_indices_vec);
    pos += derivative_data->n_ineq_u;
    for (j=0; j<derivative_data->n_eq; j++) {
        if (nu_sign_vec[j]==1) {
            r_yl[eq_indices_vec[j]] = 0;
            r_yu[eq_indices_vec[j]] = rhs_data[pos+j] / y_data[eq_indices_vec[j]];
        } else {
            r_yl[eq_indices_vec[j]] = -rhs_data[pos+j] / y_data[eq_indices_vec[j]];
            r_yu[eq_indices_vec[j]] = 0;
        }
    }
    c_free(nu_sign_vec);
    c_free(eq_indices_vec);

    OSQPVectorf_from_raw(derivative_data->ryl, r_yl);
    c_free(r_yl);
    OSQPVectorf_ew_prod(derivative_data->ryl, derivative_data->ryl, derivative_data->y_l);
    OSQPVectorf_mult_scalar(derivative_data->ryl, -1);
    OSQPVectorf_from_raw(derivative_data->ryu, r_yu);
    c_free(r_yu);
    OSQPVectorf_ew_prod(derivative_data->ryu, derivative_data->ryu, derivative_data->y_u);

    //cleanup
    OSQPVectorf_free(y);

    return 0;
}
