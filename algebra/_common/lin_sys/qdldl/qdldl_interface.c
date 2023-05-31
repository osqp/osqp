#include "glob_opts.h"
#include "algebra_impl.h"
#include "printing.h"

#include "qdldl.h"
#include "qdldl_interface.h"

#ifndef OSQP_EMBEDDED_MODE
#include "amd.h"
#endif

#if OSQP_EMBEDDED_MODE != 1
#include "kkt.h"
#endif

#define STRINGIZE_(x) #x
#define STRINGIZE(x) STRINGIZE_(x)


void update_settings_linsys_solver_qdldl(qdldl_solver*       s,
                                         const OSQPSettings* settings) {
  return;
}

// Warm starting not used by direct solvers
void warm_start_linsys_solver_qdldl(qdldl_solver*      s,
                                    const OSQPVectorf* x) {
  return;
}

#ifndef OSQP_EMBEDDED_MODE

// Free LDL Factorization structure
void free_linsys_solver_qdldl(qdldl_solver* s) {
    if (s) {
        if (s->L) {
            if (s->L->p) c_free(s->L->p);
            if (s->L->i) c_free(s->L->i);
            if (s->L->x) c_free(s->L->x);
            c_free(s->L);
        }

        if (s->P)           c_free(s->P);
        if (s->Dinv)        c_free(s->Dinv);
        if (s->bp)          c_free(s->bp);
        if (s->sol)         c_free(s->sol);
        if (s->rho_inv_vec) c_free(s->rho_inv_vec);

        // These are required for matrix updates
        if (s->KKT)       csc_spfree(s->KKT);
        if (s->PtoKKT)    c_free(s->PtoKKT);
        if (s->AtoKKT)    c_free(s->AtoKKT);
        if (s->rhotoKKT)  c_free(s->rhotoKKT);

        if (s->adj)         c_free(s->adj);

        // QDLDL workspace
        if (s->D)         c_free(s->D);
        if (s->etree)     c_free(s->etree);
        if (s->Lnz)       c_free(s->Lnz);
        if (s->iwork)     c_free(s->iwork);
        if (s->bwork)     c_free(s->bwork);
        if (s->fwork)     c_free(s->fwork);
        c_free(s);

    }
}


/**
 * Compute LDL factorization of matrix A
 * @param  A    Matrix to be factorized
 * @param  p    Private workspace
 * @param  nvar Number of QP variables
 * @return      exitstatus (0 is good)
 */
static OSQPInt LDL_factor(OSQPCscMatrix* A,
                          qdldl_solver*  p,
                          OSQPInt        nvar) {

    OSQPInt sum_Lnz;
    OSQPInt factor_status;

    // Compute elimination tree
    sum_Lnz = QDLDL_etree(A->n, A->p, A->i, p->iwork, p->Lnz, p->etree);

    if (sum_Lnz < 0){
      // Error
      c_eprint("Error in KKT matrix LDL factorization when computing the elimination tree.");
      if(sum_Lnz == -1){
        c_eprint("Matrix is not perfectly upper triangular.");
      }
      else if(sum_Lnz == -2){
        c_eprint("Integer overflow in L nonzero count.");
      }
      return sum_Lnz;
    }

    // Allocate memory for Li and Lx
    p->L->i = (OSQPInt *)c_malloc(sizeof(OSQPInt)*sum_Lnz);
    p->L->x = (OSQPFloat *)c_malloc(sizeof(OSQPFloat)*sum_Lnz);
    p->L->nzmax = sum_Lnz;

    // Factor matrix
    factor_status = QDLDL_factor(A->n, A->p, A->i, A->x,
                                 p->L->p, p->L->i, p->L->x,
                                 p->D, p->Dinv, p->Lnz,
                                 p->etree, p->bwork, p->iwork, p->fwork);

    if (factor_status < 0){
      // Error
      c_eprint("Error in KKT matrix LDL factorization when computing the nonzero elements. There are zeros in the diagonal matrix");
      return factor_status;
    } else if (factor_status < nvar) {
      // Error: Number of positive elements of D should be equal to nvar
      c_eprint("Error in KKT matrix LDL factorization when computing the nonzero elements. The problem seems to be non-convex");
      return -2;
    }

    return 0;

}


static OSQPInt permute_KKT(OSQPCscMatrix** KKT,
                           qdldl_solver*   p,
                           OSQPInt         Pnz,
                           OSQPInt         Anz,
                           OSQPInt         m,
                           OSQPInt*        PtoKKT,
                           OSQPInt*        AtoKKT,
                           OSQPInt*        rhotoKKT) {
    OSQPFloat* info;
    OSQPInt    amd_status;
    OSQPInt*   Pinv;
    OSQPInt*   KtoPKPt;
    OSQPInt    i; // Indexing

    OSQPCscMatrix* KKT_temp;

    info = (OSQPFloat *)c_malloc(AMD_INFO * sizeof(OSQPFloat));

    // Compute permutation matrix P using AMD
#ifdef OSQP_USE_LONG
    amd_status = amd_l_order((*KKT)->n, (*KKT)->p, (*KKT)->i, p->P, (OSQPFloat *)OSQP_NULL, info);
#else
    amd_status = amd_order((*KKT)->n, (*KKT)->p, (*KKT)->i, p->P, (OSQPFloat *)OSQP_NULL, info);
#endif
    if (amd_status < 0) {
        // Free Amd info and return an error
        c_free(info);
        return amd_status;
    }


    // Inverse of the permutation vector
    Pinv = csc_pinv(p->P, (*KKT)->n);

    // Permute KKT matrix
    if (!PtoKKT && !AtoKKT && !rhotoKKT){  // No vectors to be stored
        // Assign values of mapping
        KKT_temp = csc_symperm((*KKT), Pinv, OSQP_NULL, 1);
    }
    else {
        // Allocate vector of mappings from unpermuted to permuted
        KtoPKPt = c_malloc((*KKT)->p[(*KKT)->n] * sizeof(OSQPInt));
        KKT_temp = csc_symperm((*KKT), Pinv, KtoPKPt, 1);

        // Update vectors PtoKKT, AtoKKT and rhotoKKT
        if (PtoKKT){
            for (i = 0; i < Pnz; i++){
                PtoKKT[i] = KtoPKPt[PtoKKT[i]];
            }
        }
        if (AtoKKT){
            for (i = 0; i < Anz; i++){
                AtoKKT[i] = KtoPKPt[AtoKKT[i]];
            }
        }
        if (rhotoKKT){
            for (i = 0; i < m; i++){
                rhotoKKT[i] = KtoPKPt[rhotoKKT[i]];
            }
        }

        // Cleanup vector of mapping
        c_free(KtoPKPt);
    }

    // Cleanup
    // Free previous KKT matrix and assign pointer to new one
    csc_spfree((*KKT));
    (*KKT) = KKT_temp;
    // Free Pinv
    c_free(Pinv);
    // Free Amd info
    c_free(info);

    return 0;
}


// Initialize LDL Factorization structure
OSQPInt init_linsys_solver_qdldl(qdldl_solver**      sp,
                                 const OSQPMatrix*   P,
                                 const OSQPMatrix*   A,
                                 const OSQPVectorf*  rho_vec,
                                 const OSQPSettings* settings,
                                 OSQPInt             polishing) {

    // Define Variables
    OSQPCscMatrix* KKT_temp; // Temporary KKT pointer
    OSQPInt    i;         // Loop counter
    OSQPInt    m, n;      // Dimensions of A
    OSQPInt    n_plus_m;  // Define n_plus_m dimension
    OSQPFloat* rhov;      // used for direct access to rho_vec data when polishing=false
    OSQPFloat  sigma = settings->sigma;

    // Allocate private structure to store KKT factorization
    qdldl_solver* s = c_calloc(1, sizeof(qdldl_solver));
    *sp = s;

    // Size of KKT
    n = P->csc->n;
    m = A->csc->m;
    s->n = n;
    s->m = m;
    n_plus_m = n + m;

    // Scalar parameters
    s->sigma = sigma;
    s->rho_inv = 1. / settings->rho;

    // Polishing flag
    s->polishing = polishing;

    // Link Functions
    s->name            = &name_qdldl;
    s->solve           = &solve_linsys_qdldl;
    s->update_settings = &update_settings_linsys_solver_qdldl;
    s->warm_start      = &warm_start_linsys_solver_qdldl;
    s->adjoint_derivative = &adjoint_derivative_qdldl;


#ifndef OSQP_EMBEDDED_MODE
    s->free = &free_linsys_solver_qdldl;
#endif

#if OSQP_EMBEDDED_MODE != 1
    s->update_matrices = &update_linsys_solver_matrices_qdldl;
    s->update_rho_vec  = &update_linsys_solver_rho_vec_qdldl;
#endif

    // Assign type
    s->type = OSQP_DIRECT_SOLVER;

    // Set number of threads to 1 (single threaded)
    s->nthreads = 1;

    // Sparse matrix L (lower triangular)
    // NB: We don not allocate L completely (CSC elements)
    //      L will be allocated during the factorization depending on the
    //      resulting number of elements.
    s->L = c_calloc(1, sizeof(OSQPCscMatrix));
    s->L->m  = n_plus_m;
    s->L->n  = n_plus_m;
    s->L->nz = -1;
    s->L->p  = (OSQPInt *)c_malloc((n_plus_m+1) * sizeof(QDLDL_int));

    // Diagonal matrix stored as a vector D
    s->Dinv = (QDLDL_float *)c_malloc(sizeof(QDLDL_float) * n_plus_m);
    s->D    = (QDLDL_float *)c_malloc(sizeof(QDLDL_float) * n_plus_m);

    // Permutation vector P
    s->P    = (QDLDL_int *)c_malloc(sizeof(QDLDL_int) * n_plus_m);

    // Working vector
    s->bp   = (QDLDL_float *)c_malloc(sizeof(QDLDL_float) * n_plus_m);

    // Solution vector
    s->sol  = (QDLDL_float *)c_malloc(sizeof(QDLDL_float) * n_plus_m);

    // Parameter vector
    if (rho_vec)
      s->rho_inv_vec = (OSQPFloat *)c_malloc(sizeof(OSQPFloat) * m);
    // else it is NULL

    // Elimination tree workspace
    s->etree = (QDLDL_int *)c_malloc(n_plus_m * sizeof(QDLDL_int));
    s->Lnz   = (QDLDL_int *)c_malloc(n_plus_m * sizeof(QDLDL_int));

    // Lx and Li are sparsity dependent, so set them to
    // null initially so we don't try to free them prematurely
    s->L->i = OSQP_NULL;
    s->L->x = OSQP_NULL;

    // Preallocate workspace
    s->iwork = (QDLDL_int *)c_malloc(sizeof(QDLDL_int)*(3*n_plus_m));
    s->bwork = (QDLDL_bool *)c_malloc(sizeof(QDLDL_bool)*n_plus_m);
    s->fwork = (QDLDL_float *)c_malloc(sizeof(QDLDL_float)*n_plus_m);

    // Form and permute KKT matrix
    if (polishing){ // Called from polish()

        KKT_temp = form_KKT(P->csc,A->csc,
                            0, //format = 0 means CSC
                            sigma, s->rho_inv_vec, sigma,
                            OSQP_NULL, OSQP_NULL, OSQP_NULL);

        // Permute matrix
        if (KKT_temp)
            permute_KKT(&KKT_temp, s, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL);
    }
    else { // Called from ADMM algorithm

        // Allocate vectors of indices
        s->PtoKKT = c_malloc(P->csc->p[n] * sizeof(OSQPInt));
        s->AtoKKT = c_malloc(A->csc->p[n] * sizeof(OSQPInt));
        s->rhotoKKT = c_malloc(m * sizeof(OSQPInt));

        // Use p->rho_inv_vec for storing param2 = rho_inv_vec
        if (rho_vec) {
          rhov = rho_vec->values;
          for (i = 0; i < m; i++){
              s->rho_inv_vec[i] = 1. / rhov[i];
          }
        }
        else {
          s->rho_inv = 1. / settings->rho;
        }

        KKT_temp = form_KKT(P->csc,A->csc,
                            0, //format = 0 means CSC format
                            sigma, s->rho_inv_vec, s->rho_inv,
                            s->PtoKKT, s->AtoKKT,s->rhotoKKT);

        // Permute matrix
        if (KKT_temp){
            permute_KKT(&KKT_temp, s, P->csc->p[n], A->csc->p[n], m, s->PtoKKT, s->AtoKKT, s->rhotoKKT);
        }
    }

    // Check if matrix has been created
    if (!KKT_temp){
        c_eprint("Error forming and permuting KKT matrix");
        free_linsys_solver_qdldl(s);
        *sp = OSQP_NULL;
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }

    // Factorize the KKT matrix
    if (LDL_factor(KKT_temp, s, n) < 0) {
        csc_spfree(KKT_temp);
        free_linsys_solver_qdldl(s);
        *sp = OSQP_NULL;
        return OSQP_NONCVX_ERROR;
    }

    if (polishing){ // If KKT passed, assign it to KKT_temp
        // Polish, no need for KKT_temp
        csc_spfree(KKT_temp);
    }
    else { // If not embedded option 1 copy pointer to KKT_temp. Do not free it.
        s->KKT = KKT_temp;
    }


    // No error
    return 0;
}

#endif  // OSQP_EMBEDDED_MODE

const char* name_qdldl(qdldl_solver* s) {
  return "QDLDL v" STRINGIZE(QDLDL_VERSION_MAJOR) "." STRINGIZE(QDLDL_VERSION_MINOR) "." STRINGIZE(QDLDL_VERSION_PATCH);
}


/* solve P'LDL'P x = b for x */
static void LDLSolve(OSQPFloat*           x,
                     const OSQPFloat*     b,
                     const OSQPCscMatrix* L,
                     const OSQPFloat*     Dinv,
                     const OSQPInt*       P,
                     OSQPFloat*           bp) {

  OSQPInt j;
  OSQPInt n = L->n;

  // permute_x(L->n, bp, b, P);
  for (j = 0 ; j < n ; j++) bp[j] = b[P[j]];

  QDLDL_solve(L->n, L->p, L->i, L->x, Dinv, bp);

  // permutet_x(L->n, x, bp, P);
  for (j = 0 ; j < n ; j++) x[P[j]] = bp[j];
}


OSQPInt solve_linsys_qdldl(qdldl_solver* s,
                           OSQPVectorf*  b,
                           OSQPInt       admm_iter) {

  OSQPInt    j;
  OSQPInt    n = s->n;
  OSQPInt    m = s->m;
  OSQPFloat* bv = b->values;

#ifndef OSQP_EMBEDDED_MODE
  if (s->polishing) {
    /* stores solution to the KKT system in b */
    LDLSolve(bv, bv, s->L, s->Dinv, s->P, s->bp);
  } else {
#endif
    /* stores solution to the KKT system in s->sol */
    LDLSolve(s->sol, bv, s->L, s->Dinv, s->P, s->bp);

    /* copy x_tilde from s->sol */
    for (j = 0 ; j < n ; j++) {
      bv[j] = s->sol[j];
    }

    /* compute z_tilde from b and s->sol */
    if (s->rho_inv_vec) {
      for (j = 0 ; j < m ; j++) {
        bv[j + n] += s->rho_inv_vec[j] * s->sol[j + n];
      }
    }
    else {
      for (j = 0 ; j < m ; j++) {
        bv[j + n] += s->rho_inv * s->sol[j + n];
      }
    }
#ifndef OSQP_EMBEDDED_MODE
  }
#endif
  return 0;
}


#if OSQP_EMBEDDED_MODE != 1

// Update private structure with new P and A
OSQPInt update_linsys_solver_matrices_qdldl(qdldl_solver*     s,
                                            const OSQPMatrix* P,
                                            const OSQPInt*    Px_new_idx,
                                            OSQPInt           P_new_n,
                                            const OSQPMatrix* A,
                                            const OSQPInt*    Ax_new_idx,
                                            OSQPInt           A_new_n) {

    OSQPInt pos_D_count;

    // Update KKT matrix with new P
    update_KKT_P(s->KKT, P->csc, Px_new_idx, P_new_n, s->PtoKKT, s->sigma, 0);

    // Update KKT matrix with new A
    update_KKT_A(s->KKT, A->csc, Ax_new_idx, A_new_n, s->AtoKKT);

    pos_D_count = QDLDL_factor(s->KKT->n, s->KKT->p, s->KKT->i, s->KKT->x,
        s->L->p, s->L->i, s->L->x, s->D, s->Dinv, s->Lnz,
        s->etree, s->bwork, s->iwork, s->fwork);

    //number of positive elements in D should match the
    //dimension of P if P + \sigma I is PD.   Error otherwise.
    return (pos_D_count == P->csc->n) ? 0 : 1;
}


OSQPInt update_linsys_solver_rho_vec_qdldl(qdldl_solver*      s,
                                           const OSQPVectorf* rho_vec,
                                           OSQPFloat          rho_sc) {

    OSQPInt i;
    OSQPInt m = s->m;
    OSQPFloat* rhov;

    // Update internal rho_inv_vec
    if (s->rho_inv_vec) {
      rhov = rho_vec->values;
      for (i = 0; i < m; i++){
          s->rho_inv_vec[i] = 1. / rhov[i];
      }
    }
    else {
      s->rho_inv = 1. / rho_sc;
    }

    // Update KKT matrix with new rho_vec
    update_KKT_param2(s->KKT, s->rho_inv_vec, s->rho_inv, s->rhotoKKT, s->m);

    return (QDLDL_factor(s->KKT->n, s->KKT->p, s->KKT->i, s->KKT->x,
        s->L->p, s->L->i, s->L->x, s->D, s->Dinv, s->Lnz,
        s->etree, s->bwork, s->iwork, s->fwork) < 0);
}

#endif

#ifndef OSQP_EMBEDDED_MODE

// --------- Derivative functions -------- //

//increment the D colptr by the number of nonzeros
//in a square diagonal matrix.
static void _colcount_diag(OSQPCscMatrix* D,
                           OSQPInt        initcol,
                           OSQPInt        blockcols) {

    OSQPInt j;
    for(j = initcol; j < (initcol + blockcols); j++){
        D->p[j]++;
    }
}

//increment D colptr by the number of nonzeros in M
static void _colcount_block(OSQPCscMatrix* D,
                            OSQPCscMatrix* M,
                            OSQPInt        initcol,
                            OSQPInt        istranspose) {

    OSQPInt nnzM, j;

    if(istranspose){
        nnzM = M->p[M->n];
        for (j = 0; j < nnzM; j++){
            D->p[M->i[j] + initcol]++;
        }
    }
    else {
        //just add the column count
        for (j = 0; j < M->n; j++){
            D->p[j + initcol] += M->p[j+1] - M->p[j];
        }
    }
}

static void _colcount_to_colptr(OSQPCscMatrix* D) {

    OSQPInt j, count;
    OSQPInt currentptr = 0;

    for(j = 0; j <= D->n; j++){
        count        = D->p[j];
        D->p[j]      = currentptr;
        currentptr  += count;
    }
}

//populate values from M using the K colptr as indicator of
//next fill location in each row
static void _fill_block(OSQPCscMatrix* K,
                        OSQPCscMatrix* M,
                        OSQPInt*       index_mapping,
                        OSQPInt        initrow,
                        OSQPInt        initcol,
                        OSQPInt        istranspose) {
    OSQPInt ii, jj, row, col, dest;

    for(ii=0; ii < M->n; ii++){
        for(jj = M->p[ii]; jj < M->p[ii+1]; jj++){
            if(istranspose){
                col = M->i[jj] + initcol;
                row = ii + initrow;
            }
            else {
                col = ii + initcol;
                row = M->i[jj] + initrow;
            }

            dest       = K->p[col]++;
            K->i[dest] = row;
            K->x[dest] = M->x[jj];
            if (index_mapping != OSQP_NULL) { index_mapping[jj] = dest; }
        }
    }
}

static void _fill_diag_values(OSQPCscMatrix* K,
                              OSQPInt*       index_mapping,
                              OSQPInt        initrow,
                              OSQPInt        initcol,
                              OSQPFloat*     values,
                              OSQPFloat      value_scalar,
                              OSQPInt        n) {

    OSQPInt j, dest, row, col;
    for (j = 0; j < n; j++) {
        row         = j + initrow;
        col         = j + initcol;
        dest        = K->p[col];
        K->i[dest]  = row;
        if (values != OSQP_NULL) {
            K->x[dest] = values[j];
        } else {
            K->x[dest] = value_scalar;
        }
        K->p[col]++;
        if (index_mapping != OSQP_NULL) { index_mapping[j] = dest; }
    }
}

static void _backshift_colptrs(OSQPCscMatrix* K) {

    int j;
    for(j = K->n; j > 0; j--){
        K->p[j] = K->p[j-1];
    }
    K->p[0] = 0;
}

static void _adj_perturb(OSQPCscMatrix* D,
                         OSQPFloat      eps) {
    OSQPInt j, dest;

    dest = 0;
    for (j = 0; j < D->m / 2; j++) {
        dest = D->p[j+1]-1;
        D->x[dest] += eps;
    }
    for (j = D->m / 2; j < D->m; j++) {
        dest = D->p[j+1]-1;
        D->x[dest] -= eps;
    }
}

static void _adj_assemble_csc(OSQPCscMatrix*     D,
                              const OSQPMatrix*  P_full,
                              const OSQPMatrix*  G,
                              const OSQPMatrix*  A_eq,
                              const OSQPMatrix*  GDiagLambda,
                              const OSQPVectorf* slacks) {

    OSQPInt n = OSQPMatrix_get_m(P_full);
    OSQPInt x = OSQPMatrix_get_m(G);        // No. of inequality constraints
    OSQPInt y = OSQPMatrix_get_m(A_eq);     // No. of equality constraints

    OSQPInt j;
    //use D.p to hold nnz entries in each column of the D matrix
    for (j=0; j <= 2*(n+x+y); j++){D->p[j] = 0;}

    _colcount_diag(D, 0, n+x+y);
    _colcount_block(D, P_full->csc, n+x+y, 0);
    _colcount_block(D, G->csc, n+x+y, 0);
    _colcount_block(D, A_eq->csc, n+x+y, 0);
    _colcount_block(D, GDiagLambda->csc, n+x+y+n, 1);
    _colcount_diag(D, n+x+y+n, x);
    _colcount_block(D, A_eq->csc, n+x+y+n+x, 1);
    _colcount_diag(D, n+x+y, n+x+y);

    //cumsum total entries to convert to D.p
    _colcount_to_colptr(D);

    _fill_diag_values(D, OSQP_NULL, 0, 0, OSQP_NULL, 1, n+x+y);
    _fill_block(D, P_full->csc, OSQP_NULL, 0, n+x+y, 0);
    _fill_block(D, G->csc, OSQP_NULL, n, n+x+y, 0);
    _fill_block(D, A_eq->csc, OSQP_NULL, n+x, n+x+y, 0);
    _fill_block(D, GDiagLambda->csc, OSQP_NULL, 0, n+x+y+n, 1);
    _fill_diag_values(D, OSQP_NULL, n, n+x+y+n, slacks->values, 0, x);
    _fill_block(D, A_eq->csc, OSQP_NULL, 0, n+x+y+n+x, 1);
    _fill_diag_values(D, OSQP_NULL, n+x+y, n+x+y, OSQP_NULL, 0, n+x+y);

    _backshift_colptrs(D);

}

OSQPInt adjoint_derivative_qdldl(qdldl_solver*      s,
                                 const OSQPMatrix*  P_full,
                                 const OSQPMatrix*  G,
                                 const OSQPMatrix*  A_eq,
                                 const OSQPMatrix*  GDiagLambda,
                                 const OSQPVectorf* slacks,
                                 const OSQPVectorf* rhs) {

    OSQPInt n = OSQPMatrix_get_m(P_full);
    OSQPInt n_ineq = OSQPMatrix_get_m(G);
    OSQPInt n_eq = OSQPMatrix_get_m(A_eq);

    // Get maximum number of nonzero elements (only upper triangular part)
    OSQPInt P_full_nnz = OSQPMatrix_get_nz(P_full);
    OSQPInt G_nnz = OSQPMatrix_get_nz(G);
    OSQPInt A_eq_nnz = OSQPMatrix_get_nz(A_eq);

    OSQPInt nnzKKT = n + n_ineq + n_eq +           // Number of diagonal elements in I (+eps)
                   P_full_nnz +                  // Number of elements in P_full
                   G_nnz +                       // Number of nonzeros in G
                   A_eq_nnz +                    // Number of nonzeros in A_eq
                   G_nnz +                       // Number of nonzeros in G'
                   n_ineq +                      // Number of diagonal elements in slacks
                   A_eq_nnz +                    // Number of nonzeros in A_eq'
                   n + n_ineq + n_eq;            // Number of -eps entries on diagonal

    OSQPInt dim = 2 * (n + n_ineq + n_eq);
    OSQPCscMatrix* adj = csc_spalloc(dim, dim, nnzKKT, 1, 0);
    if (!adj) return OSQP_NULL;
    _adj_assemble_csc(adj, P_full, G, A_eq, GDiagLambda, slacks);

    OSQPMatrix *adj_matrix = OSQPMatrix_new_from_csc(adj, 1);

    _adj_perturb(adj, 1e-6);

    // ----------------------------
    // QDLDL formulation + solve
    // ----------------------------
    const QDLDL_int   An   = dim;
    QDLDL_int i; // Counter

    //data for L and D factors
    QDLDL_int Ln = An;
    QDLDL_int *Lp;
    QDLDL_int *Li;
    QDLDL_float *Lx;
    QDLDL_float *D;
    QDLDL_float *Dinv;

    //permutation
    QDLDL_int   *P;
    QDLDL_int   *Pinv;

    //data for elim tree calculation
    QDLDL_int *etree;
    QDLDL_int *Lnz;
    QDLDL_int  sumLnz;

    //working data for factorisation
    QDLDL_int   *iwork;
    QDLDL_bool  *bwork;
    QDLDL_float *fwork;

    //Data for results of A\b
    QDLDL_float *x;
    QDLDL_float *x_work;

    etree = (QDLDL_int*)malloc(sizeof(QDLDL_int)*An);
    Lnz   = (QDLDL_int*)malloc(sizeof(QDLDL_int)*An);

    Lp    = (QDLDL_int*)malloc(sizeof(QDLDL_int)*(An+1));
    D     = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);
    Dinv  = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);

    iwork = (QDLDL_int*)malloc(sizeof(QDLDL_int)*(3*An));
    bwork = (QDLDL_bool*)malloc(sizeof(QDLDL_bool)*An);
    fwork = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);

    P = (QDLDL_int*)malloc(sizeof(QDLDL_int)*(An));
    Pinv = (QDLDL_int*)malloc(sizeof(QDLDL_int)*(An));

    OSQPInt amd_status;
#ifdef OSQP_USE_LONG
    amd_status = amd_l_order(An, adj->p, adj->i, P, (OSQPFloat *)OSQP_NULL, (OSQPFloat *)OSQP_NULL);
#else
    amd_status = amd_order(An, adj->p, adj->i, P, (OSQPFloat *)OSQP_NULL, (OSQPFloat *)OSQP_NULL);
#endif
    if (amd_status < 0) {
        return amd_status;
    }

    // Inverse of the permutation vector
    Pinv = csc_pinv(P, An);

    OSQPCscMatrix* adj_permuted;
    adj_permuted = csc_symperm(adj, Pinv, OSQP_NULL, 1);

    sumLnz = QDLDL_etree(An, adj_permuted->p, adj_permuted->i, iwork, Lnz, etree);

    Li    = (QDLDL_int*)malloc(sizeof(QDLDL_int)*sumLnz);
    Lx    = (QDLDL_float*)malloc(sizeof(QDLDL_float)*sumLnz);

    QDLDL_factor(An, adj_permuted->p, adj_permuted->i, adj_permuted->x, Lp, Li, Lx, D, Dinv, Lnz, etree, bwork, iwork, fwork);

    x = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);
    x_work = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);

    //when solving A\b, start with x = b
    for (i = 0 ; i < An ; i++) x_work[i] = rhs->values[P[i]];
    QDLDL_solve(Ln, Lp, Li, Lx, Dinv, x_work);
    for (i = 0 ; i < An ; i++) x[P[i]] = x_work[i];

    OSQPVectorf *sol = OSQPVectorf_new(x, An);
    OSQPVectorf *residual = OSQPVectorf_malloc(An);

    OSQPInt k;
    for (k=0; k<200; k++) {
        OSQPVectorf_copy(residual, rhs);
        OSQPMatrix_Axpy(adj_matrix, sol, residual, 1, -1);
        if (OSQPVectorf_norm_2(residual) < 1e-12) break;

        for (i = 0 ; i < An ; i++) x_work[i] = residual->values[P[i]];
        QDLDL_solve(Ln, Lp, Li, Lx, Dinv, x_work);
        for (i = 0 ; i < An ; i++) residual->values[P[i]] = x_work[i];

        OSQPVectorf_minus(sol, sol, residual);
    }

    OSQPVectorf_copy(rhs, sol);

    c_free(Lp);
    c_free(Li);
    c_free(Lx);
    c_free(D);
    c_free(Dinv);
    c_free(P);
    c_free(Pinv);
    c_free(etree);
    c_free(Lnz);
    c_free(iwork);
    c_free(bwork);
    c_free(fwork);
    c_free(x);
    c_free(x_work);

    csc_spfree(adj_permuted);
    OSQPMatrix_free(adj_matrix);
    csc_spfree(adj);

    OSQPVectorf_free(sol);
    OSQPVectorf_free(residual);

    return 0;
}

#endif
