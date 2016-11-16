#include "private.h"
#include "util.h"



// Free LDL Factorization structure
void free_priv(Priv *p) {
    if (p) {
        if (p->L)
            csc_spfree(p->L);
        if (p->P)
            c_free(p->P);
        if (p->Dinv)
            c_free(p->Dinv);
        if (p->bp)
            c_free(p->bp);
        c_free(p);
    }
}


/* Form square symmetric KKT matrix of the form
   [P + rho I,         A';
    A           -1/rhoI]

Arguments
---------
P : cost matrix (already just upper triangular part)
A: linear constraint matrix
rho: ADMM step
rho_inv:
N.B. Only the upper triangular part is stuffed!
*/
csc * form_KKT(const csc * P, const  csc * A, c_float rho, c_int rho_inv){
    c_int nKKT, nnzKKTmax; // Size, number of nonzeros and max number of nonzeros in KKT matrix
    csc *KKT_trip, *KKT;           // KKT matrix in triplet format and CSC format
    c_int ptr, i, j; // Counters for elements (i,j) and index pointer
    c_int z_P=0, z_KKT=0;   // Counter for total number of elements in P and in KKT

    // Get matrix dimensions
    nKKT = P->m + A->m;

    // Get maximum number of nonzero elements (only upper triangular part)
    nnzKKTmax = P->nzmax +           // Number of elements in P
                P->m +               // Number of elements in rhoI
                A->nzmax +           // Number of nonzeros in A
                A->m;                // Number of elements in -1/rho I

    // Preallocate KKT matrix in triplet format
    KKT_trip = csc_spalloc(nKKT, nKKT, nnzKKTmax, 1, 1);

    // #if PRINTLEVEL > 2
    //     c_print("Forming KKT matrix\n");
    // #endif

    if (!KKT_trip) return OSQP_NULL;  // Failed to preallocate matrix

    // Allocate Triplet matrices
    // P + rho I
    for (j = 0; j < P->n; j++){ // cycle over columns
        // No elements in column j => add diagonal element rho
        if (P->p[j] == P->p[j+1]){
            KKT_trip->i[z_KKT] = j;
            KKT_trip->p[z_KKT] = j;
            KKT_trip->x[z_KKT] = rho;
            z_KKT++;
        }
        for (ptr = P->p[j]; ptr < P->p[j + 1]; ptr++) { // cycle over rows
            // Get current row
            i = P->i[ptr];

            //DEBUG
            // c_print("\n\nP(%i, %i) = %.4f\n", i, j, P->x[z_P]);
            // c_print("P->p[j] = %i\n", P->p[j]);
            // c_print("P->p[j+1] = %i\n", P->p[j+1]);


            // Add element of P
            KKT_trip->i[z_KKT] = i;
            KKT_trip->p[z_KKT] = j;
            KKT_trip->x[z_KKT] = P->x[z_P];
            if (i == j){ // P has a diagonal element, add rho
                KKT_trip->x[z_KKT] += rho;
            }
            z_P++;
            z_KKT++;

            // Add diagonal rho in case
            if ((i < j) && // Diagonal element not reached
                (ptr + 1 == P->p[j+1])){ // last element of column j

                // Add diagonal element rho
                KKT_trip->i[z_KKT] = j;
                KKT_trip->p[z_KKT] = j;
                KKT_trip->x[z_KKT] = rho;
                z_KKT++;
            }
        }
    }


    // A' at top right
    for (j = 0; j < A->n; j++) {  // Cycle over columns of A
        for (ptr = A->p[j]; ptr < A->p[j + 1]; ptr++) {
            // DEBUG
            // c_print("A(%i, %i) = %.4f\n", A->i[ptr], j, A->x[ptr]);

            KKT_trip->p[z_KKT] = P->m + A->i[ptr];  // Assign column index from row index of A
            KKT_trip->i[z_KKT] = j; // Assign row index from column index of A
            KKT_trip->x[z_KKT] = A->x[ptr];  // Assign A value element
            z_KKT++;
        }
    }

    // rho_inv = 1:  -1/rho*I at bottom right
    // rho_inv = 0:     rho*I at bottom right
    if (rho_inv > 0) {
        for (j = 0; j < A->m; j++) {
            KKT_trip->i[z_KKT] = j + P->n;
            KKT_trip->p[z_KKT] = j + P->n;
            KKT_trip->x[z_KKT] = -1./rho;
            z_KKT++;
        }
    }
    else {
        for (j = 0; j < A->m; j++) {
            KKT_trip->i[z_KKT] = j + P->n;
            KKT_trip->p[z_KKT] = j + P->n;
            KKT_trip->x[z_KKT] = -rho;
            z_KKT++;
        }
    }


    // Allocate number of nonzeros
    KKT_trip->nz = z_KKT;

    // DEBUG: Print matrix
    // print_trip_matrix(KKT_trip, "KKT_trip");

    // Convert triplet matrix to csc format
    KKT = triplet_to_csc(KKT_trip);

    // print_csc_matrix(KKT, "KKT");

    // DEBUG
    // c_print("nKKT = %i\n", nKKT);
    // c_print("KKT->nnz = %i\n", KKT->nzmax);
    // c_print("KKT_trip->nz = %i\n", KKT_trip->nz);
    // c_print("KKT_trip->nzmax = %i\n", KKT_trip->nzmax);

    // c_float * KKTdns =  csc_to_dns(KKT);
    // print_dns_matrix(KKTdns, P->n + A->n, P->n + A->n, "KKTdns");

    // Clean matrix in triplet format and return result
    csc_spfree(KKT_trip);
    return KKT;

}


/**
 * Compute LDL factorization of matrix P A P'. If P = Pinv = OSQP_NULL,
 * then factorize matrix A.
 * @param  A    Matrix to be factorized
 * @param  P    Permutation matrix (stored as a vector)
 * @param  Pinv Inverse of the permutation matrix (stored as a vector)
 * @param  L    <out> Lower triangular matrix
 *              NB: Diagonal of L is assumed to be unit, and is not stored
 * @param  D    Diagonal matrix (stored as a vector)
 * @return      Status of the routine
 */
c_int LDLFactor(csc *A, c_int P[], c_int Pinv[], csc **L, c_float **D) {
    c_int kk, n = A->n;
    c_int *Parent = c_malloc(n * sizeof(c_int));
    c_int *Lnz = c_malloc(n * sizeof(c_int));
    c_int *Flag = c_malloc(n * sizeof(c_int));
    c_int *Pattern = c_malloc(n * sizeof(c_int));
    c_float *Y = c_malloc(n * sizeof(c_float));
    (*L)->p = (c_int *)c_malloc((1 + n) * sizeof(c_int));

    // Symbolic factorization
    LDL_symbolic(n, A->p, A->i, (*L)->p, Parent, Lnz, Flag, P, Pinv);

    (*L)->nzmax = *((*L)->p + n);
    (*L)->x = (c_float *)c_malloc((*L)->nzmax * sizeof(c_float));
    (*L)->i = (c_int *)c_malloc((*L)->nzmax * sizeof(c_int));
    // *D = (c_float *)c_malloc(n * sizeof(c_float));

    if (!(*D) || !(*L)->i || !(*L)->x || !Y || !Pattern || !Flag || !Lnz ||
        !Parent)
        return -1;

    // Numeric factorization
    kk = LDL_numeric(n, A->p, A->i, A->x, (*L)->p, Parent, Lnz, (*L)->i,
                     (*L)->x, *D, Y, Pattern, Flag, P, Pinv);

    // Memory clean-up
    c_free(Parent);
    c_free(Lnz);
    c_free(Flag);
    c_free(Pattern);
    c_free(Y);
    return (kk - n);
}




/**
 *  Factorize matrix A using sparse LDL factorization with pivoting as:
 *      P A P' = L D L'
 *  The result is stored in the LDL Factorization structure Priv.
 */
c_int factorize(csc *A, Priv *p) {
    c_float *info;
    c_int *Pinv, amd_status, ldl_status;
    csc *C;
    info = (c_float *)c_malloc(AMD_INFO * sizeof(c_float));

    // Compute permutation metrix P using SuiteSparse/AMD
    amd_status = amd_order(A->n, A->p, A->i, p->P, (c_float *)OSQP_NULL, info);
    if (amd_status < 0)
        return (amd_status);

    // Compute inverse of permutation matrix P
    Pinv = csc_pinv(p->P, A->n);
    // Symmetric permutation of A:  permA = P A P'
    C = csc_symperm(A, Pinv, 1);

    // Compute LDL factorization of  C = P A P'
    // NB: D matrix is stored in Dinv.
    ldl_status = LDLFactor(C, OSQP_NULL, OSQP_NULL, &p->L, &p->Dinv);

    // Invert elements of D that are stored in p->Dinv
    vec_ew_recipr(p->Dinv, p->Dinv, A->n);

    // Memory clean-up
    csc_spfree(C);
    c_free(Pinv);
    c_free(info);
    return (ldl_status);
}


// Initialize LDL Factorization structure
Priv *init_priv(const csc * P, const csc * A, const Settings *settings){
    // Define Variables
    csc * KKT;  // KKT Matrix
    Priv * p;   // KKT factorization structure

    // Allocate private structure to store KKT factorization
    // Allocate pointers
    p = c_calloc(1, sizeof(Priv));

    // Size of KKT
    c_int n_plus_m = P->m + A->m;

    // Sparse matrix L (lower triangular)
    // N.B. Do not allocate L completely (CSC elements)
    //      L will be allocated during the factorization depending on the
    //      resulting number of elements.
    p->L = c_malloc(sizeof(csc));
    p->L->m = n_plus_m;
    p->L->n = n_plus_m;
    p->L->nz = -1;

    // Diagonal matrix stored as a vector D
    p->Dinv = c_malloc(sizeof(c_float) * n_plus_m);

    // Permutation vector P
    p->P = c_malloc(sizeof(c_int) * n_plus_m);

    // Working vector
    p->bp = c_malloc(sizeof(c_float) * n_plus_m);

    // Solve time (for reporting)
    p->total_solve_time = 0.0;

    // Form KKT matrix
    KKT = form_KKT(P, A, settings->rho, 1);

    // Factorize the KKT matrix
    // TODO: Store factorization timings
    if (factorize(KKT, p) < 0) {
        free_priv(p);
        return OSQP_NULL;
    }

    // Memory clean-up
    csc_spfree(KKT);

    return p;
}

// TODO: Remove this function
// Initialize private variable with given matrix L, and vector D and P
Priv *set_priv(csc *L, c_float *Dinv, c_int *P){
    Priv * p;   // LDL structure
    c_int n = L->n;
    // Allocate pointers
    p = c_calloc(1, sizeof(Priv));
    // Set LDL factorization data: L, D, P
    p->L = L;   // lower triangular matrix (stored without unit diagonal)
    p->Dinv = Dinv;   // diagonal matrix (stored as a vector)
    p->P = P;   // permutation matrix (stored as a vector)
    // Working vector
    p->bp = c_malloc(sizeof(c_float) * n);
    // Solve time (for reporting)
    p->total_solve_time = 0.0;
    return p;
}

// x = Dinv*x
void LDL_dinvsolve(c_int n, c_float *x, c_float *Dinv){
    c_int i;
    for (i = 0 ; i < n ; i++){
        x[i] *= Dinv[i];
    }
}

void LDLSolve(c_float *x, c_float *b, csc *L, c_float *Dinv, c_int *P,
              c_float *bp) {
    /* solves PLDL'P' x = b for x */
    c_int n = L->n;
    LDL_perm(n, bp, b, P);
    LDL_lsolve(n, bp, L->p, L->i, L->x);
    LDL_dinvsolve(n, bp, Dinv);
    LDL_ltsolve(n, bp, L->p, L->i, L->x);
    LDL_permt(n, x, bp, P);
}


/* TODO: Adjust arguments of the function with other linear system solvers */
c_int solve_lin_sys(const Settings *settings, Priv *p, c_float *b) {
    /* returns solution to linear system */
    /* Ax = b with solution stored in b */
    LDLSolve(b, b, p->L, p->Dinv, p->P, p->bp);
    return 0;
}


void polish(Work *work){
    c_int j, ptr, mred=0, Ared_nnz=0;
    c_float *Ax, *prim_resid, *dual_resid, tmp, prim_resid_norm, dual_resid_norm;
    Priv *plsh = c_calloc(1, sizeof(Priv));

    // Initialize counters for active/inactive constraints
    work->act->n_lAct = 0;
    work->act->n_uAct = 0;
    work->act->n_free = 0;
    /* Guess which linear constraints are lower-active, upper-active and free
     *    A2Ared[j] = -1    (if j-th row of A is not inserted in Ared)
     *    A2Ared[j] =  i    (if j-th row of A is inserted at i-th row of Ared)
     */
    for (j = 0; j < work->data->m; j++) {
        if ( work->z[work->data->n + j] - work->data->lA[j] <
             -work->settings->rho * work->u[j] ) {
                work->act->ind_lAct[work->act->n_lAct++] = j;     // lower-active
                work->act->A2Ared[j] = mred++;
        }
        else if ( work->data->uA[j] - work->z[work->data->n + j] <
                  work->settings->rho * work->u[j] ) {
                    work->act->ind_uAct[work->act->n_uAct++] = j; // upper-active
                    work->act->A2Ared[j] = mred++;
        }
        else {
            work->act->ind_free[work->act->n_free++] = j;        // free
            work->act->A2Ared[j] = -1;
        }
    }
    work->act->lambda_red = c_malloc(mred * sizeof(c_float));

    // Count number of elements in Ared
    for (j = 0; j < work->data->A->nzmax; j++) {
        if (work->act->A2Ared[work->data->A->i[j]] != -1)
            Ared_nnz++;
    }
    // Form Ared
    csc *Ared = csc_spalloc(mred, work->data->n, Ared_nnz, 1, 0);
    Ared_nnz = 0;
    for (j = 0; j < work->data->n; j++) {  // Cycle over columns of A
        Ared->p[j] = Ared_nnz;
        for (ptr = work->data->A->p[j]; ptr < work->data->A->p[j + 1]; ptr++) {
            if (work->act->A2Ared[work->data->A->i[ptr]] != -1) {
                // if row of A should be added to Ared
                Ared->i[Ared_nnz] = work->act->A2Ared[work->data->A->i[ptr]];
                Ared->x[Ared_nnz++] = work->data->A->x[ptr];
            }
        }
    }
    Ared->p[work->data->n] = Ared_nnz;

    // Form and factorize reduced KKT
    csc *KKTred= form_KKT(work->data->P, Ared, work->settings->delta, 0);
    c_int n_KKTred = work->data->n + mred;
    plsh->L = c_malloc(sizeof(csc));
    plsh->L->m = n_KKTred;
    plsh->L->n = n_KKTred;
    plsh->L->nz = -1;
    plsh->Dinv = c_malloc(sizeof(c_float) * n_KKTred);
    plsh->P = c_malloc(sizeof(c_int) * n_KKTred);
    plsh->bp = c_malloc(sizeof(c_float) * n_KKTred);
    if (factorize(KKTred, plsh) < 0) {
        free_priv(plsh);
    }

    // Form the rhs of the reduced KKT linear system
    c_float *rhs = c_malloc(sizeof(c_float) * n_KKTred);
    for (j = 0; j < work->data->n; j++) {
        rhs[j] = -work->data->q[j];
    }
    for (j = 0; j < work->act->n_lAct; j++) {
        rhs[work->data->n + j] = work->data->lA[work->act->ind_lAct[j]];
    }
    for (j = 0; j < work->act->n_uAct; j++) {
        rhs[work->data->n + work->act->n_lAct + j] =
            work->data->uA[work->act->ind_uAct[j]];
    }

    // Solve the reduced KKT system
    LDLSolve(rhs, rhs, plsh->L, plsh->Dinv, plsh->P, plsh->bp);
    prea_vec_copy(rhs, work->act->x, work->data->n);
    prea_vec_copy(rhs + work->data->n, work->act->lambda_red, mred);

    // Compute primal residual:  pr = min(Ax-lA, 0) + max(Ax-uA, 0)
    Ax = c_malloc(work->data->m * sizeof(c_float));
    mat_vec(work->data->A, work->act->x, Ax, 0);
    prim_resid = c_calloc(1, work->data->m * sizeof(c_float));
    for (j = 0; j < work->data->m; j++) {
        tmp = Ax[j] - work->data->lA[j];
        if (tmp < 0.)
            prim_resid[j] += tmp;
        tmp = Ax[j] - work->data->uA[j];
        if (tmp > 0.)
            prim_resid[j] += tmp;
    }
    prim_resid_norm = vec_norm2(prim_resid, work->data->m);

    // Compute dual residual:  dr = q + Ared'*lambda_red + P*x
    dual_resid = vec_copy(work->data->q, work->data->n);          // dr = q
    mat_vec_tpose(Ared, work->act->lambda_red, dual_resid, 1, 0); //   += Ared'*lambda
    mat_vec(work->data->P, work->act->x, dual_resid, 1);          //   += P*x (1st part)
    mat_vec_tpose(work->data->P, work->act->x, dual_resid, 1, 1); //   += P*x (2nd part)
    dual_resid_norm = vec_norm2(dual_resid, work->data->n);


    // DEBUG
    c_print("Polished primal residual: %.2e\n", prim_resid_norm);
    c_print("Polished dual residual:   %.2e\n", dual_resid_norm);


    // Check if the residuals are smaller than in the ADMM solution
    if (prim_resid_norm < work->info->pri_res &&
        dual_resid_norm < work->info->dua_res) {
            // Update primal and dual variables
            prea_vec_copy(work->act->x, work->solution->x, work->data->n);
            for (j = 0; j < work->data->m; j++) {
                if (work->act->A2Ared[j] != -1){
                    work->solution->lambda[j] = work->act->lambda_red[work->act->A2Ared[j]];
                } else {
                    work->solution->lambda[j] = 0.;
                }
            }
            // Update solver information
            work->info->pri_res = prim_resid_norm;
            work->info->pri_res = dual_resid_norm;
            work->info->obj_val = quad_form(work->data->P, work->act->x) +
                                  vec_prod(work->data->q, work->act->x, work->data->n);
            // Polishing successful
            work->act->polish_success = 1;
            #if PRINTLEVEL > 1
            c_print("Polishing: Successful.\n");
            #endif
    } else {
        work->act->polish_success = 0;
        #if PRINTLEVEL > 1
        c_print("Polishing: Unsuccessful.\n");
        #endif
    }


    // Memory clean-up
    csc_spfree(Ared);
    csc_spfree(KKTred);
    free_priv(plsh);
    c_free(rhs);
    c_free(Ax);
    c_free(prim_resid);
    c_free(dual_resid);
}
