#include "private.h"
#include "util.h"

// TODO: Add functions for defining factorizing and solving linear systems with direct methods

// form_KKT, factorize... (see scs)



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
A: linear equalities matrix
rho: ADMM step
N.B. Only the upper triangular part is stuffed!
*/
csc * form_KKT(const csc * P, const  csc * A, c_float rho){
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

    // /* -1./rho I at bottom right */
    for (j = 0; j < A->m; j++) {
        KKT_trip->i[z_KKT] = j + P->n;
        KKT_trip->p[z_KKT] = j + P->n;
        KKT_trip->x[z_KKT] = -1./rho;
        z_KKT++;
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
    *D = (c_float *)c_malloc(n * sizeof(c_float));

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
















/*  Reduced KKT with regularization delta
 *             [ P + delta*I   Ared'  ]
 *  KKT_red =  [    Ared     -delta*I ]
 */
void form_redKKT(Work * work) {
    c_int ptr, i, j;
    c_int z_P=0, z_KKT=0;   // Counter for total number of elements in P and in KKT

    // Set (1,1) block of reduced KKT matrix: P + delta I
    for (j = 0; j < work->data->n; j++) {    // cycle over columns

        // No elements in column j => add diagonal element delta
        if (work->data->P->p[j] == work->data->P->p[j+1]) {
            work->plsh->KKT_trip->i[z_KKT] = j;
            work->plsh->KKT_trip->p[z_KKT] = j;
            work->plsh->KKT_trip->x[z_KKT++] = work->settings->delta;
        }

        for (ptr = work->data->P->p[j]; ptr < work->data->P->p[j + 1]; ptr++) {

            // Get current row
            i = work->data->P->i[ptr];

            // Add element of P
            work->plsh->KKT_trip->i[z_KKT] = i;
            work->plsh->KKT_trip->p[z_KKT] = j;
            work->plsh->KKT_trip->x[z_KKT] = work->data->P->x[z_P];
            if (i == j) { // P has a diagonal element, add rho
                work->plsh->KKT_trip->x[z_KKT] += work->settings->delta;
            }
            z_P++;
            z_KKT++;

            // Add diagonal rho in case
            if ((i < j) && // Diagonal element not reached
                (ptr + 1 == work->data->P->p[j+1])) { // last element of column j

                // Add diagonal element rho
                work->plsh->KKT_trip->i[z_KKT] = j;
                work->plsh->KKT_trip->p[z_KKT] = j;
                work->plsh->KKT_trip->x[z_KKT] = work->settings->delta;
                z_KKT++;
            }
        }
    }

    // Set (1,2) block to Ared'
    for (j = 0; j < work->data->n; j++) {  // Cycle over columns of Ared
        for (ptr = work->data->A->p[j]; ptr < work->data->A->p[j + 1]; ptr++) {
            if (work->plsh->tableA[work->data->A->i[ptr]] != -1) {
                // if row of A should be added to Ared
                work->plsh->KKT_trip->p[z_KKT] = work->data->P->m
                    + work->plsh->tableA[work->data->A->i[ptr]];
                work->plsh->KKT_trip->i[z_KKT] = j;
                work->plsh->KKT_trip->x[z_KKT++] = work->data->A->x[ptr];
            }
        }
    }

    // Set (2,2) block to -delta*I
    for (j = 0; j < work->plsh->n_lA + work->plsh->n_uA; j++) {
        work->plsh->KKT_trip->i[z_KKT] = j + work->data->n;
        work->plsh->KKT_trip->p[z_KKT] = j + work->data->n;
        work->plsh->KKT_trip->x[z_KKT++] = -work->settings->delta;
    }

}



// Print int array
// TODO: This function is only for debugging. To be removed.
void print_vec_int(c_int * x, c_int n, char *name) {
    c_print("%s = [", name);
    for(c_int i=0; i<n; i++) {
        c_print(" %d ", x[i]);
    }
    c_print("]\n");
}


// Initialize polishing structure
Polish *init_polish(const csc * P, const csc * A) {
    // Allocate memory for polishing structure
    Polish * plsh = c_calloc(1, sizeof(Polish));

    // Dimensions of matrices
    c_int m_red = c_min(A->m, A->n);   // upper bound on number of rows in reduced A
    c_int nKKT = m_red + A->n;  // reduced KKT ==> [P Ared'; Ared 0]

    // Maximum number of nnz elements in L and KKT
    c_int Ared_nz = c_min(A->n * m_red, A->nzmax);
    c_int Lred_nz = P->nzmax + Ared_nz;
    c_int KKTred_nz = Lred_nz + nKKT;

    // Allocate memory for storing reduced KKT in triplet format
    plsh->KKT_trip = csc_spalloc(nKKT, nKKT, KKTred_nz, 1, 1);

    // Allocate memory for storing reduced KKT in CSC format
    plsh->KKT = csc_spalloc(nKKT, nKKT, KKTred_nz, 1, 0);

    // Allocate memory for storing LDL factorization of reduced KKT
    plsh->L = csc_spalloc(nKKT, nKKT, Lred_nz, 1, 0); // Lower triang matrix
    plsh->Dinv = c_malloc(sizeof(c_float) * nKKT);    // Inverse of diag matrix
    plsh->P = c_malloc(sizeof(c_int) * nKKT);         // Permutation vector
    plsh->bp = c_malloc(sizeof(c_float) * nKKT);    // Working vector

    // Allocate memory for active constraints
    plsh->n_lA = 0;
    plsh->n_uA = 0;
    plsh->n_fA = 0;
    plsh->ind_lA = c_calloc(1, A->m * sizeof(c_int));
    plsh->ind_uA = c_calloc(1, A->m * sizeof(c_int));
    plsh->ind_fA = c_calloc(1, A->m * sizeof(c_int));
    plsh->tableA = c_calloc(1, A->m * sizeof(c_int));

    return plsh;
}


c_int solve_polish(Work *work) {
    c_int i, cnt=0;

    // Guess which linear constraints are lower-active, upper-active and free
    for (i = 0; i < work->data->m; i++) {
        if ( work->z[work->data->n + i] - work->data->lA[i] <
             -work->settings->rho * work->u[work->data->n + i] ) {
                work->plsh->ind_lA[work->plsh->n_lA++] = i;     // lower-active
                work->plsh->tableA[i] = cnt++;
        }
        else if ( work->data->uA[i] - work->z[work->data->n + i] <
                  work->settings->rho * work->u[work->data->n + i] ) {
                    work->plsh->ind_uA[work->plsh->n_uA++] = i; // upper-active
                    work->plsh->tableA[i] = cnt++;
        }
        else {
            work->plsh->ind_fA[work->plsh->n_fA++] = i;         // free
            work->plsh->tableA[i] = -1;
        }
    }

    // DEBUG
    print_vec_int(work->plsh->ind_lA, work->plsh->n_lA, "ind_lA");
    print_vec_int(work->plsh->ind_uA, work->plsh->n_uA, "ind_uA");
    print_vec_int(work->plsh->ind_fA, work->plsh->n_fA, "ind_fA");
    print_vec_int(work->plsh->tableA, work->data->m, "tableA");
    c_print("\n");
    print_vec(work->z + work->data->n, work->data->m, "Ax");
    print_vec(work->data->uA, work->data->m, "uA");


    // TODO: Form reduced KKT matrix


    // Check whether the dual vars stored in rhs have correct signs

    // If yes, update solution. Otherwise, keep the old solution.
    return 0;
}

// Free polishing structure
void free_polish(Polish *plsh) {
    if (plsh) {
        if (plsh->KKT_trip)
            csc_spfree(plsh->KKT_trip);
        if (plsh->KKT)
            csc_spfree(plsh->KKT);
        if (plsh->Dinv)
            c_free(plsh->Dinv);
        if (plsh->P)
            c_free(plsh->P);
        if (plsh->bp)
            c_free(plsh->bp);
        if (plsh->ind_lA)
            c_free(plsh->ind_lA);
        if (plsh->ind_uA)
            c_free(plsh->ind_uA);
        if (plsh->ind_fA)
            c_free(plsh->ind_fA);
        c_free(plsh);
    }
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
    // Set nzmax to 1 and null pointer to elements (to be filled during factorization)
    p->L = csc_spalloc(n_plus_m, n_plus_m, 1, 0, 0);

    // Diagonal matrix stored as a vector D
    p->Dinv = c_malloc(sizeof(c_float) * n_plus_m);

    // Permutation vector P
    p->P = c_malloc(sizeof(c_int) * n_plus_m);

    // Working vector
    p->bp = c_malloc(sizeof(c_float) * n_plus_m);

    // Solve time (for reporting)
    p->total_solve_time = 0.0;

    // Form KKT matrix
    KKT = form_KKT(P, A, settings->rho);

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
