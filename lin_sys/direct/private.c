#include "private.h"



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
    c_int check_Li_Lx;
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

    // If there are no elements in L, i.e. if the matrix A is already diagona, do not check if L->x or L->i are different than zero.
    if ((*L)->nzmax == 0) {
        check_Li_Lx = 0;
    }
    else{
        check_Li_Lx = !(*L)->i || !(*L)->x;
    }

    // Check if symbolic factorization worked our correctly
    if (!(*D) || check_Li_Lx || !Y || !Pattern || !Flag || !Lnz ||
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
    #ifdef DLONG
    amd_status = amd_l_order(A->n, A->p, A->i, p->P, (c_float *)OSQP_NULL, info);
    #else
    amd_status = amd_order(A->n, A->p, A->i, p->P, (c_float *)OSQP_NULL, info);
    #endif
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
Priv *init_priv(const csc * P, const csc * A, const Settings *settings,
                c_int polish){
    // Define Variables
    csc * KKT;       // KKT Matrix
    Priv * p;        // KKT factorization structure
    c_int n_plus_m;  // Define n_plus_m dimension

    // Allocate private structure to store KKT factorization
    // Allocate pointers
    p = c_calloc(1, sizeof(Priv));

    // Size of KKT
    n_plus_m = P->m + A->m;

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
    // p->total_solve_time = 0.0;

    // Form KKT matrix
    if (!polish)
        // Called from ADMM algorithm
        KKT = form_KKT(P, A, settings->sigma, 1./settings->rho);
    else
        // Called from polish()
        KKT = form_KKT(P, A, settings->delta, settings->delta);

    // Factorize the KKT matrix
    if (factorize(KKT, p) < 0) {
        free_priv(p);
        return OSQP_NULL;
    }

    // Memory clean-up
    csc_spfree(KKT);

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
