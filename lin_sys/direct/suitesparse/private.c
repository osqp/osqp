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

        #if EMBEDDED != 1
        // These are required for matrix updates
        if (p->KKT)
            csc_spfree(p->KKT);
        if (p->PtoKKT)
            c_free(p->PtoKKT);
        if (p->AtoKKT)
            c_free(p->AtoKKT);
        if (p->Pinv)
            c_free(p->Pinv);
        if (p->Parent)
            c_free(p->Parent);
        if (p->Lnz)
            c_free(p->Lnz);
        if (p->Flag)
            c_free(p->Flag);
        if (p->Pattern)
            c_free(p->Pattern);
        if (p->Y)
            c_free(p->Y);
        #endif
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
c_int LDLFactor(csc *A,  Priv * p){
    // c_int P[], c_int Pinv[], csc **L, c_float **D) {
    c_int kk, n = A->n;
    c_int check_Li_Lx;
    c_int * Parent = c_malloc(n * sizeof(c_int));
    c_int * Lnz = c_malloc(n * sizeof(c_int));
    c_int * Flag = c_malloc(n * sizeof(c_int));
    c_int * Pattern = c_malloc(n * sizeof(c_int));
    c_float * Y = c_malloc(n * sizeof(c_float));
    p->L->p = (c_int *)c_malloc((1 + n) * sizeof(c_int));

    // Symbolic factorization
    LDL_symbolic(n, A->p, A->i, p->L->p, Parent, Lnz, Flag, p->P, p->Pinv);

    p->L->nzmax = *(p->L->p + n);
    p->L->x = (c_float *)c_malloc(p->L->nzmax * sizeof(c_float));
    p->L->i = (c_int *)c_malloc(p->L->nzmax * sizeof(c_int));

    // If there are no elements in L, i.e. if the matrix A is already diagona, do not check if L->x or L->i are different than zero.
    if (p->L->nzmax == 0) {
        check_Li_Lx = 0;
    }
    else{
        check_Li_Lx = !p->L->i || !p->L->x;
    }

    // Check if symbolic factorization worked our correctly
    if (!(p->Dinv) || check_Li_Lx || !Y || !Pattern || !Flag || !Lnz ||
        !Parent)
        return -1;

    // Numeric factorization
    kk = LDL_numeric(A->n, A->p, A->i, A->x, p->L->p, Parent, Lnz, p->L->i,
                     p->L->x, p->Dinv, Y, Pattern, Flag, p->P, p->Pinv);

    // If not embedded option 1 store values into private structure
    #if EMBEDDED != 1
    p->Parent = Parent;
    p->Lnz = Lnz;
    p->Flag = Flag;
    p->Pattern = Pattern;
    p->Y = Y;
    #endif

    // return exit flag
    return (kk - n);
}



/**
 *  Factorize matrix A using sparse LDL factorization with pivoting as:
 *      P A P' = L D L'
 *  The result is stored in the LDL Factorization structure Priv.
 */
c_int factorize(csc *A, Priv *p) {
    c_float *info;
    c_int amd_status, ldl_status;
    info = (c_float *)c_malloc(AMD_INFO * sizeof(c_float));
    c_int * Pinv_temp;

    // Compute permutation metrix P using SuiteSparse/AMD
    #ifdef DLONG
    amd_status = amd_l_order(A->n, A->p, A->i, p->P, (c_float *)OSQP_NULL, info);
    #else
    amd_status = amd_order(A->n, A->p, A->i, p->P, (c_float *)OSQP_NULL, info);
    #endif
    if (amd_status < 0) return (amd_status);

    // Compute inverse of permutation matrix P
    Pinv_temp = csc_pinv(p->P, A->n);
    #if EMBEDDED != 1 // if not embedded 1 store it (used to update P and A)
        p->Pinv = Pinv_temp;
    #endif

    // Compute LDL factorization of  P A P'
    // NB: D matrix is stored in p->Dinv.
    ldl_status = LDLFactor(A, p);

    // Invert elements of D that are stored in p->Dinv
    vec_ew_recipr(p->Dinv, p->Dinv, A->n);

    // Memory clean-up
    #if EMBEDDED == 1
    c_free(Pinv_temp);
    #endif
    c_free(info);
    return (ldl_status);
}


/**
 * Initialize LDL Factorization structure
 *
 * @param  P        Cost function matrix (upper triangular form)
 * @param  A        Constraints matrix
 * @param  settings Solver settings
 * @param  polish   Flag whether we are initializing for polishing or not
 * @return          Initialized private structure
 */
Priv *init_priv(const csc * P, const csc * A, const OSQPSettings *settings, c_int polish){
    // Define Variables
    Priv * p;                    // KKT factorization structure
    c_int n_plus_m;              // Define n_plus_m dimension
    csc * KKT_temp;              // Temporary KKT pointer

    // Allocate private structure to store KKT factorization
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


    // Form KKT matrix
    if (polish){ // Called from polish()
        KKT_temp = form_KKT(P, A, settings->delta, settings->delta, OSQP_NULL, OSQP_NULL);
    }
    else { // Called from ADMM algorithm
        #if EMBEDDED != 1
        // Allocate vectors of indeces
        p->PtoKKT = c_malloc((P->p[P->n]) * sizeof(c_int));
        p->AtoKKT = c_malloc((A->p[A->n]) * sizeof(c_int));

        KKT_temp = form_KKT(P, A, settings->sigma, 1./settings->rho,
                            p->PtoKKT, p->AtoKKT);
        #else
        KKT_temp = form_KKT(P, A, settings->sigma, 1./settings->rho,
                            OSQP_NULL, OSQP_NULL);
        #endif
    }

    if (KKT_temp == OSQP_NULL){
        #ifdef PRINTING
            c_print("Error forming KKT matrix!\n");
        #endif
        return OSQP_NULL;
    }

    // Factorize the KKT matrix
    if (factorize(KKT_temp, p) < 0) {
        free_priv(p);
        return OSQP_NULL;
    }

    if (polish){ // If KKT passed, assign it to KKT_temp
        // Polish, no need for KKT_temp
        csc_spfree(KKT_temp);
    }
    else { // If not embedded option 1 copy pointer to KKT_temp. Do not free it.
        #if EMBEDDED != 1
        p->KKT = KKT_temp;
        #else
        csc_spfree(KKT_temp);
        #endif
    }

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


c_int solve_lin_sys(const OSQPSettings *settings, Priv *p, c_float *b) {
    /* returns solution to linear system */
    /* Ax = b with solution stored in b */
    LDLSolve(b, b, p->L, p->Dinv, p->P, p->bp);
    return 0;
}


// Update private structure with new P and A
c_int update_priv(Priv * p, const csc *P, const csc *A,
                 const OSQPWorkspace * work, const OSQPSettings *settings){
    c_int kk;

    // Update KKT matrix with new P
    update_KKT_P(p->KKT, P, p->PtoKKT, settings->sigma, work->Pdiag_idx, work->Pdiag_n);

    // Update KKT matrix with new A
    update_KKT_A(p->KKT, A, p->AtoKKT);

    // Perform numeric factorization
    kk = LDL_numeric(p->KKT->n, p->KKT->p, p->KKT->i, p->KKT->x,
                     p->L->p, p->Parent, p->Lnz, p->L->i,
                     p->L->x, p->Dinv, p->Y, p->Pattern, p->Flag, p->P, p->Pinv);

    // return exit flag
    return (kk - p->KKT->n);

}
