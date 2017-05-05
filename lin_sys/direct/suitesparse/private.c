#include "private.h"

#ifndef EMBEDDED

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

        // These are required for matrix updates
        if (p->Pdiag_idx)
            c_free(p->Pdiag_idx);
        if (p->KKT)
            csc_spfree(p->KKT);
        if (p->PtoKKT)
            c_free(p->PtoKKT);
        if (p->AtoKKT)
            c_free(p->AtoKKT);
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

        c_free(p);

    }
}


/**
 * Compute LDL factorization of matrix A
 * @param  A Matrix to be factorized
 * @param  p Private workspace
 * @return   [description]
 */
c_int LDL_factor(csc *A,  Priv * p){
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
    LDL_symbolic(n, A->p, A->i, p->L->p, Parent, Lnz, Flag,
         OSQP_NULL, OSQP_NULL);

    p->L->nzmax = p->L->p[n];
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
    //                  p->L->x, p->Dinv, Y, Pattern, Flag, p->P, Pinv);
    kk = LDL_numeric(A->n, A->p, A->i, A->x, p->L->p, Parent, Lnz, p->L->i,
                     p->L->x, p->Dinv, Y, Pattern, Flag, OSQP_NULL, OSQP_NULL);


    // If not embedded option 1 store values into private structure
    p->Parent = Parent;
    p->Lnz = Lnz;
    p->Flag = Flag;
    p->Pattern = Pattern;
    p->Y = Y;

    // return exit flag
    return (kk - n);
}


c_int permute_KKT(csc ** KKT, Priv * p, c_int Pnz, c_int Anz, c_int * PtoKKT, c_int * AtoKKT){
    c_float *info;
    c_int amd_status;
    info = (c_float *)c_malloc(AMD_INFO * sizeof(c_float));
    c_int * Pinv;
    csc *KKT_temp;
    c_int * KtoPKPt;
    c_int i; // Indexing

    // Compute permutation metrix P using AMD
    #ifdef DLONG
    amd_status = amd_l_order((*KKT)->n, (*KKT)->p, (*KKT)->i, p->P, (c_float *)OSQP_NULL, info);
    #else
    amd_status = amd_order((*KKT)->n, (*KKT)->p, (*KKT)->i, p->P, (c_float *)OSQP_NULL, info);
    #endif
    if (amd_status < 0) return (amd_status);


    // Converse of the permutation vector
    Pinv = csc_pinv(p->P, (*KKT)->n);

    // Permute KKT matrix
    if (!PtoKKT && !AtoKKT){  // No vectors to be stored
        // Assign values of mapping
        KKT_temp = csc_symperm((*KKT), Pinv, OSQP_NULL, 1);
    }
    else {
        // Allocate vector of mappings from unpermuted to permuted
        KtoPKPt = c_malloc((*KKT)->p[(*KKT)->n] * sizeof(c_int));
        KKT_temp = csc_symperm((*KKT), Pinv, KtoPKPt, 1);

        // Update vectors PtoKKT and AtoKKT
        for (i = 0; i < Pnz; i++){
            PtoKKT[i] = KtoPKPt[PtoKKT[i]];
        }
        for (i = 0; i < Anz; i++){
            AtoKKT[i] = KtoPKPt[AtoKKT[i]];
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

/**
 * Initialize LDL Factorization structure
 *
 * @param  P        Cost function matrix (upper triangular form)
 * @param  A        Constraints matrix
 * @param  settings Solver settings
 * @param  polish   Flag whether we are initializing for polish or not
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


    // Form and permute KKT matrix
    if (polish){ // Called from polish()
        KKT_temp = form_KKT(P, A, settings->delta, settings->delta, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL);

        // Permute matrix
        permute_KKT(&KKT_temp, p, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL);
    }
    else { // Called from ADMM algorithm

        // Allocate vectors of indeces
        p->PtoKKT = c_malloc((P->p[P->n]) * sizeof(c_int));
        p->AtoKKT = c_malloc((A->p[A->n]) * sizeof(c_int));

        KKT_temp = form_KKT(P, A, settings->sigma, 1./settings->rho,
                            p->PtoKKT, p->AtoKKT,
                            &(p->Pdiag_idx), &(p->Pdiag_n));

        // Permute matrix
        permute_KKT(&KKT_temp, p, P->p[P->n], A->p[A->n], p->PtoKKT, p->AtoKKT);
    }

    // Check if matrix has been created
    if (!KKT_temp){
        #ifdef PRINTING
            c_print("Error forming and permuting KKT matrix!\n");
        #endif
        return OSQP_NULL;
    }


    // Factorize the KKT matrix
    if (LDL_factor(KKT_temp, p) < 0) {
        free_priv(p);
        return OSQP_NULL;
    }

    // Invert elements of D that are stored in p->Dinv
    vec_ew_recipr(p->Dinv, p->Dinv, KKT_temp->n);

    if (polish){ // If KKT passed, assign it to KKT_temp
        // Polish, no need for KKT_temp
        csc_spfree(KKT_temp);
    }
    else { // If not embedded option 1 copy pointer to KKT_temp. Do not free it.
        p->KKT = KKT_temp;
    }


    return p;
}

#endif  // EMBEDDED



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


#if EMBEDDED != 1
// Update private structure with new P and A
c_int update_priv(Priv * p, const csc *P, const csc *A,
                 const OSQPWorkspace * work, const OSQPSettings *settings){
    c_int kk;

    // Update KKT matrix with new P
    update_KKT_P(p->KKT, P, p->PtoKKT, settings->sigma, p->Pdiag_idx, p->Pdiag_n);

    // Update KKT matrix with new A
    update_KKT_A(p->KKT, A, p->AtoKKT);

    // Perform numeric factorization
    kk = LDL_numeric(p->KKT->n, p->KKT->p, p->KKT->i, p->KKT->x,
                     p->L->p, p->Parent, p->Lnz, p->L->i,
                     p->L->x, p->Dinv, p->Y, p->Pattern, p->Flag,
                     OSQP_NULL, OSQP_NULL);

     // Invert elements of D that are stored in p->Dinv
     vec_ew_recipr(p->Dinv, p->Dinv, p->KKT->n);

    // return exit flag
    return (kk - p->KKT->n);

}

#endif
