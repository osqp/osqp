#include "ldl.h"

#ifndef EMBEDDED
#include "amd.h"
#endif

#include "lin_alg.h"
#include "suitesparse_ldl.h"

#if EMBEDDED != 1
#include "kkt.h"
#endif

#ifndef EMBEDDED

// Free LDL Factorization structure
void free_linsys_solver_suitesparse_ldl(suitesparse_ldl_solver *s) {
    if (s) {
        if (s->L)         csc_spfree(s->L);
        if (s->P)         c_free(s->P);
        if (s->Dinv)      c_free(s->Dinv);
        if (s->bp)        c_free(s->bp);

        // These are required for matrix updates
        if (s->Pdiag_idx) c_free(s->Pdiag_idx);
        if (s->KKT)       csc_spfree(s->KKT);
        if (s->PtoKKT)    c_free(s->PtoKKT);
        if (s->AtoKKT)    c_free(s->AtoKKT);
        if (s->rhotoKKT)  c_free(s->rhotoKKT);
        if (s->Parent)    c_free(s->Parent);
        if (s->Lnz)       c_free(s->Lnz);
        if (s->Flag)      c_free(s->Flag);
        if (s->Pattern)   c_free(s->Pattern);
        if (s->Y)         c_free(s->Y);

        c_free(s);

    }
}


/**
 * Compute LDL factorization of matrix A
 * @param  A Matrix to be factorized
 * @param  p Private workspace
 * @return   [description]
 */
static c_int LDL_factor(csc *A,  suitesparse_ldl_solver * p){
    // c_int P[], c_int Pinv[], csc **L, c_float **D) {
    c_int kk, n = A->n;
    c_int check_Li_Lx;
    c_int * Parent = c_malloc(n * sizeof(c_int));
    c_int * Lnz = c_malloc(n * sizeof(c_int));
    c_int * Flag = c_malloc(n * sizeof(c_int));
    c_int * Pattern = c_malloc(n * sizeof(c_int));
    c_float * Y = c_malloc(n * sizeof(c_float));
    p->L->p = (c_int *)c_malloc((1 + n) * sizeof(c_int));

    // Set number of threads to 1 (single threaded)
    p->nthreads = 1;

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


static c_int permute_KKT(csc ** KKT, suitesparse_ldl_solver * p, c_int Pnz, c_int Anz, c_int m, c_int * PtoKKT, c_int * AtoKKT, c_int * rhotoKKT){
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


    // Inverse of the permutation vector
    Pinv = csc_pinv(p->P, (*KKT)->n);

    // Permute KKT matrix
    if (!PtoKKT && !AtoKKT && !rhotoKKT){  // No vectors to be stored
        // Assign values of mapping
        KKT_temp = csc_symperm((*KKT), Pinv, OSQP_NULL, 1);
    }
    else {
        // Allocate vector of mappings from unpermuted to permuted
        KtoPKPt = c_malloc((*KKT)->p[(*KKT)->n] * sizeof(c_int));
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
suitesparse_ldl_solver *init_linsys_solver_suitesparse_ldl(const csc * P, const csc * A, c_float sigma, c_float * rho_vec, c_int polish){
    c_int i;                     // loop counter
    // Define Variables
    suitesparse_ldl_solver * p;  // Initialize LDL solver
    c_int n_plus_m;              // Define n_plus_m dimension
    csc * KKT_temp;              // Temporary KKT pointer

    // Allocate private structure to store KKT factorization
    p = c_calloc(1, sizeof(suitesparse_ldl_solver));

    // Size of KKT
    n_plus_m = P->m + A->m;

    // Sparse matrix L (lower triangular)
    // NB: Do not allocate L completely (CSC elements)
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
        // Use p->bp for storing param2 = vec(delta)
        for (i = 0; i < A->m; i++){
            p->bp[i] = sigma;
        }

        KKT_temp = form_KKT(P, A, 0, sigma, p->bp, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL);

        // Permute matrix
        permute_KKT(&KKT_temp, p, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL);
    }
    else { // Called from ADMM algorithm

        // Allocate vectors of indices
        p->PtoKKT = c_malloc((P->p[P->n]) * sizeof(c_int));
        p->AtoKKT = c_malloc((A->p[A->n]) * sizeof(c_int));
        p->rhotoKKT = c_malloc((A->m) * sizeof(c_int));

        // Use p->bp for storing param2 = rho_inv_vec
        for (i = 0; i < A->m; i++){
            p->bp[i] = 1. / rho_vec[i];
        }

        KKT_temp = form_KKT(P, A, 0, sigma, p->bp,
                            p->PtoKKT, p->AtoKKT,
                            &(p->Pdiag_idx), &(p->Pdiag_n), p->rhotoKKT);

        // Permute matrix
        permute_KKT(&KKT_temp, p, P->p[P->n], A->p[A->n], A->m, p->PtoKKT, p->AtoKKT, p->rhotoKKT);
    }

    // Check if matrix has been created
    if (!KKT_temp){
        #ifdef PRINTING
            c_eprint("Error forming and permuting KKT matrix");
        #endif
        return OSQP_NULL;
    }


    // Factorize the KKT matrix
    if (LDL_factor(KKT_temp, p) < 0) {
        csc_spfree(KKT_temp);
        free_linsys_solver_suitesparse_ldl(p);
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

    // Link Functions
    p->solve = &solve_linsys_suitesparse_ldl;

    #ifndef EMBEDDED
    p->free = &free_linsys_solver_suitesparse_ldl;
    #endif

    #if EMBEDDED != 1
    p->update_matrices = &update_linsys_solver_matrices_suitesparse_ldl;
    p->update_rho_vec = &update_linsys_solver_rho_vec_suitesparse_ldl;
    #endif

    // Assign type
    p->type = SUITESPARSE_LDL_SOLVER;

    return p;
}

#endif  // EMBEDDED



static void LDLSolve(c_float *x, c_float *b, csc *L, c_float *Dinv, c_int *P,
              c_float *bp) {
    /* solves PLDL'P' x = b for x */
    c_int n = L->n;

    LDL_perm(n, bp, b, P);
    LDL_lsolve(n, bp, L->p, L->i, L->x);
    LDL_dinvsolve(n, bp, Dinv);
    LDL_ltsolve(n, bp, L->p, L->i, L->x);
    LDL_permt(n, x, bp, P);

}


c_int solve_linsys_suitesparse_ldl(suitesparse_ldl_solver * s, c_float * b, const OSQPSettings *settings) {
    /* returns solution to linear system */
    /* Ax = b with solution stored in b */
    LDLSolve(b, b, s->L, s->Dinv, s->P, s->bp);

    return 0;
}


#if EMBEDDED != 1
// Update private structure with new P and A
c_int update_linsys_solver_matrices_suitesparse_ldl(suitesparse_ldl_solver * s,
		const csc *P, const csc *A, const OSQPSettings *settings){
    c_int kk;

    // Update KKT matrix with new P
    update_KKT_P(s->KKT, P, s->PtoKKT, settings->sigma, s->Pdiag_idx, s->Pdiag_n);

    // Update KKT matrix with new A
    update_KKT_A(s->KKT, A, s->AtoKKT);

    // Perform numeric factorization
    kk = LDL_numeric(s->KKT->n, s->KKT->p, s->KKT->i, s->KKT->x,
                     s->L->p, s->Parent, s->Lnz, s->L->i,
                     s->L->x, s->Dinv, s->Y, s->Pattern, s->Flag,
                     OSQP_NULL, OSQP_NULL);

     // Invert elements of D that are stored in s->Dinv
     vec_ew_recipr(s->Dinv, s->Dinv, s->KKT->n);

    // return exit flag
    return (kk - s->KKT->n);

}



c_int update_linsys_solver_rho_vec_suitesparse_ldl(suitesparse_ldl_solver * s, const c_float * rho_vec, const c_int m){
    c_int kk, i;

    // Use s->bp for storing param2 = rho_inv_vec
    for (i = 0; i < m; i++){
        s->bp[i] = 1. / rho_vec[i];
    }

    // Update KKT matrix with new rho
    update_KKT_param2(s->KKT, s->bp, s->rhotoKKT, m);

    // Perform numeric factorization
    kk = LDL_numeric(s->KKT->n, s->KKT->p, s->KKT->i, s->KKT->x,
                     s->L->p, s->Parent, s->Lnz, s->L->i,
                     s->L->x, s->Dinv, s->Y, s->Pattern, s->Flag,
                     OSQP_NULL, OSQP_NULL);

     // Invert elements of D that are stored in s->Dinv
     vec_ew_recipr(s->Dinv, s->Dinv, s->KKT->n);

    // return exit flag
    return (kk - s->KKT->n);
}


#endif
