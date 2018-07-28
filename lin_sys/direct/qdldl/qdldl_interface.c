#include "glob_opts.h"

#include "qdldl.h"
#include "qdldl_interface.h"

#ifndef EMBEDDED
#include "amd.h"
#endif

#include "lin_alg.h"

//Need to expose internal matrix implementation of OSQPMatrix
//in order to use the CSC specific QDLDL solver
#include "lin_alg_impl.h"

#if EMBEDDED != 1
#include "kkt.h"
#endif

#ifndef EMBEDDED

// Free LDL Factorization structure
void free_linsys_solver_qdldl(qdldl_solver *s) {
    if (s) {

        //The matrix factorisation goes here
        if (s->L)         OSQPMatrix_free(s->L);
        if (s->P)         OSQPVectori_free(s->P);
        if (s->Dinv)      OSQPVectorf_free(s->Dinv);

        //working memory for solves
        if (s->bp)        OSQPVectorf_free(s->bp);

        // These are required for matrix updates
        if (s->Pdiag_idx) OSQPVectori_free(s->Pdiag_idx);
        if (s->KKT)       OSQPMatrix_free(s->KKT);
        if (s->PtoKKT)    OSQPVectori_free(s->PtoKKT);
        if (s->AtoKKT)    OSQPVectori_free(s->AtoKKT);
        if (s->rhotoKKT)  OSQPVectori_free(s->rhotoKKT);

        //These are working memory for the LDL solver
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
 * @param  A Matrix to be factorized
 * @param  p Private workspace
 * @return   exitstatus (0 is good)
 */
static c_int LDL_factor(OSQPMatrix *A,  qdldl_solver * p){

    c_int sum_Lnz;
    c_int factor_status;

    //pointer to inner CSC structure for A and L
    CscMatrix* Acsc = A->csc;
    CscMatrix* Lcsc;

    // Compute elimination tree
    sum_Lnz = QDLDL_etree(Acsc->n, Acsc->p, Acsc->i, p->iwork, p->Lnz, p->etree);

    if (sum_Lnz < 0){
      // Error
      c_eprint("Error in KKT matrix LDL factorization when computing the elimination tree. A is not perfectly upper triangular");
      return sum_Lnz;
    }

    // Allocate memory for L
    p->L = OSQPMatrix_new(Acsc->n, Acsc->n, sum_Lnz);
    Lcsc = p->L->csc;

    // Factor matrix
    factor_status = QDLDL_factor(Acsc->n, Acsc->p, Acsc->i, Acsc->x,
                                 Lcsc->p, Lcsc->i, Lcsc->x,
                                 p->D,
                                 OSQPVectorf_data(p->Dinv),
                                 p->Lnz,
                                 p->etree,
                                 (QDLDL_bool *)p->bwork,
                                 p->iwork,
                                 p->fwork);


    if (factor_status < 0){
      // Error
      c_eprint("Error in KKT matrix LDL factorization when in computing the nonzero elements. There are zeros in the diagonal matrix");
      return factor_status;
    }

    return 0;

}


static c_int permute_KKT(csc ** KKT, qdldl_solver * p, c_int Pnz, c_int Anz, c_int m, c_int * PtoKKT, c_int * AtoKKT, c_int * rhotoKKT){
    c_float *info;
    c_int amd_status;
    c_int * Pinv;
    csc *KKT_temp;
    c_int * KtoPKPt;
    c_int i; // Indexing

    info = (c_float *)c_malloc(AMD_INFO * sizeof(c_float));

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
qdldl_solver *init_linsys_solver_qdldl(const csc * P, const csc * A, c_float sigma, c_float * rho_vec, c_int polish){

    c_int i;                     // loop counter

    // Define Variables
    qdldl_solver * p;            // Initialize LDL solver
    c_int n_plus_m;              // Define n_plus_m dimension
    csc * KKT_temp;              // Temporary KKT pointer

    // Allocate private structure to store KKT factorization
    p = c_calloc(1, sizeof(qdldl_solver));

    // Size of KKT
    n_plus_m = P->m + A->m;

    // Sparse matrix L (lower triangular)
    // NB: We don not allocate L completely (CSC elements)
    //      L will be allocated during the factorization depending on the
    //      resulting number of elements.
    p->L = c_malloc(sizeof(csc));
    p->L->m = n_plus_m;
    p->L->n = n_plus_m;
    p->L->nz = -1;

    // Diagonal matrix stored as a vector D
    p->Dinv = c_malloc(sizeof(c_float) * n_plus_m);
    p->D = c_malloc(sizeof(c_float) * n_plus_m);

    // Permutation vector P
    p->P = c_malloc(sizeof(c_int) * n_plus_m);

    // Working vector
    p->bp = c_malloc(sizeof(c_float) * n_plus_m);

    // Elimination tree workspace
    p->etree = (c_int *)c_malloc(n_plus_m * sizeof(c_int));
    p->Lnz = (c_int *)c_malloc(n_plus_m * sizeof(c_int));

    // Preallocate L matrix (Lx and Li are sparsity dependent)
    p->L->p = (c_int *)c_malloc((n_plus_m+1) * sizeof(c_int));

    //Lx and Li are sparsity dependent, so set them to
    //null initially so we don't try to free them prematurely
    p->L->i = OSQP_NULL;
    p->L->x = OSQP_NULL;


    // Preallocate workspace
    p->iwork = (c_int *)c_malloc(sizeof(c_int)*(3*n_plus_m));
    p->bwork = (c_int *)c_malloc(sizeof(c_int)*n_plus_m);
    p->fwork = (c_float *)c_malloc(sizeof(c_float)*n_plus_m);

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
        free_linsys_solver_qdldl(p);
        return OSQP_NULL;
    }

    if (polish){ // If KKT passed, assign it to KKT_temp
        // Polish, no need for KKT_temp
        csc_spfree(KKT_temp);
    }
    else { // If not embedded option 1 copy pointer to KKT_temp. Do not free it.
        p->KKT = KKT_temp;
    }

    // Link Functions
    p->solve = &solve_linsys_qdldl;

    #ifndef EMBEDDED
    p->free = &free_linsys_solver_qdldl;
    #endif

    #if EMBEDDED != 1
    p->update_matrices = &update_linsys_solver_matrices_qdldl;
    p->update_rho_vec = &update_linsys_solver_rho_vec_qdldl;
    #endif

    // Assign type
    p->type = QDLDL_SOLVER;
    //
    // Set number of threads to 1 (single threaded)
    p->nthreads = 1;

    return p;
}

#endif  // EMBEDDED


static void LDLSolve(c_float *x, c_float *b, csc *L, c_float *Dinv, c_int *P,
              c_float *bp) {
    /* solves PLDL'P' x = b for x */
    OSQPVectorf_perm(bp,b,P);
    QDLDL_solve(L->n, L->p, L->i, L->x, Dinv, bp);
    OSQPVectorf_iperm(x,bp,P);

}

c_int solve_linsys_qdldl(qdldl_solver * s, c_float * b, const OSQPSettings *settings) {
    /* returns solution to linear system */
    /* Ax = b with solution stored in b */
    LDLSolve(b, b, s->L, s->Dinv, s->P, s->bp);

    return 0;
}


#if EMBEDDED != 1
// Update private structure with new P and A
c_int update_linsys_solver_matrices_qdldl(qdldl_solver * s,
		const csc *P, const csc *A, const OSQPSettings *settings){
    c_int kk;

    // Update KKT matrix with new P
    update_KKT_P(s->KKT, P, s->PtoKKT, settings->sigma, s->Pdiag_idx, s->Pdiag_n);

    // Update KKT matrix with new A
    update_KKT_A(s->KKT, A, s->AtoKKT);

    return (QDLDL_factor(s->KKT->n, s->KKT->p, s->KKT->i, s->KKT->x,
        s->L->p, s->L->i, s->L->x, s->D, s->Dinv, s->Lnz,
        s->etree, (QDLDL_bool *)s->bwork, s->iwork, s->fwork) < 0);

}



c_int update_linsys_solver_rho_vec_qdldl(qdldl_solver * s, const OSQPVectorf* rho_vec){
    c_int kk, i;

    // Use s->bp for storing param2 = rho_inv_vec
    OSQPVectorf_ew_reciprocal(rho_vec,s->bp);

    // Update KKT matrix with new rho
    update_KKT_param2(s->KKT, s->bp, s->rhotoKKT);

    return (QDLDL_factor(s->KKT->n, s->KKT->p, s->KKT->i, s->KKT->x,
        s->L->p, s->L->i, s->L->x, s->D, s->Dinv, s->Lnz,
        s->etree, (QDLDL_bool *)s->bwork, s->iwork, s->fwork) < 0);
}


#endif
