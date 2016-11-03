#include "private.h"
#include "util.h"

// TODO: Add functions for defining factorizing and solving linear systems with direct methods

// formKKT, factorize... (see scs)


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
csc * formKKT(csc * P, csc * A, c_float rho){
    c_int nKKT, nnzKKTmax; // Size, number of nonzeros and max number of nonzeros in KKT matrix
    csc *KKT_trip, *KKT;           // KKT matrix in triplet format and CSC format
    c_int i, j, z=0;   // Counters for elements (i,j) and for total number of elements z

    // Get matrix dimensions
    nKKT = P->n + A->n;

    // Get maximum number of nonzero elements (only upper triangular part)
    nnzKKTmax = P->nnz +           // Number of elements in P
                P->n +             // Number of elements in rhoI
                A->nnz +           // Number of nonzeros in A
                A->n;               // Number of elements in -1/rho I

    // Preallocate KKT matrix in triplet format
    KKT_trip = csc_spalloc(nKKT, nKKT, nnzKKTmax, 1, 1);

    #if PRINTLEVEL > 2
        c_print("Forming KKT matrix\n");
    #endif

    if (!KKT_trip) return OSQP_NULL;  // Failed to preallocate matrix

    // Allocate Triplet matrices
    // P + rho I
    for (j = 0; j < P->n; j++){ // cycle over columns
            for (i = P->p[j]; i < P->p[j + 1]; i++) { // cycle over rows
                // Add element of P
                KKT_trip->i[z] = i;
                KKT_trip->p[z] = j;
                KKT_trip->x[z] = P->x[z];
                if (i == j){ // P has a diagonal element, add rho
                    KKT_trip->x[z] += rho;
                }
                z++;

                // Add diagonal rho in case
                if ((i < j) && // Diagonal element not reached
                         (P->p[j+1] == i + 1 )){ // last element of column j
                    // Add diagonal element rho
                    KKT_trip->i[z] = j;
                    KKT_trip->p[z] = j;
                    KKT_trip->x[z] = rho;
                    z++;
            }
        }
    }

    // Allocate number of nonzeros
    KKT_trip->nnz = z;

    // Convert triplet matrix to csc format
    KKT = triplet_to_csc(KKT_trip);

    // Clean matrix in triplet format and return result
    csc_spfree(KKT_trip);
    return KKT;

}

// /* TODO: Adjust arguments of the function with other linear system solvers */
// c_int solveLinSys(const c_priv *p, scs_float *b) {
//     /* returns solution to linear system */
//     /* Ax = b with solution stored in b */
//     LDLSolve(b, b, p->L, p->D, p->P, p->bp);
//     return 0;
// }
//
// void LDLSolve(c_float *x, c_float *b, csc *L, c_float *D, c_int *P,
//               c_float *bp) {
//     /* solves PLDL'P' x = b for x */
//     c_int n = L->n;
//     LDL_perm(n, bp, b, P);
//     LDL_lsolve(n, bp, L->p, L->i, L->x);
//     LDL_dsolve(n, bp, D);
//     LDL_ltsolve(n, bp, L->p, L->i, L->x);
//     LDL_permt(n, x, bp, P);
// }
