#include "kkt.h"


/**
 * Form square symmetric KKT matrix of the form
 *
 * [P + scalar1 I,         A';
 *  A             -scalar2 I]
 *
 * N.B. Only the upper triangular part is stuffed!
 *
 * @param  P         cost matrix (already just upper triangular part)
 * @param  A         linear constraint matrix
 * @param  scalar1   regularization parameter scalar1
 * @param  scalar2   regularization parameter scalar2
 * @param P2KKT      index mapping from elements of P to KKT matrix
 * @param A2KT       index mapping from elements of A to KKT matrix
 * @param P_diag_idx index mapping which elements of P are on the diagonal
 * @return           return status flag
 */
csc * form_KKT(const csc * P, const  csc * A, c_float scalar1, c_float scalar2
            //    c_int * P2KKT, c_int * A2KKT, c_int * P_diag_idx
           ){
    c_int nKKT, nnzKKTmax; // Size, number of nonzeros and max number of nonzeros in KKT matrix
    csc *KKT_trip, *KKT;           // KKT matrix in triplet format and CSC format
    c_int ptr, i, j; // Counters for elements (i,j) and index pointer
    c_int z_KKT=0;   // Counter for total number of elements in P and in KKT

    // Get matrix dimensions
    nKKT = P->m + A->m;

    // Get maximum number of nonzero elements (only upper triangular part)
    nnzKKTmax = P->nzmax +           // Number of elements in P
                P->m +               // Number of elements in scalar1 * I
                A->nzmax +           // Number of nonzeros in A
                A->m;                // Number of elements in - scalar2 * I

    // Preallocate KKT matrix in triplet format
    KKT_trip = csc_spalloc(nKKT, nKKT, nnzKKTmax, 1, 1);

    if (!KKT_trip) return OSQP_NULL;  // Failed to preallocate matrix

    // Allocate Triplet matrices
    // P + scalar1 I
    for (j = 0; j < P->n; j++){ // cycle over columns
        // No elements in column j => add diagonal element scalar1
        if (P->p[j] == P->p[j+1]){
            KKT_trip->i[z_KKT] = j;
            KKT_trip->p[z_KKT] = j;
            KKT_trip->x[z_KKT] = scalar1;
            z_KKT++;
        }
        for (ptr = P->p[j]; ptr < P->p[j + 1]; ptr++) { // cycle over rows
            // Get current row
            i = P->i[ptr];

            // Add element of P
            KKT_trip->i[z_KKT] = i;
            KKT_trip->p[z_KKT] = j;
            KKT_trip->x[z_KKT] = P->x[ptr];
            if (i == j){ // P has a diagonal element, add scalar1
                KKT_trip->x[z_KKT] += scalar1;
            }
            z_KKT++;

            // Add diagonal scalar1 in case
            if ((i < j) && // Diagonal element not reached
                (ptr + 1 == P->p[j+1])){ // last element of column j

                // Add diagonal element scalar1
                KKT_trip->i[z_KKT] = j;
                KKT_trip->p[z_KKT] = j;
                KKT_trip->x[z_KKT] = scalar1;
                z_KKT++;
            }
        }
    }


    // A' at top right
    for (j = 0; j < A->n; j++) {  // Cycle over columns of A
        for (ptr = A->p[j]; ptr < A->p[j + 1]; ptr++) {
            KKT_trip->p[z_KKT] = P->m + A->i[ptr];  // Assign column index from row index of A
            KKT_trip->i[z_KKT] = j; // Assign row index from column index of A
            KKT_trip->x[z_KKT] = A->x[ptr];  // Assign A value element
            z_KKT++;
        }
    }

    // - scalar2 * I at bottom right
    for (j = 0; j < A->m; j++) {
        KKT_trip->i[z_KKT] = j + P->n;
        KKT_trip->p[z_KKT] = j + P->n;
        KKT_trip->x[z_KKT] = - scalar2;
        z_KKT++;
    }


    // Allocate number of nonzeros
    KKT_trip->nz = z_KKT;

    // Convert triplet matrix to csc format
    KKT = triplet_to_csc(KKT_trip);

    // Clean matrix in triplet format and return result
    csc_spfree(KKT_trip);
    return KKT;

}
