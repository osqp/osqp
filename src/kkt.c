#include "kkt.h"


/**
 * Form square symmetric KKT matrix of the form
 *
 * [P + scalar1 I,         A';
 *  A             -scalar2 I]
 *
 * N.B. Only the upper triangular part is stuffed!
 *
 * @param  P          cost matrix (already just upper triangular part)
 * @param  A          linear constraint matrix
 * @param  scalar1    regularization parameter scalar1
 * @param  scalar2    regularization parameter scalar2
 * @param  PtoKKT     (modified) index mapping from elements of P to KKT matrix
 * @param  AtoKKT     (modified) index mapping from elements of A to KKT matrix
 * @return            return status flag
 */
csc * form_KKT(const csc * P, const  csc * A, c_float scalar1, c_float scalar2,
               c_int * PtoKKT, c_int * AtoKKT){
    c_int nKKT, nnzKKTmax; // Size, number of nonzeros and max number of nonzeros in KKT matrix
    csc * KKT_trip, * KKT;   // KKT matrix in triplet format and CSC format
    c_int ptr, i, j;       // Counters for elements (i,j) and index pointer
    c_int zKKT = 0;       // Counter for total number of elements in P and in KKT
    c_int * KKT_TtoC;  // Pointer to vector mapping from KKT in triplet form to CSC

    // Get matrix dimensions
    nKKT = P->m + A->m;

    // Get maximum number of nonzero elements (only upper triangular part)
    nnzKKTmax = P->p[P->n] +         // Number of elements in P
                P->m +               // Number of elements in scalar1 * I
                A->p[A->n] +         // Number of nonzeros in A
                A->m;                // Number of elements in - scalar2 * I

    // Preallocate KKT matrix in triplet format
    KKT_trip = csc_spalloc(nKKT, nKKT, nnzKKTmax, 1, 1);

    if (!KKT_trip) return OSQP_NULL;  // Failed to preallocate matrix

    // Allocate Triplet matrices
    // P + scalar1 I
    for (j = 0; j < P->n; j++){ // cycle over columns
        // No elements in column j => add diagonal element scalar1
        if (P->p[j] == P->p[j+1]){
            KKT_trip->i[zKKT] = j;
            KKT_trip->p[zKKT] = j;
            KKT_trip->x[zKKT] = scalar1;
            zKKT++;
        }
        for (ptr = P->p[j]; ptr < P->p[j + 1]; ptr++) { // cycle over rows
            // Get current row
            i = P->i[ptr];

            // Add element of P
            KKT_trip->i[zKKT] = i;
            KKT_trip->p[zKKT] = j;
            KKT_trip->x[zKKT] = P->x[ptr];
            if (PtoKKT != OSQP_NULL) PtoKKT[ptr] = zKKT;  // Update index from P to KKTtrip
            if (i == j){ // P has a diagonal element, add scalar1
                KKT_trip->x[zKKT] += scalar1;
            }
            zKKT++;

            // Add diagonal scalar1 in case
            if ((i < j) && // Diagonal element not reached
                (ptr + 1 == P->p[j+1])){ // last element of column j

                // Add diagonal element scalar1
                KKT_trip->i[zKKT] = j;
                KKT_trip->p[zKKT] = j;
                KKT_trip->x[zKKT] = scalar1;
                zKKT++;
            }
        }
    }


    // A' at top right
    for (j = 0; j < A->n; j++) {  // Cycle over columns of A
        for (ptr = A->p[j]; ptr < A->p[j + 1]; ptr++) {
            KKT_trip->p[zKKT] = P->m + A->i[ptr];  // Assign column index from row index of A
            KKT_trip->i[zKKT] = j; // Assign row index from column index of A
            KKT_trip->x[zKKT] = A->x[ptr];  // Assign A value element
            if (AtoKKT != OSQP_NULL) AtoKKT[ptr] = zKKT;  // Update index from A to KKTtrip
            zKKT++;
        }
    }

    // - scalar2 * I at bottom right
    for (j = 0; j < A->m; j++) {
        KKT_trip->i[zKKT] = j + P->n;
        KKT_trip->p[zKKT] = j + P->n;
        KKT_trip->x[zKKT] = - scalar2;
        zKKT++;
    }


    // Allocate number of nonzeros
    KKT_trip->nz = zKKT;

    // Convert triplet matrix to csc format
    if ((PtoKKT == OSQP_NULL) && (AtoKKT == OSQP_NULL)){
        // If no index vectors passed, do not store KKT mapping from Trip to CSC
        KKT = triplet_to_csc(KKT_trip, OSQP_NULL);
    }
    else{

        // Allocate vector of indeces from triplet to csc
        KKT_TtoC = c_malloc((zKKT) * sizeof(c_int));
        if(!KKT_TtoC) return OSQP_NULL; // Error in allocating KKT_TtoC vector

        // Store KKT mapping from Trip to CSC
        KKT = triplet_to_csc(KKT_trip, KKT_TtoC);

        // Update vectors of indeces from P and A to KKT (now in CSC format)
        for (i = 0; i < P->p[P->n]; i++){
            PtoKKT[i] = KKT_TtoC[PtoKKT[i]];
        }
        for (i = 0; i < A->p[A->n]; i++){
            AtoKKT[i] = KKT_TtoC[AtoKKT[i]];
        }

        // Free mapping
        c_free(KKT_TtoC);

    }

    // Clean matrix in triplet format and return result
    csc_spfree(KKT_trip);

    return KKT;

}
