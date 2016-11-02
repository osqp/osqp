#include "private.h"


// TODO: Add functions for defining factorizing and solving linear systems with direct methods

// formKKT, factorize... (see scs)


/* Form square symmetric KKT matrix of the form
   [Q + rhoI,   A';
    A  -1./rhoI]

   Arguments
   ---------
   Q : cost matrix (upper triangular part)
   A: matrix of linear inequalities
   rho: ADMM step
   N.B. Only the upper triangular part is stuffed!
 */
// csc * formKKT(csc * Q, csc * A, c_float rho){
//     c_int nKKT, nnzKKT; // Size and number of nonzeros in KKT matrix
//     csc *KKT_trip, *KKT;  // KKT matrix in triplet format and CSC format
//
//     // Get matrix dimensions
//     nKKT = Q->n; // TODO:Fix!
//
//     // Get number of nonzero elements
//     nnzKKT = Q->n;                  // Size of Q (upper triangular) TODO: Fix size with + rhoI!!!
//             //  Go on from here
//
//     // Preallocate memory KKT matrix
//     // KKT_trip =
//
//
//     // 1) Define matrix in triplet format (easier to stack matrices)
//
//     // 2) Convert triplet matrix to csc format
//
//     // 3) Clean matrix in triplet format
//
//
//     return KKT;
//
// }
