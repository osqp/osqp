#include "private.h"


// TODO: Add functions for defining factorizing and solving linear systems with direct methods

// formKKT, factorize... (see scs)


/* Form KKT matrix of the form
[Q + rhoI,   F',  G';
 F      G,  -1./rhoI]

Arguments
---------
Q : cost matrix (upper triangular part)
Ft: Transpose of linear equalities
Gt: transpose of linear inequalities
rho: ADMM step
N.B. Only the upper triangular part is stuffed!
*/
csc * formKKT(csc * Q, csc * F, csc *G, c_float rho);
