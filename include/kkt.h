/* KKT linear system definition and solution */

#include "cs.h"

/* Form KKT matrix of the form
[Q + rhoI,   F',  G';
 F      G,  -1./rhoI]
N.B. Only the upper triangular part is stuffed!
*/
csc * formKKT(csc * Q, csc * F, csc *G, c_float rho);
