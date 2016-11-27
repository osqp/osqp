#ifndef KKT_H
#define KKT_H

#include "osqp.h"

/**
 * Form square symmetric KKT matrix of the form
 *
 * [P + scalar I,         A';
 * A             -1/scalar I]
 *
 * for the ADMM iterations (polishing == 0). Otherwise, if polishing == 1, it forms
 *
 * [P + scalar I,         A';
 * A               -scalar I]
 *
 * N.B. Only the upper triangular part is stuffed!
 *
 * @param  P         cost matrix (already just upper triangular part)
 * @param  A         linear constraint matrix
 * @param  scalar    ADMM step rho (polish == 0), or polishing delta (polish == 1)
 * @param  polish    boolean to define if matrix is defined for polishing or not
 * @return           return status flag
 */
csc * form_KKT(const csc * P, const  csc * A, c_float rho, c_int polish);


#endif
