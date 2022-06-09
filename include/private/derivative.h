#ifndef DERIVATIVE_H
#define DERIVATIVE_H


# include "osqp.h"
# include "types.h"

c_int adjoint_derivative(OSQPSolver *solver, c_float *dx, c_float *dy_l, c_float *dy_u, const csc *check1, const c_float *check2, c_float tol1, c_float tol2, csc* dP, c_float* dq, csc* dA, c_float* dl, c_float* du);


#endif /* ifndef DERIVATIVE_H */
