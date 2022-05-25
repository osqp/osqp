#ifndef DERIVATIVE_H
#define DERIVATIVE_H


# include "osqp.h"
# include "types.h"

c_int adjoint_derivative(OSQPSolver *solver, c_float *dx, c_float *dy_l, c_float *dy_u, const csc *check1, const c_float *check2);


#endif /* ifndef DERIVATIVE_H */
