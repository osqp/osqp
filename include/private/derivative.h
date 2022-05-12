#ifndef DERIVATIVE_H
#define DERIVATIVE_H


# include "osqp.h"
# include "types.h"

c_int adjoint_derivative(OSQPSolver *solver, const csc *check);


#endif /* ifndef DERIVATIVE_H */
