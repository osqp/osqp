#ifndef DERIVATIVE_H
#define DERIVATIVE_H


# include "osqp.h"
# include "types.h"

OSQPInt adjoint_derivative(OSQPSolver*    solver,
                           OSQPFloat*     dx,
                           OSQPFloat*     dy_l,
                           OSQPFloat*     dy_u,
                           OSQPCscMatrix* dP,
                           OSQPFloat*     dq,
                           OSQPCscMatrix* dA,
                           OSQPFloat*     dl,
                           OSQPFloat*     du);


#endif /* ifndef DERIVATIVE_H */
