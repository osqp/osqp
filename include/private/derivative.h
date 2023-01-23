#ifndef DERIVATIVE_H
#define DERIVATIVE_H


# include "osqp.h"
# include "types.h"

OSQPInt adjoint_derivative_get_mat(OSQPSolver *solver,
                                   OSQPCscMatrix* dP,
                                   OSQPCscMatrix* dA);

OSQPInt adjoint_derivative_get_vec(OSQPSolver *solver,
                                   OSQPFloat*     dq,
                                   OSQPFloat*     dl,
                                   OSQPFloat*     du);

OSQPInt adjoint_derivative_compute(OSQPSolver *solver,
                                   OSQPFloat*     dx,
                                   OSQPFloat*     dy_l,
                                   OSQPFloat*     dy_u);

#endif /* ifndef DERIVATIVE_H */
