#ifndef DERIVATIVE_H
#define DERIVATIVE_H


#include "osqp.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif

#endif /* ifndef DERIVATIVE_H */
