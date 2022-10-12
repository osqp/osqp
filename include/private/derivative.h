#ifndef DERIVATIVE_H
#define DERIVATIVE_H


# include "osqp.h"
# include "types.h"

OSQPInt adjoint_derivative(OSQPSolver*    solver,
                           OSQPFloat*     dx,
                           OSQPFloat*     dy_l,
                           OSQPFloat*     dy_u);

OSQPInt adjoint_derivative_get_mat(OSQPSolver *solver,
                                   OSQPCscMatrix* dP,
                                   OSQPCscMatrix* dA);

OSQPInt adjoint_derivative_get_vec(OSQPSolver *solver,
                                   OSQPFloat*     dq,
                                   OSQPFloat*     dl,
                                   OSQPFloat*     du);

OSQPInt adjoint_derivative_compute(OSQPSolver *solver,
                                   const OSQPMatrix*   P,
                                   const OSQPMatrix*   G,
                                   const OSQPMatrix*   A_eq,
                                   OSQPMatrix*         GDiagLambda,
                                   OSQPVectorf*        slacks);

#endif /* ifndef DERIVATIVE_H */
