#ifndef TEST_UTILS_H_
#define TEST_UTILS_H_

#include "osqp.h"

OSQPFloat vec_norm_inf(const OSQPFloat* v, OSQPInt l);
OSQPFloat vec_norm_inf_diff(const OSQPFloat* a, const OSQPFloat* b, OSQPInt l);
OSQPInt isLinsysSupported(enum osqp_linsys_solver_type solver);

#endif
