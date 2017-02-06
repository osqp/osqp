/* Solution polishing based on assuming the active set */
#ifndef POLISH_H
#define POLISH_H

#ifdef __cplusplus
extern "C" {
#endif

#include "osqp.h"
#include "lin_sys.h"
#include "auxil.h"
#include "kkt.h"

// Solution polishing: Solve equality constrained QP with assumed active constr.
c_int polish(Work *work);


#ifdef __cplusplus
}
#endif

#endif
