/* Solution polish based on assuming the active set */
#ifndef POLISH_H
#define POLISH_H

#ifdef __cplusplus
extern "C" {
#endif

#include "types.h"
#include "lin_alg.h"
#include "util.h"
#include "auxil.h"
#include "kkt.h"

/**
 * Solution polish: Solve equality constrained QP with assumed active constraints
 * @param  work Workspace
 * @return      Exitflag:  0: Factorization successfull
 *                         1: Factorization unsuccessfull
 */
c_int polish(OSQPWorkspace *work);


#ifdef __cplusplus
}
#endif

#endif
