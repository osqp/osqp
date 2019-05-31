#include "error.h"

const char *OSQP_ERROR_DESCR[] = {
  "Problem data validation error",
  "Solver settings validation error",
  "Linear system solver not available.\nTried to obtain it from shared library",
  "Linear system solver initialization error",
  "Error in KKT matrix factorization.\nThe problem seems to be non-convex",
  "Solver workspace memory allocation error",
  "Solver workspace not initialized",
  "Polishing workspace memory allocation error",
};


c_int osqp_error(enum osqp_error_type error_code) {
# ifdef PRINTING
  c_eprint(OSQP_ERROR_DESCR[error_code]);
# endif
  return (c_int)error_code;
}
