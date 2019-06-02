#include "error.h"

const char *OSQP_ERROR_DESCR[] = {
  "Problem data validation.",
  "Solver settings validation.",
  "Linear system solver not available.\nTried to obtain it from shared library.",
  "Linear system solver initialization.",
  "KKT matrix factorization.\nThe problem seems to be non-convex.",
  "Solver workspace memory allocation.",
  "Solver workspace not initialized.",
  "Polishing workspace memory allocation.",
};


c_int osqp_error(enum osqp_error_type error_code) {
# ifdef PRINTING
  c_eprint(OSQP_ERROR_DESCR[error_code]);
# endif
  return (c_int)error_code;
}
