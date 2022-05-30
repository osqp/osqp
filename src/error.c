#include "glob_opts.h"
#include "error.h"

const char *OSQP_ERROR_MESSAGE[] = {
  "Problem data validation.",
  "Solver settings validation.",
  "Linear system solver not available.\nTried to obtain it from shared library.",
  "Linear system solver initialization.",
  "KKT matrix factorization.\nThe problem seems to be non-convex.",
  "Memory allocation.",
  "Solver workspace not initialized.",
  "Algebra libraries not loaded.",
  "Unable to open file for writing.",
  "Invalid defines for codegen",

  /* This must always be the last item in the list */
  "Unknown error code."
};


c_int _osqp_error(enum osqp_error_type error_code,
		 const char * function_name) {
  c_print("ERROR in %s: %s\n", function_name, OSQP_ERROR_MESSAGE[error_code-1]);
  return (c_int)error_code;
}

