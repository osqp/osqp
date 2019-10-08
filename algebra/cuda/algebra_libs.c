
#include "osqp_api_types.h"
#include "cuda_handler.h"

CUDA_Handle_t *CUDA_handle = NULL;

c_int osqp_algebra_init_libs(void) {
  /* This is to prevent a memory leak when multiple OSQP objects are created */
  if (CUDA_handle) return 0;

  CUDA_handle = cuda_init_libs();
  if (!CUDA_handle) return 1;
  return 0;
}

void osqp_algebra_free_libs(void) {
  /* This is to prevent a double free error when multiple OSQP objects are created */
  if (!CUDA_handle) return;

  cuda_free_libs(CUDA_handle);
  CUDA_handle = NULL;
}
