
#include "osqp_api_types.h"
#include "cuda_handler.h"

CUDA_Handle_t *CUDA_handle;

c_int algebra_init_libs(void) {
  CUDA_handle = cuda_init_libs();
  if (!CUDA_handle) return 1;
  return 0;
}

void algebra_free_libs(void) {
  cuda_free_libs(CUDA_handle);
}
