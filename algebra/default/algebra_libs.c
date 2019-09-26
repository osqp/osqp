
#include "cuda_handler.h"

CUDA_Handle_t *CUDA_handle;

void algebra_init_libs(void) {
  CUDA_handle = cuda_init_libs();
}

void algebra_free_libs(void) {
  cuda_free_libs(CUDA_handle);
}
