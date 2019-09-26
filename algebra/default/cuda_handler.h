#ifndef CUDA_HANDLER_H
# define CUDA_HANDLER_H

#include <cusparse_v2.h>
#include <cublas_v2.h>

# ifdef __cplusplus
extern "C" {
# endif

typedef struct {
  cublasHandle_t    cublasHandle;
  cusparseHandle_t  cusparseHandle;
  int              *d_index;
} CUDA_Handle_t;


/* Initialize CUDA library handle. */
void CUDA_init_libs(void);


/* Free CUDA library handle. */
void CUDA_free_libs(void);


# ifdef __cplusplus
}
# endif

#endif /* ifndef CUDA_HANDLER_H */
