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

extern CUDA_Handle_t CUDA_handle;


/** Initialize CUDA library handle.
 * @return	CUDA library handle, or NULL if failure.
 */
CUDA_Handle_t* CUDA_init_libs(void);


/** Free CUDA library handle.
 * @param CUDA_handle	CUDA library handle.
 */
void CUDA_free_libs(CUDA_Handle_t *CUDA_handle);


# ifdef __cplusplus
}
# endif

#endif
