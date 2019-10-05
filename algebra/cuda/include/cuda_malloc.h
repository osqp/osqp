#ifndef CUDA_MALLOC_H
# define CUDA_MALLOC_H

# ifdef __cplusplus
extern "C" {
# endif /* ifdef __cplusplus */


void cuda_malloc(void** devPtr, size_t size);

void cuda_malloc_host(void** devPtr, size_t size);

void cuda_calloc(void** devPtr, size_t size);

void cuda_free(void** devPtr);

void cuda_free_host(void** devPtr);


# ifdef __cplusplus
}
# endif /* ifdef __cplusplus */

#endif /* ifndef CUDA_MALLOC_H */
