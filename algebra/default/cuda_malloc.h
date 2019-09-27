#ifndef CUDA_MALLOC_H
# define CUDA_MALLOC_H


# ifdef __cplusplus
extern "C" {
# endif /* ifdef __cplusplus */


void cuda_malloc(void** devPtr, size_t size);

void cuda_calloc(void** devPtr, size_t size);

void cuda_free(void** devPtr);


# ifdef __cplusplus
}
# endif

#endif /* ifndef CUDA_MALLOC_H */
