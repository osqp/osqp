#ifndef CUDA_CONFIGURE_H
# define CUDA_CONFIGURE_H

#ifdef __cplusplus
extern "C" {
#endif


#define ELEMENTS_PER_THREAD (8)
#define THREADS_PER_BLOCK   (1024)
#define NUMBER_OF_BLOCKS    (2)
#define NUMBER_OF_SM        (68)


#ifdef __cplusplus
}
#endif

#endif /* ifndef CUDA_CONFIGURE_H */
