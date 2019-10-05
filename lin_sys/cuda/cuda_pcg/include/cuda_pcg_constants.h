#ifndef CUDA_PCG_CONSTANTS_H
# define CUDA_PCG_CONSTANTS_H

#ifdef __cplusplus
extern "C" {
#endif

/* PCG parameters */
#define CUDA_PCG_PRECONDITION        (1)
#define CUDA_PCG_MAX_ITER            (20)
#define CUDA_PCG_WARM_START          (1)
#define CUDA_PCG_NORM                (0)     /* 0: inf;  2: Euclidean */
#define CUDA_PCG_EPS_MIN             (1e-7)

/* Tolerance parameters */
#define CUDA_PCG_START_TOL           (50)
#define CUDA_PCG_DECAY_RATE          (2.75)
#define CUDA_PCG_REDUCTION_FACTOR    (0.17)
#define CUDA_PCG_REDUCTION_THRESHOLD (10)

/* Polishing parameters */
#define CUDA_PCG_POLISH_ACCURACY     (1e-5)
#define CUDA_PCG_POLISH_MAX_ITER     (1e3)


// GB: These values should be passed from the main OSQP interface.


/****************************
 * PCG Tolerance Strategies *
 ****************************/
enum pcg_eps_strategy { SCS_STRATEGY, RESIDUAL_STRATEGY };


#ifdef __cplusplus
}
#endif

#endif /* #ifndef CUDA_PCG_CONSTANTS_H */
