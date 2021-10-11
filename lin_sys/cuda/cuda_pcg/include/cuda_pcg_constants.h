/**
 *  Copyright (c) 2019-2021 ETH Zurich, Automatic Control Lab,
 *  Michel Schubiger, Goran Banjac.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef CUDA_PCG_CONSTANTS_H
# define CUDA_PCG_CONSTANTS_H

#ifdef __cplusplus
extern "C" {
#endif

/* PCG parameters */
#define CUDA_PCG_MAX_ITER            (20)
#define CUDA_PCG_EPS_MIN             (1e-7)

/* Tolerance parameters */
#define CUDA_PCG_REDUCTION_FACTOR    (0.15)
#define CUDA_PCG_REDUCTION_THRESHOLD (10)

/* Polishing parameters */
#define CUDA_PCG_POLISH_ACCURACY     (1e-5)
#define CUDA_PCG_POLISH_MAX_ITER     (100)


// GB: These values should be passed from the main OSQP interface.

#ifdef __cplusplus
}
#endif

#endif /* ifndef CUDA_PCG_CONSTANTS_H */
