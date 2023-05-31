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

#ifndef CUDA_CONFIGURE_H
# define CUDA_CONFIGURE_H


#ifdef OSQP_USE_FLOAT
#define CUDA_FLOAT CUDA_R_32F
#else
#define CUDA_FLOAT CUDA_R_64F
#endif

#define ELEMENTS_PER_THREAD (8)
#define THREADS_PER_BLOCK   (1024)
#define NUMBER_OF_BLOCKS    (2)
#define NUMBER_OF_SM        (68)


#endif /* ifndef CUDA_CONFIGURE_H */
