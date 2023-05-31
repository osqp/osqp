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

#ifndef CUDA_HANDLER_H
# define CUDA_HANDLER_H

#include <cusparse.h>
#include <cublas_v2.h>


typedef struct {
  cublasHandle_t    cublasHandle;
  cusparseHandle_t  cusparseHandle;
  int              *d_index;
} CUDA_Handle_t;


/** Initialize CUDA library handle.
 * @param  device CUDA device ID
 * @return        CUDA library handle, or NULL if failure.
 */
CUDA_Handle_t* cuda_init_libs(int device);


/** Free CUDA library handle.
 * @param CUDA_handle	CUDA library handle.
 */
void cuda_free_libs(CUDA_Handle_t *CUDA_handle);


#endif /* ifndef CUDA_HANDLER_H */
