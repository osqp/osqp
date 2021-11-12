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

#include "osqp_api_types.h"
#include "cuda_handler.h"

CUDA_Handle_t *CUDA_handle = OSQP_NULL;

c_int osqp_algebra_init_libs(c_int device) {
  /* This is to prevent a memory leak when multiple OSQP objects are created */
  if (CUDA_handle) return 0;

  CUDA_handle = cuda_init_libs((int)device);
  if (!CUDA_handle) return 1;
  return 0;
}

void osqp_algebra_free_libs(void) {
  /* This is to prevent a double free error when multiple OSQP objects are created */
  if (!CUDA_handle) return;

  cuda_free_libs(CUDA_handle);
  CUDA_handle = OSQP_NULL;
}
