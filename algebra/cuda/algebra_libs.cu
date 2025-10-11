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

#include "osqp_api_constants.h"
#include "osqp_api_types.h"
#include "lin_alg.h"
#include "cuda_handler.h"
#include "cuda_pcg_interface.h"

#include "profilers.h"

 #include <stdio.h>


CUDA_Handle_t *CUDA_handle = OSQP_NULL;

OSQPInt osqp_algebra_linsys_supported(void) {
  /* Only has a PCG (indirect) solver */
  return OSQP_CAPABILITY_INDIRECT_SOLVER;
}

enum osqp_linsys_solver_type osqp_algebra_default_linsys(void) {
  /* Prefer the PCG solver (it is also the only one available) */
  return OSQP_INDIRECT_SOLVER;
}

OSQPInt osqp_algebra_init_ctx(OSQPAlgebraContext** alg_context,
                              OSQPInt device) {
  alg_context = c_malloc(sizeof(OSQPAlgebraContext));

  alg_context->device = device;
  alg_context->CUDA_handles = cuda_init_libs((int)device);
  if (!alg_context->CUDA_handle) return 1;
  return 0;
}

void osqp_algebra_free_ctx(OSQPAlgebraContext* alg_context) {
  if (!alg_context) return;

  // Free CUDA library handles
  cuda_free_libs(alg_context->CUDA_Handles);

  c_free(alg_context);
}

OSQPInt osqp_algebra_name(char* name, OSQPInt nameLen) {
  OSQPInt runtimeVersion = 0;

  cudaRuntimeGetVersion(&runtimeVersion);

  return snprintf(name, nameLen, "CUDA %d.%d",
                  runtimeVersion / 1000, (runtimeVersion % 100) / 10);
}

OSQPInt osqp_algebra_device_name(char* name, OSQPInt nameLen) {
  OSQPInt dev;
  cudaDeviceProp deviceProp;

  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&deviceProp, dev);

  return snprintf(name, nameLen, "%s (Compute capability %d.%d)", deviceProp.name, deviceProp.major, deviceProp.minor);
}

// Initialize linear system solver structure
// NB: Only the upper triangular part of P is filled
OSQPInt osqp_algebra_init_linsys_solver(LinSysSolver**      s,
                                        const OSQPMatrix*   P,
                                        const OSQPMatrix*   A,
                                        const OSQPVectorf*  rho_vec,
                                        const OSQPSettings* settings,
                                        OSQPFloat*          scaled_prim_res,
                                        OSQPFloat*          scaled_dual_res,
                                        OSQPInt             polishing) {
  OSQPInt retval = 0;

  osqp_profiler_sec_push(OSQP_PROFILER_SEC_LINSYS_INIT);

  switch (settings->linsys_solver) {
  default:
  case OSQP_INDIRECT_SOLVER:
    retval = init_linsys_solver_cudapcg((cudapcg_solver **)s, P, A, rho_vec, settings, scaled_prim_res, scaled_dual_res, polishing);
  }

  osqp_profiler_sec_pop(OSQP_PROFILER_SEC_LINSYS_INIT);
  return retval;
}
