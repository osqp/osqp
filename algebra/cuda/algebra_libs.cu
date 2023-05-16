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

OSQPInt osqp_algebra_init_libs(OSQPInt device) {
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

  switch (settings->linsys_solver) {
  default:
  case OSQP_INDIRECT_SOLVER:
    return init_linsys_solver_cudapcg((cudapcg_solver **)s, P, A, rho_vec, settings, scaled_prim_res, scaled_dual_res, polishing);
  }
}
