#include "cudss_interface.h"
#include "algebra_types.h"
#include "printing.h"
#include "util.h"
#include "profilers.h"
#include "cuda_memory.h"
#include "cuda_handler.h"
#include "cuda_csr.h"

#if OSQP_EMBEDDED_MODE != 1
#include "kkt.h"
#include "csc_utils.h"
#endif

#include <cuda_runtime.h>
#include <cudss.h>

// Error checking macros
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        c_eprint("CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return 1; \
    } \
} while(0)

#define CUDSS_CHECK(call) do { \
    cudssStatus_t status = call; \
    if (status != CUDSS_STATUS_SUCCESS) { \
        c_eprint("cuDSS error at %s:%d: status %d", __FILE__, __LINE__, status); \
        return 1; \
    } \
} while(0)

#define CUDSS_CHECK_VOID(call) do { \
    cudssStatus_t status = call; \
    if (status != CUDSS_STATUS_SUCCESS) { \
        c_eprint("cuDSS error at %s:%d: status %d", __FILE__, __LINE__, status); \
        return; \
    } \
} while(0)

void update_settings_linsys_solver_cudss(cudss_solver*       s,
                                         const OSQPSettings* settings) {
  // No settings to update for direct solver
  OSQP_UnusedVar(s);
  OSQP_UnusedVar(settings);
  return;
}

void warm_start_linsys_solver_cudss(cudss_solver*      s,
                                    const OSQPVectorf* x) {
  // Warm starting not used by direct solvers
  OSQP_UnusedVar(s);
  OSQP_UnusedVar(x);
  return;
}

OSQPInt adjoint_derivative_linsys_solver_cudss(cudss_solver* s) {
  // Adjoint derivative not implemented for direct solvers
  OSQP_UnusedVar(s);
  return 0;
}

// Free cuDSS Factorization structure
void free_linsys_solver_cudss(cudss_solver* s) {
  if (s) {

    // Destroy cuDSS objects
    if (s->matA) cudssMatrixDestroy(s->matA);
    if (s->vecX) cudssMatrixDestroy(s->vecX);
    if (s->vecB) cudssMatrixDestroy(s->vecB);
    if (s->cudss_data) cudssDataDestroy(s->cudss_handle, s->cudss_data);
    if (s->cudss_config) cudssConfigDestroy(s->cudss_config);
    if (s->cudss_handle) cudssDestroy(s->cudss_handle);

    // Free KKT matrix
    if (s->KKT) csc_spfree(s->KKT);

    // Free device memory
    if (s->d_KKT_i) cuda_free((void**)&s->d_KKT_i);
    if (s->d_KKT_p) cuda_free((void**)&s->d_KKT_p);
    if (s->d_KKT_x) cuda_free((void**)&s->d_KKT_x);
    if (s->d_sol) cuda_free((void**)&s->d_sol);
    if (s->d_rhs) cuda_free((void**)&s->d_rhs);
    if (s->rho_inv_vec) c_free(s->rho_inv_vec);

    // Matrix update indices
    if (s->PtoKKT) c_free(s->PtoKKT);
    if (s->AtoKKT) c_free(s->AtoKKT);
    if (s->rhotoKKT) c_free(s->rhotoKKT);

    c_free(s);
  }
}

const char* name_cudss(cudss_solver* s) {
  OSQP_UnusedVar(s);
  return "NVIDIA cuDSS (direct)";
}


// Initialize factorization structure
OSQPInt init_linsys_solver_cudss(cudss_solver**      sp,
                                 const OSQPMatrix*   P,
                                 const OSQPMatrix*   A,
                                 const OSQPVectorf*  rho_vec,
                                 const OSQPSettings* settings,
                                 OSQPFloat*          scaled_prim_res,
                                 OSQPFloat*          scaled_dual_res,
                                 OSQPInt             polishing) {

  OSQPInt    i;         // loop counter
  OSQPInt    nnzKKT;    // Number of nonzeros in KKT
  OSQPInt    m, n;      // Dimensions of A
  OSQPInt    n_plus_m;  // n_plus_m dimension
  OSQPFloat* rhov;      // used for direct access to rho_vec data when polishing=false

  OSQPFloat sigma = settings->sigma;

  // Allocate private structure to store KKT factorization
  cudss_solver *s = (cudss_solver*)c_calloc(1, sizeof(cudss_solver));
  *sp = s;

  // Size of KKT
  n = P->csc->n;
  m = A->csc->m;
  s->n = n;
  s->m = m;
  n_plus_m = n + m;
  s->nKKT = n_plus_m;

  // Sigma parameter
  s->sigma = sigma;

  // Polishing flag
  s->polishing = polishing;

  // Link Functions
  s->name               = &name_cudss;
  s->solve              = &solve_linsys_cudss;
  s->free               = &free_linsys_solver_cudss;
  s->warm_start         = &warm_start_linsys_solver_cudss;
  s->update_matrices    = &update_linsys_solver_matrices_cudss;
  s->update_rho_vec     = &update_linsys_solver_rho_vec_cudss;
  s->update_settings    = &update_settings_linsys_solver_cudss;
  s->adjoint_derivative = &adjoint_derivative_linsys_solver_cudss;

  // Assign type
  s->type = OSQP_DIRECT_SOLVER;

  // Allocate device memory for solution and RHS vectors
  if (cuda_malloc((void**)&s->d_sol, n_plus_m * sizeof(OSQPFloat)) ||
      cuda_malloc((void**)&s->d_rhs, n_plus_m * sizeof(OSQPFloat))) {
    free_linsys_solver_cudss(s);
    return 1;
  }

  // Parameter vector
  if (rho_vec) {
    s->rho_inv_vec = (OSQPFloat *)c_malloc(sizeof(OSQPFloat) * m);
  }

  // Form KKT matrix
  if (polishing){ // Called from polish()
    s->KKT = form_KKT(P->csc, A->csc,
                      1,  //format = 1 means CSR
                      sigma, s->rho_inv_vec, sigma,
                      OSQP_NULL, OSQP_NULL, OSQP_NULL);
  }
  else { // Called from ADMM algorithm

    // Allocate vectors of indices
    s->PtoKKT   = (OSQPInt*)c_malloc(P->csc->p[n] * sizeof(OSQPInt));
    s->AtoKKT   = (OSQPInt*)c_malloc(A->csc->p[n] * sizeof(OSQPInt));
    s->rhotoKKT = (OSQPInt*)c_malloc(m * sizeof(OSQPInt));

    // Use s->rho_inv_vec for storing param2 = rho_inv_vec
    if (rho_vec) {
        rhov = rho_vec->values;
        for (i = 0; i < m; i++){
            s->rho_inv_vec[i] = 1. / rhov[i];
        }
    }
    else {
      s->rho_inv = 1. / settings->rho;
    }

    s->KKT = form_KKT(P->csc, A->csc,
                      1,  //format = 1 means CSR
                      sigma, s->rho_inv_vec, s->rho_inv,
                      s->PtoKKT, s->AtoKKT, s->rhotoKKT);
  }

  // Check KKT formation
  if (!s->KKT) {
    free_linsys_solver_cudss(s);
    return 1;
  }

  nnzKKT = s->KKT->p[n_plus_m];

  // Copy KKT matrix to device in CSR format (already in CSR from form_KKT)
  if (cuda_malloc((void**)&s->d_KKT_p, (n_plus_m + 1) * sizeof(OSQPInt)) ||
      cuda_malloc((void**)&s->d_KKT_i, nnzKKT * sizeof(OSQPInt)) ||
      cuda_malloc((void**)&s->d_KKT_x, nnzKKT * sizeof(OSQPFloat))) {
    free_linsys_solver_cudss(s);
    return 1;
  }

  CUDA_CHECK(cudaMemcpy(s->d_KKT_p, s->KKT->p, (n_plus_m + 1) * sizeof(OSQPInt), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(s->d_KKT_i, s->KKT->i, nnzKKT * sizeof(OSQPInt), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(s->d_KKT_x, s->KKT->x, nnzKKT * sizeof(OSQPFloat), cudaMemcpyHostToDevice));

  // Initialize cuDSS
  CUDSS_CHECK(cudssCreate(&s->cudss_handle));

  // Create cuDSS configuration
  CUDSS_CHECK(cudssConfigCreate(&s->cudss_config));

  // Set matrix type (symmetric indefinite)
  CUDSS_CHECK(cudssConfigSet(s->cudss_config, CUDSS_CONFIG_MTYPE, CUDSS_MTYPE_SYM_INDEF));

  // Create cuDSS matrix descriptors
#ifdef OSQP_USE_FLOAT
  CUDSS_CHECK(cudssMatrixCreateCsr(&s->matA, n_plus_m, n_plus_m, nnzKKT,
                                   s->d_KKT_p, NULL, s->d_KKT_i, s->d_KKT_x,
                                   CUDA_R_32I, CUDA_R_32F, CUDSS_MTYPE_SYM_INDEF,
                                   CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
#else
  CUDSS_CHECK(cudssMatrixCreateCsr(&s->matA, n_plus_m, n_plus_m, nnzKKT,
                                   s->d_KKT_p, NULL, s->d_KKT_i, s->d_KKT_x,
                                   CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_SYM_INDEF,
                                   CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
#endif

#ifdef OSQP_USE_FLOAT
  CUDSS_CHECK(cudssMatrixCreateDn(&s->vecX, n_plus_m, 1, n_plus_m, s->d_sol,
                                  CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR));

  CUDSS_CHECK(cudssMatrixCreateDn(&s->vecB, n_plus_m, 1, n_plus_m, s->d_rhs,
                                  CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR));
#else
  CUDSS_CHECK(cudssMatrixCreateDn(&s->vecX, n_plus_m, 1, n_plus_m, s->d_sol,
                                  CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));

  CUDSS_CHECK(cudssMatrixCreateDn(&s->vecB, n_plus_m, 1, n_plus_m, s->d_rhs,
                                  CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
#endif

  // Create solver data structure
  CUDSS_CHECK(cudssDataCreate(s->cudss_handle, s->cudss_config, &s->cudss_data));

  // Perform symbolic analysis
  CUDSS_CHECK(cudssExecute(s->cudss_handle, CUDSS_PHASE_ANALYSIS, s->cudss_config, 
                           s->cudss_data, s->matA, s->vecX, s->vecB));

  // Perform numeric factorization
  CUDSS_CHECK(cudssExecute(s->cudss_handle, CUDSS_PHASE_FACTORIZATION, s->cudss_config,
                           s->cudss_data, s->matA, s->vecX, s->vecB));

  s->factorized = 1;

  return 0;
}

// Solve linear system and store result in b
OSQPInt solve_linsys_cudss(cudss_solver* s,
                           OSQPVectorf*  b,
                           OSQPInt       admm_iter) {

  OSQPInt n_plus_m = s->nKKT;

  // Copy RHS to device
  CUDA_CHECK(cudaMemcpy(s->d_rhs, b->values, n_plus_m * sizeof(OSQPFloat), cudaMemcpyHostToDevice));

  // Solve the system
  CUDSS_CHECK(cudssExecute(s->cudss_handle, CUDSS_PHASE_SOLVE, s->cudss_config,
                           s->cudss_data, s->matA, s->vecX, s->vecB));

  // Copy solution back to host
  CUDA_CHECK(cudaMemcpy(b->values, s->d_sol, n_plus_m * sizeof(OSQPFloat), cudaMemcpyDeviceToHost));

  return 0;
}

// Update matrices P and A
OSQPInt update_linsys_solver_matrices_cudss(cudss_solver*     s,
                                            const OSQPMatrix* P,
                                            const OSQPInt*    Px_new_idx,
                                            OSQPInt           P_new_n,
                                            const OSQPMatrix* A,
                                            const OSQPInt*    Ax_new_idx,
                                            OSQPInt           A_new_n) {

  // Update KKT matrix
  if (P_new_n > 0) {
    update_KKT_P(s->KKT, P->csc, Px_new_idx, P_new_n, s->PtoKKT, s->sigma, 1);
  }

  if (A_new_n > 0) {
    update_KKT_A(s->KKT, A->csc, Ax_new_idx, A_new_n, s->AtoKKT);
  }

  // Update device KKT matrix
  OSQPInt nnzKKT = s->KKT->p[s->nKKT];
  CUDA_CHECK(cudaMemcpy(s->d_KKT_x, s->KKT->x, nnzKKT * sizeof(OSQPFloat), cudaMemcpyHostToDevice));

  // Perform new numeric factorization
  CUDSS_CHECK(cudssExecute(s->cudss_handle, CUDSS_PHASE_FACTORIZATION, s->cudss_config,
                           s->cudss_data, s->matA, s->vecX, s->vecB));

  return 0;
}

// Update rho parameter
OSQPInt update_linsys_solver_rho_vec_cudss(cudss_solver*      s,
                                           const OSQPVectorf* rho_vec,
                                           OSQPFloat          rho_sc) {

  // Update rho_inv_vec
  if (rho_vec) {
    for (OSQPInt i = 0; i < s->m; i++) {
      s->rho_inv_vec[i] = 1. / rho_vec->values[i];
    }
    update_KKT_param2(s->KKT, s->rho_inv_vec, 0.0, s->rhotoKKT, s->m);
  } else {
    s->rho_inv = 1. / rho_sc;
    update_KKT_param2(s->KKT, OSQP_NULL, s->rho_inv, s->rhotoKKT, s->m);
  }

  // Update device KKT matrix
  OSQPInt nnzKKT = s->KKT->p[s->nKKT];
  CUDA_CHECK(cudaMemcpy(s->d_KKT_x, s->KKT->x, nnzKKT * sizeof(OSQPFloat), cudaMemcpyHostToDevice));

  // Perform new numeric factorization
  CUDSS_CHECK(cudssExecute(s->cudss_handle, CUDSS_PHASE_FACTORIZATION, s->cudss_config,
                           s->cudss_data, s->matA, s->vecX, s->vecB));

  return 0;
}