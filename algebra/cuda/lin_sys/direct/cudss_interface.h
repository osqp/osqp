#ifndef CUDSS_INTERFACE_H
#define CUDSS_INTERFACE_H

#include <cudss.h>
#include <cuda_runtime.h>
#include "osqp.h"
#include "types.h"                /* OSQPMatrix and OSQPVector[fi] types */
#include "algebra_types.h"        /* csr type */

/**
 * cuDSS solver structure
 */
typedef struct cudss_solver_ {

  enum osqp_linsys_solver_type type;

  /**
   * @name Functions
   * @{
   */
  const char* (*name)(struct cudss_solver_* self);

  OSQPInt (*solve)(struct cudss_solver_* self,
                          OSQPVectorf*     b,
                          OSQPInt          admm_iter);

  void (*update_settings)(struct cudss_solver_* self,
                          const  OSQPSettings*    settings);

  void (*warm_start)(struct cudss_solver_* self,
                     const  OSQPVectorf*     x);

  OSQPInt (*adjoint_derivative)(struct cudss_solver_* self);

  void (*free)(struct cudss_solver_* self);

  OSQPInt (*update_matrices)(struct cudss_solver_* self,
                             const  OSQPMatrix*      P,
                             const  OSQPInt*         Px_new_idx,
                                    OSQPInt          P_new_n,
                             const  OSQPMatrix*      A,
                             const  OSQPInt*         Ax_new_idx,
                                    OSQPInt          A_new_n);

  OSQPInt (*update_rho_vec)(struct cudss_solver_* self,
                            const  OSQPVectorf*     rho_vec,
                                   OSQPFloat        rho_sc);

  /* threads count */
  OSQPInt nthreads;

  /* Dimensions */
  OSQPInt n;                  ///<  number of QP variables
  OSQPInt m;                  ///<  number of QP constraints
  OSQPInt nKKT;               ///<  dimension of the KKT system

  /* States */
  OSQPInt polishing;
  OSQPInt factorized;         ///<  flag indicating if factorization is done

  /* cuDSS handle and configuration */
  cudssHandle_t cudss_handle;
  cudssConfig_t cudss_config;
  cudssData_t   cudss_data;

  /* Matrix descriptors */
  cudssMatrix_t matA;         ///<  KKT matrix descriptor
  cudssMatrix_t vecX;         ///<  solution vector descriptor  
  cudssMatrix_t vecB;         ///<  RHS vector descriptor

  /* KKT matrix storage */
  OSQPCscMatrix* KKT;         ///<  KKT matrix (CSR data in OSQPCscMatrix structure)
  OSQPInt*   d_KKT_i;         ///<  KKT column indices (device memory)
  OSQPInt*   d_KKT_p;         ///<  KKT row pointers (device memory)
  OSQPFloat* d_KKT_x;         ///<  KKT matrix values (device memory)

  /* Solution vectors */
  OSQPFloat* d_sol;           ///<  solution vector (device memory)
  OSQPFloat* d_rhs;           ///<  right-hand side vector (device memory)

  /* Parameter vectors */
  OSQPFloat* rho_inv_vec;     ///<  parameter vector
  OSQPFloat  sigma;           ///<  scalar parameter
  OSQPFloat  rho_inv;         ///<  scalar parameter (used if rho_inv_vec == NULL)

  /* Matrix update indices */
  OSQPInt* PtoKKT;            ///<  Index of elements from P to KKT matrix
  OSQPInt* AtoKKT;            ///<  Index of elements from A to KKT matrix
  OSQPInt* rhotoKKT;          ///<  Index of rho places in KKT matrix

} cudss_solver;

#ifdef __cplusplus
extern "C" {
#endif

OSQPInt init_linsys_solver_cudss(cudss_solver**      sp,
                                 const OSQPMatrix*   P,
                                 const OSQPMatrix*   A,
                                 const OSQPVectorf*  rho_vec,
                                 const OSQPSettings* settings,
                                 OSQPFloat*          scaled_prim_res,
                                 OSQPFloat*          scaled_dual_res,
                                 OSQPInt             polishing);

#ifdef __cplusplus
}
#endif

/**
 * Get the user-friendly name of the cuDSS solver.
 * @return The user-friendly name
 */
const char* name_cudss(cudss_solver* s);

OSQPInt solve_linsys_cudss(cudss_solver* s,
                           OSQPVectorf*  b,
                           OSQPInt       admm_iter);

void update_settings_linsys_solver_cudss(cudss_solver*       s,
                                         const OSQPSettings* settings);

void warm_start_linsys_solver_cudss(cudss_solver*      s,
                                    const OSQPVectorf* x);

void free_linsys_solver_cudss(cudss_solver* s);

OSQPInt update_linsys_solver_matrices_cudss(cudss_solver*     s,
                                            const OSQPMatrix* P,
                                            const OSQPInt*    Px_new_idx,
                                            OSQPInt           P_new_n,
                                            const OSQPMatrix* A,
                                            const OSQPInt*    Ax_new_idx,
                                            OSQPInt           A_new_n);

OSQPInt update_linsys_solver_rho_vec_cudss(cudss_solver*      s,
                                           const OSQPVectorf* rho_vec,
                                           OSQPFloat          rho_sc);

OSQPInt adjoint_derivative_linsys_solver_cudss(cudss_solver* s);

#endif /* ifndef CUDSS_INTERFACE_H */