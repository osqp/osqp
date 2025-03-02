/**
 * @file cuda-cudss_interface.cu
 * @author OSQP Team
 * @brief Implementation of the interface to the NVIDIA cuDSS direct linear system solver.
 */

#include "cuda-cudss_interface.h"
#include "cuda_memory.h"
#include "cuda_lin_alg.h"
#include "glob_opts.h"
#include "kkt.h"
#include "csc_math.h"

#include <cuda_runtime.h>
#include <cudss.h>

/**
 * cuDSS solver structure
 */
struct cudss_solver {
    enum osqp_linsys_solver_type type;  ///< Linear system solver type

    /**
     * @name Functions
     * @{
     */
    const char* (*name)(struct cudss_solver_* self);  ///< Name of the solver

    OSQPInt n;                          ///< Dimension of the linear system
    OSQPInt m;                          ///< Number of rows in matrix A
    
    OSQPCscMatrix* KKT;                 ///< KKT matrix (on host)
    OSQPFloat*     rho_vec;             ///< Diagonal rho vector (on host)
    OSQPFloat      sigma;               ///< Regularization parameter
    OSQPInt*       PtoKKT;              ///< Indices of elements from P to KKT matrix
    OSQPInt*       AtoKKT;              ///< Indices of elements from A to KKT matrix
    OSQPInt*       rhotoKKT;            ///< Indices of rho places in KKT matrix
    
    // cuDSS specific members
    cudssHandle_t  handle;              ///< cuDSS handle
    cudssConfig_t  config;              ///< Solver configuration
    cudssData_t    data;                ///< Solver data
    cudssMatrix_t  A_mat;               ///< Matrix object for KKT matrix
    cudssMatrix_t  x_mat;               ///< Matrix object for solution vector
    cudssMatrix_t  b_mat;               ///< Matrix object for right-hand side vector
    cudssStatus_t  status;              ///< Status of the last operation
    
    // Device memory
    OSQPInt*     d_KKT_p;               ///< KKT column pointers on device
    OSQPInt*     d_KKT_i;               ///< KKT row indices on device
    OSQPFloat*   d_KKT_x;               ///< KKT values on device
    OSQPFloat*   d_b;                   ///< Right-hand side on device
    OSQPFloat*   d_x;                   ///< Solution vector on device
    
    OSQPInt polish;                     ///< Flag indicating if we are in polish phase
    OSQPInt nthreads;                   ///< Number of threads used (1 for cuDSS)
};

OSQPInt init_linsys_solver_cudss(
    cudss_solver**       s,
    const OSQPCscMatrix* P,
    const OSQPCscMatrix* A,
    const OSQPFloat*     rho_vec,
    OSQPInt              polish,
    const OSQPSettings*  settings) {
    
    OSQPInt n, m, nnzKKT;
    cudss_solver* solver;
    
    // Dimensions
    n = P->n;
    m = A->m;
    
    // Allocate solver structure
    solver = (cudss_solver*)c_calloc(1, sizeof(cudss_solver));
    if (!solver) return OSQP_MEMORY_ALLOC_ERROR;
    
    // Set type
    solver->type = OSQP_DIRECT_SOLVER;
    
    // Set dimensions
    solver->n = n;
    solver->m = m;
    
    // Set polish flag
    solver->polish = polish;
    
    // Regularization parameter
    solver->sigma = settings->sigma;
    
    // Allocate KKT matrix
    if (polish) {
        // Allocate empty KKT matrix for polish (no regularization)
        solver->KKT = form_KKT(P, A, 1, 0.0, OSQP_NULL, 0.0, &(solver->PtoKKT), &(solver->AtoKKT), &(solver->rhotoKKT));
    } else {
        // Allocate empty KKT matrix (with regularization)
        solver->KKT = form_KKT(P, A, 1, settings->sigma, rho_vec, settings->rho, &(solver->PtoKKT), &(solver->AtoKKT), &(solver->rhotoKKT));
    }
    
    if (!(solver->KKT)) {
        free_linsys_solver_cudss(solver);
        return OSQP_MEMORY_ALLOC_ERROR;
    }
    
    // Save pointer to rho vector
    if (!polish) {
        solver->rho_vec = (OSQPFloat*)c_malloc(m * sizeof(OSQPFloat));
        if (!(solver->rho_vec)) {
            free_linsys_solver_cudss(solver);
            return OSQP_MEMORY_ALLOC_ERROR;
        }
        prea_vec_copy(rho_vec, solver->rho_vec, m);
    } else {
        solver->rho_vec = OSQP_NULL;
    }
    
    // Get number of nonzeros in KKT
    nnzKKT = solver->KKT->p[n+m];
    
    // Initialize cuDSS
    solver->status = cudssCreate(&(solver->handle));
    if (solver->status != CUDSS_STATUS_SUCCESS) {
        free_linsys_solver_cudss(solver);
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }
    
    // Create solver configuration
    solver->status = cudssConfigCreate(&(solver->config));
    if (solver->status != CUDSS_STATUS_SUCCESS) {
        free_linsys_solver_cudss(solver);
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }
    
    // Create solver data
    solver->status = cudssDataCreate(solver->handle, &(solver->data));
    if (solver->status != CUDSS_STATUS_SUCCESS) {
        free_linsys_solver_cudss(solver);
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }
    
    // Allocate device memory for KKT matrix
    cuda_malloc((void**)&(solver->d_KKT_p), (n+m+1) * sizeof(OSQPInt));
    cuda_malloc((void**)&(solver->d_KKT_i), nnzKKT * sizeof(OSQPInt));
    cuda_malloc((void**)&(solver->d_KKT_x), nnzKKT * sizeof(OSQPFloat));
    cuda_malloc((void**)&(solver->d_b), (n+m) * sizeof(OSQPFloat));
    cuda_malloc((void**)&(solver->d_x), (n+m) * sizeof(OSQPFloat));
    
    // Copy KKT matrix to device
    cuda_vec_int_copy_h2d(solver->d_KKT_p, solver->KKT->p, n+m+1);
    cuda_vec_int_copy_h2d(solver->d_KKT_i, solver->KKT->i, nnzKKT);
    cuda_vec_copy_h2d(solver->d_KKT_x, solver->KKT->x, nnzKKT);
    
    // Create matrix objects
    cudssMatrixType_t mtype = CUDSS_MTYPE_SPD;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t base = CUDSS_BASE_ZERO;
    
    // Create matrix object for the sparse KKT matrix
    solver->status = cudssMatrixCreateCsr(
        &(solver->A_mat),           // matrix object
        n+m,                        // number of rows
        n+m,                        // number of columns
        nnzKKT,                     // number of non-zeros
        solver->d_KKT_p,            // row offsets
        NULL,                       // row indices (not needed for CSR)
        solver->d_KKT_i,            // column indices
        solver->d_KKT_x,            // values
        CUDA_R_32I,                 // index data type
        CUDA_R_64F,                 // value data type
        mtype,                      // matrix type
        mview,                      // matrix view
        base                        // index base
    );
    
    if (solver->status != CUDSS_STATUS_SUCCESS) {
        free_linsys_solver_cudss(solver);
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }
    
    // Create matrix objects for the right-hand side and solution vectors
    solver->status = cudssMatrixCreateDn(
        &(solver->b_mat),           // matrix object
        n+m,                        // number of rows
        1,                          // number of columns
        n+m,                        // leading dimension
        solver->d_b,                // values
        CUDA_R_64F,                 // data type
        CUDSS_LAYOUT_COL_MAJOR      // layout
    );
    
    if (solver->status != CUDSS_STATUS_SUCCESS) {
        free_linsys_solver_cudss(solver);
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }
    
    solver->status = cudssMatrixCreateDn(
        &(solver->x_mat),           // matrix object
        n+m,                        // number of rows
        1,                          // number of columns
        n+m,                        // leading dimension
        solver->d_x,                // values
        CUDA_R_64F,                 // data type
        CUDSS_LAYOUT_COL_MAJOR      // layout
    );
    
    if (solver->status != CUDSS_STATUS_SUCCESS) {
        free_linsys_solver_cudss(solver);
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }
    
    // Symbolic factorization
    solver->status = cudssExecute(
        solver->handle,
        CUDSS_PHASE_ANALYSIS,
        solver->config,
        solver->data,
        solver->A_mat,
        solver->x_mat,
        solver->b_mat
    );
    
    if (solver->status != CUDSS_STATUS_SUCCESS) {
        free_linsys_solver_cudss(solver);
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }
    
    // Numerical factorization
    solver->status = cudssExecute(
        solver->handle,
        CUDSS_PHASE_FACTORIZATION,
        solver->config,
        solver->data,
        solver->A_mat,
        solver->x_mat,
        solver->b_mat
    );
    
    if (solver->status != CUDSS_STATUS_SUCCESS) {
        free_linsys_solver_cudss(solver);
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }
    
    // Set function pointers
    solver->name = &name_cudss;
    
    // Assign output
    *s = solver;
    
    return 0;
}

void solve_linsys_cudss(
    cudss_solver* s,
    OSQPFloat*     b) {
    
    OSQPInt n = s->n;
    OSQPInt m = s->m;
    
    // Copy right-hand side to device
    cuda_vec_copy_h2d(s->d_b, b, n+m);
    
    // Solve the system
    s->status = cudssExecute(
        s->handle,
        CUDSS_PHASE_SOLVE,
        s->config,
        s->data,
        s->A_mat,
        s->x_mat,
        s->b_mat
    );
    
    // Copy solution back to host
    cuda_vec_copy_d2h(b, s->d_x, n+m);
}

OSQPInt update_linsys_solver_matrices_cudss(
    cudss_solver*       s,
    const OSQPCscMatrix* P,
    const OSQPCscMatrix* A,
    const OSQPFloat*     rho_vec,
    OSQPInt              polish) {
    
    OSQPInt n = s->n;
    OSQPInt m = s->m;
    OSQPInt nnzKKT;
    
    // Update KKT matrix with new P, A and rho
    if (polish) {
        // Update KKT matrix in polish (no regularization)
        update_KKT_P(s->KKT, P, OSQP_NULL, 0, s->PtoKKT, 0.0, 1);
        update_KKT_A(s->KKT, A, OSQP_NULL, 0, s->AtoKKT);
    } else {
        // Update KKT matrix (with regularization)
        update_KKT_P(s->KKT, P, OSQP_NULL, 0, s->PtoKKT, s->sigma, 1);
        update_KKT_A(s->KKT, A, OSQP_NULL, 0, s->AtoKKT);
        
        // Update rho vector
        if (rho_vec) {
            prea_vec_copy(rho_vec, s->rho_vec, m);
            update_KKT_param2(s->KKT, s->rho_vec, s->sigma, s->rhotoKKT, m);
        }
    }
    
    // Get number of nonzeros in KKT
    nnzKKT = s->KKT->p[n+m];
    
    // Copy updated KKT values to device
    cuda_vec_copy_h2d(s->d_KKT_x, s->KKT->x, nnzKKT);
    
    // Refactorize the matrix
    s->status = cudssExecute(
        s->handle,
        CUDSS_PHASE_FACTORIZATION,
        s->config,
        s->data,
        s->A_mat,
        s->x_mat,
        s->b_mat
    );
    
    if (s->status != CUDSS_STATUS_SUCCESS) {
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }
    
    return 0;
}

void update_linsys_solver_rho_vec_cudss(
    cudss_solver*   s,
    const OSQPFloat* rho_vec,
    OSQPInt          polish) {
    
    OSQPInt n = s->n;
    OSQPInt m = s->m;
    OSQPInt nnzKKT;
    
    // Update KKT matrix with new rho
    if (!polish) {
        // Update rho vector
        prea_vec_copy(rho_vec, s->rho_vec, m);
        update_KKT_param2(s->KKT, s->rho_vec, s->sigma, s->rhotoKKT, m);
        
        // Get number of nonzeros in KKT
        nnzKKT = s->KKT->p[n+m];
        
        // Copy updated KKT values to device
        cuda_vec_copy_h2d(s->d_KKT_x, s->KKT->x, nnzKKT);
        
        // Refactorize the matrix
        s->status = cudssExecute(
            s->handle,
            CUDSS_PHASE_FACTORIZATION,
            s->config,
            s->data,
            s->A_mat,
            s->x_mat,
            s->b_mat
        );
    }
}

void free_linsys_solver_cudss(cudss_solver* s) {
    if (s) {
        // Free cuDSS resources
        if (s->x_mat) cudssMatrixDestroy(s->x_mat);
        if (s->b_mat) cudssMatrixDestroy(s->b_mat);
        if (s->A_mat) cudssMatrixDestroy(s->A_mat);
        if (s->data) cudssDataDestroy(s->handle, s->data);
        if (s->config) cudssConfigDestroy(s->config);
        if (s->handle) cudssDestroy(s->handle);
        
        // Free device memory
        if (s->d_KKT_p) cuda_free((void**)&(s->d_KKT_p));
        if (s->d_KKT_i) cuda_free((void**)&(s->d_KKT_i));
        if (s->d_KKT_x) cuda_free((void**)&(s->d_KKT_x));
        if (s->d_b) cuda_free((void**)&(s->d_b));
        if (s->d_x) cuda_free((void**)&(s->d_x));
        
        // Free host memory
        if (s->KKT) csc_spfree(s->KKT);
        if (s->PtoKKT) c_free(s->PtoKKT);
        if (s->AtoKKT) c_free(s->AtoKKT);
        if (s->rhotoKKT) c_free(s->rhotoKKT);
        if (s->rho_vec) c_free(s->rho_vec);
        
        // Free solver structure
        c_free(s);
    }
}

const char* name_cudss(cudss_solver* s) {
    return "NVIDIA cuDSS (direct)";
}
