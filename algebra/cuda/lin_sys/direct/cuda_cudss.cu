#include "cuda_configure.h"
#include "cuda_cudss.h"
#include "cuda_wrapper.h"       // For BLAS functions
#include "helper_cuda.h"

#include "csr_type.h"
#include "cuda_csr.h"
#include "cuda_lin_alg.h"
#include "cuda_memory.h"

// CUDA library handles
#include "cuda_handler.h"
extern CUDA_Handle_t *CUDA_handle;

#include "kkt.h"

#include "util.h"
#include "glob_opts.h"
#include "profilers.h"

/*
 * Helper function to check and terminate on error
 */
#define checkCudssErrors(val) _checkCudssErrors((val), __FILE__, __LINE__)
#define checkCudssErrorsAndTerminate(val) _checkCudssErrorsAndTerminate((val), __FILE__, __LINE__)

OSQPInt _checkCudssErrors(cudssStatus_t status, const char *file, const int line) {
    int  errstr_len = 250;
    char errstr[250];

    switch(status) {
    case CUDSS_STATUS_SUCCESS:
        return OSQP_NO_ERROR;

    case CUDSS_STATUS_NOT_INITIALIZED:
        snprintf(errstr, errstr_len, "Operand not initialized");
        break;

    case CUDSS_STATUS_ALLOC_FAILED:
        snprintf(errstr, errstr_len, "Allocation failed");
        break;

    case CUDSS_STATUS_INVALID_VALUE:
        snprintf(errstr, errstr_len, "Invalid value or parameter");
        break;

    case CUDSS_STATUS_NOT_SUPPORTED:
        snprintf(errstr, errstr_len, "Unsupported parameter");
        break;

    case CUDSS_STATUS_EXECUTION_FAILED:
        snprintf(errstr, errstr_len, "GPU execution failed");
        break;

    case CUDSS_STATUS_INTERNAL_ERROR:
        snprintf(errstr, errstr_len, "Internal error");
        break;

    default:
        snprintf(errstr, errstr_len, "Unknown error");
        break;
    }

    fprintf(stderr, "%s(%i): cuDSS error %d: %s\n", file, line, static_cast<int>(status), errstr);
    return 1;
}

OSQPInt _checkCudssErrorsAndTerminate(cudssStatus_t status, const char *file, const int line) {
    if (_checkCudssErrors(status, file, line))
    {
        exit(1);
    }

    return OSQP_NO_ERROR;
}


/*******************************************************************************
 * CUDA kernels for updating KKT matrix values                                 *
 *******************************************************************************/
 // Update the given indices in the KKT matrix with the negative of the value in the
 // same index of newValues
__global__ void kkt_update_rho_vec(OSQPFloat*       dest,
                                   const OSQPFloat* newRho,
                                   const OSQPInt*   indMap,
                                   OSQPInt          n) {
    OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
    OSQPInt grid_size = blockDim.x * gridDim.x;

    for(OSQPInt i = idx; i < n; i += grid_size) {
      dest[indMap[i]] = -( 1. / newRho[i]);
    }
}

// Update the given indices in the KKT matrix with the negative of the new scalar value
__global__ void kkt_update_rho_sca(OSQPFloat*     dest,
                                   OSQPFloat      newRho,
                                   const OSQPInt* indMap,
                                   OSQPInt        n) {
    OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
    OSQPInt grid_size = blockDim.x * gridDim.x;

    for(OSQPInt i = idx; i < n; i += grid_size) {
      dest[indMap[i]] = -(1. / newRho);
    }
}

// Update the values of the diagonal by adding sigma to them
// Since this is in CSR notation, the diagonal values are at the beginning of the rows
__global__ void kkt_add_sigma(OSQPFloat*     csrVal,
                              const OSQPInt* csrRowInd,
                              OSQPFloat      sigma,
                              OSQPInt        n) {
    OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
    OSQPInt grid_size = blockDim.x * gridDim.x;

    for (OSQPInt i = idx; i < n; i+= grid_size) {
        csrVal[csrRowInd[i]] += sigma;
    }
}

// TODO: This kernel is probably not very performant, because it will be trying to
// write to the same piece of memory multiple times
__global__ void triu_update_kernel(const OSQPFloat* fullCsrVal,
                                   OSQPFloat*       triuCsrVal,
                                   const OSQPInt*   triuToFullMap,
                                   OSQPInt          full_nnz) {
    OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
    OSQPInt grid_size = blockDim.x * gridDim.x;

    for (OSQPInt i = idx; i < full_nnz; i+= grid_size) {
        triuCsrVal[triuToFullMap[i]] = fullCsrVal[i];
    }
}

/*******************************************************************************
 * CUDA kernels for solve phase                                                *
 *******************************************************************************/
__global__ void vec_ew_fma_kernel(OSQPFloat*       c,
                                  const OSQPFloat* a,
                                  const OSQPFloat* b,
                                  OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
#ifdef OSQP_USE_FLOAT
    c[i] = __fmaf_rn(a[i], b[i], c[i]);
#else
    c[i] = __fma_rn(a[i], b[i], c[i]);
#endif
  }
}

/*******************************************************************************
 *                              API Functions                                  *
 *******************************************************************************/

OSQPInt init_linsys_solver_cudss(      cudss_solver** sp,
                                 const OSQPMatrix*    P,
                                 const OSQPMatrix*    A,
                                 const OSQPVectorf*   rho_vec,
                                 const OSQPSettings*  settings,
                                       OSQPFloat*     scaled_prim_res,
                                       OSQPFloat*     scaled_dual_res,
                                       OSQPInt        polishing) {
    OSQPInt    n, m;
    OSQPInt    n_plus_m;  // Define n_plus_m dimension

    /* Allocate linsys solver structure */
    cudss_solver *s = (cudss_solver *)c_calloc(1, sizeof(cudss_solver));
    *sp = s;

    /* Create the handles */
    checkCudssErrorsAndTerminate(cudssCreate(&(s->lib_handle)));
    checkCudssErrorsAndTerminate(cudssConfigCreate(&(s->config_handle)));
    checkCudssErrorsAndTerminate(cudssDataCreate(s->lib_handle, &(s->data_handle)));
    checkCudaErrors(cudaStreamCreate(&s->cudss_stream));

    checkCudssErrorsAndTerminate(cudssSetStream(s->lib_handle, s->cudss_stream));

    /* Assign type and the number of threads */
    s->type = OSQP_DIRECT_SOLVER;
    s->nthreads = 1;

    /* Problem dimensions */
    n = OSQPMatrix_get_n(P);
    m = OSQPMatrix_get_m(A);
    s->n = n;
    s->m = m;
    n_plus_m = n+m;

    /* States */
    s->rho_is_vec = settings->rho_is_vec;
    s->polishing = polishing;

    /* Parameters*/
    if (polishing) {
        s->sigma = settings->delta;
        s->rho_inv = 1. / settings->delta;
    } else {
        s->sigma = settings->sigma;
        s->rho_inv = 1. / settings->rho;
    }
    s->d_rho_inv_vec = OSQP_NULL;       // Null here - will be created when actually needed

    // Symmetric full matrix with zero-based indexing
    cudssMatrixType_t     mtype = CUDSS_MTYPE_SYMMETRIC;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;
    cudssIndexBase_t      base  = CUDSS_BASE_ZERO;

    // Form and permute KKT matrix
    if (polishing) { // Called from polish()
        int nnzP = P->h_csc->p[n];

        csr* d_A = OSQP_NULL;
        csr_to_csc(&(d_A), A->S);

        // Create a temporary set of matrices to form the KKT
        OSQPCscMatrix* h_A = csc_spalloc(d_A->m, d_A->n, d_A->nnz, 1, 0);

        // Need the output mapping to scatter the scaled P matrix values into this KKT
        OSQPInt* PtoKKT = (OSQPInt*) c_malloc(nnzP * sizeof(OSQPInt));

        OSQPFloat* h_rho_inv_vec = OSQP_NULL;

//        if (rho_vec)
//            h_rho_inv_vec = (OSQPFloat*) c_calloc(m, sizeof(OSQPFloat));

        cuda_vec_copy_d2h(h_A->x, d_A->val, d_A->nnz);
        cuda_vec_int_copy_d2h(h_A->i, d_A->col_ind, d_A->nnz);
        cuda_vec_int_copy_d2h(h_A->p, d_A->row_ptr, d_A->n+1);

        //print_csc_matrix(h_A, "Polish Ared");

        s->h_kkt_csr = form_KKT(P->h_csc,       // P CSC matrix
                                h_A,            // A CSC matrix
                                1,              // CSR output format
                                s->sigma,       // Sigma offset for upper-left block (P) matrix
                                h_rho_inv_vec,  // Vector rho for lower right block
                                s->sigma,       // Scalar rho for lower right block
                                PtoKKT,         // Output mapping for P matrix to indices
                                OSQP_NULL,      // Don't need the output mapping for A indices to KKT indices
                                OSQP_NULL);     // Don't need the output mapping for rho indices to KKT indices

        //print_csr_matrix(s->h_kkt_csr, "Polish KKT");

        // Copy the update indices to the GPU
        cuda_malloc((void**) &s->d_PtoKKT, nnzP * sizeof(OSQPInt));
        cuda_vec_int_copy_h2d(s->d_PtoKKT, PtoKKT, nnzP);

        // Free the temporary parts
//        c_free(h_rho_inv_vec);
        csc_spfree(h_A);
    }
    else { // Called from ADMM algorithm
        int nnzP = P->h_csc->p[n];
        int nnzA = A->h_csc->p[n];

        // These are temporary host vectors, because we will actually be storing them on the
        // GPU for update kernels
        OSQPInt* PtoKKT   = (OSQPInt*) c_malloc(nnzP * sizeof(OSQPInt));
        OSQPInt* AtoKKT   = (OSQPInt*) c_malloc(nnzA * sizeof(OSQPInt));
        OSQPInt* rhotoKKT = (OSQPInt*) c_malloc(m * sizeof(OSQPInt));

        // Temporary vector just for KKT matrix formation
        OSQPFloat* h_rho_inv_vec = OSQP_NULL;

        // Compute the inverse rho values for the KKT matrix
        if (rho_vec) {
            OSQPInt number_of_blocks = (m / THREADS_PER_BLOCK) + 1;

            // Temporary host vector (only for KKT formation)
            h_rho_inv_vec = (OSQPFloat*) c_malloc(m * sizeof(OSQPFloat));

            // GPU vector - invert on GPU before copying it to host for KKT formation
            cuda_malloc((void**) &s->d_rho_inv_vec, m * sizeof(OSQPFloat));
            vec_reciprocal_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(s->d_rho_inv_vec, rho_vec->d_val, m);
            cuda_vec_copy_d2h(h_rho_inv_vec, s->d_rho_inv_vec, m);
        }

        // Form the KKT matrix
        s->h_kkt_csr = form_KKT(P->h_csc,       // P CSC matrix
                                A->h_csc,       // A CSC matrix
                                1,              // CSR format output
                                s->sigma,       // Sigma offset for upper-left block (P) matrix
                                h_rho_inv_vec,  // Vector rho for lower right block
                                s->rho_inv,     // Scalar rho for lower right block
                                PtoKKT,         // Output mapping P indices to KKT indices
                                AtoKKT,         // Output mapping A indices to KKT indices
                                rhotoKKT);      // Output mapping rho indices to KKT indices

        // Copy the update indices to the GPU
        cuda_malloc((void**) &s->d_PtoKKT,   nnzP * sizeof(OSQPInt));
        cuda_malloc((void**) &s->d_AtoKKT,   nnzA * sizeof(OSQPInt));
        cuda_malloc((void**) &s->d_rhotoKKT, m * sizeof(OSQPInt));

        cuda_vec_int_copy_h2d(s->d_PtoKKT,   PtoKKT,   nnzP);
        cuda_vec_int_copy_h2d(s->d_AtoKKT,   AtoKKT,   nnzA);
        cuda_vec_int_copy_h2d(s->d_rhotoKKT, rhotoKKT, m);

        // Free the temporary rho vector
        if (h_rho_inv_vec)
            c_free(h_rho_inv_vec);
    }

    // Create the device KKT matrix and move the data to it
    cuda_malloc((void**) &s->d_kkt_p, (s->h_kkt_csr->n+1) * sizeof(OSQPInt));
    cuda_malloc((void**) &s->d_kkt_i, s->h_kkt_csr->nzmax * sizeof(OSQPInt));
    cuda_malloc((void**) &s->d_kkt_x, s->h_kkt_csr->nzmax * sizeof(OSQPFloat));

    cuda_vec_int_copy_h2d(s->d_kkt_p, s->h_kkt_csr->p, s->h_kkt_csr->n+1);
    cuda_vec_int_copy_h2d(s->d_kkt_i, s->h_kkt_csr->i, s->h_kkt_csr->nzmax);
    cuda_vec_copy_h2d(s->d_kkt_x, s->h_kkt_csr->x, s->h_kkt_csr->nzmax);

    s->solveStatus = cudssMatrixCreateCsr(
        &(s->d_cudss_kkt),      // Matrix object
        n_plus_m,               // Number of rows
        n_plus_m,               // Number of columns
        s->h_kkt_csr->nzmax,    // Number of nonzeros in KKT
        s->d_kkt_p,             // Row offsets
        NULL,                   // Row indices (not needed in CSR)
        s->d_kkt_i,             // Column indices
        s->d_kkt_x,             // Values
        CUDA_INT,               // Index data type
        CUDA_FLOAT,             // Value data type
        mtype,                  // Matrix type
        mview,                  // Matrix view
        base                    // Index base
        );

    if (checkCudssErrors(s->solveStatus)) {
        free_linsys_solver_cudss(s);
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }

    // If we are using scaling, then we need to update the KKT matrix values with the values after scaling
    // because the KKT formation done above was done using unscaled values.
    if (settings->scaling) {
        OSQPInt m = s->m;
        OSQPInt number_of_blocks = (m / THREADS_PER_BLOCK) + 1;

        /* Ensure the upper-triangular part of P is updated with the scaled values before we use it */
        triu_update_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(P->S->val,                  // Copy from the full values of P
                                                                    P->d_P_triu_val,            // Store in the triu part
                                                                    P->d_P_triu_to_full_ind,    // Mapping from triangular to full
                                                                    P->S->nnz);                 // Number of non-zeros in the triangular portion

        // Scatter the scaled values from the P matrix into the KKT matrix
        // Only scatter the upper triangular portion though
        scatter_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(s->d_kkt_x, P->d_P_triu_val, s->d_PtoKKT, P->P_triu_nnz);

        // Add sigma to the P diagonal entries
        kkt_add_sigma<<<number_of_blocks, THREADS_PER_BLOCK>>>(s->d_kkt_x, s->d_kkt_p, s->sigma, s->n);

        if (!s->polishing) {
            // Scatter the scaled values from the A matrix into the KKT matrix.
            scatter_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(s->d_kkt_x, A->At->val, s->d_AtoKKT, A->At->nnz);

            if (s->rho_is_vec) {
                kkt_update_rho_vec<<<number_of_blocks, THREADS_PER_BLOCK>>>(s->d_kkt_x, rho_vec->d_val, s->d_rhotoKKT, m);
            } else {
                kkt_update_rho_sca<<<number_of_blocks, THREADS_PER_BLOCK>>>(s->d_kkt_x, settings->rho, s->d_rhotoKKT, m);
            }
        }
    }

    /* Allocate device RHS and b vectors */
    cuda_malloc((void**) &s->d_b, n_plus_m*sizeof(OSQPFloat));
    cuda_calloc((void**) &s->d_x, n_plus_m*sizeof(OSQPFloat));

    s->solveStatus = cudssMatrixCreateDn(
        &(s->d_cudss_b),            // matrix object
        n_plus_m,                   // number of rows
        1,                          // number of columns
        n_plus_m,                   // leading dimension
        s->d_b,                     // values
        CUDA_FLOAT,                 // data type
        CUDSS_LAYOUT_COL_MAJOR      // layout
        );

    if (checkCudssErrors(s->solveStatus)) {
        free_linsys_solver_cudss(s);
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }

    s->solveStatus = cudssMatrixCreateDn(
        &(s->d_cudss_x),            // matrix object
        n_plus_m,                   // number of rows
        1,                          // number of columns
        n_plus_m,                   // leading dimension
        s->d_x,                     // values
        CUDA_FLOAT,                 // data type
        CUDSS_LAYOUT_COL_MAJOR      // layout
        );

    if (checkCudssErrors(s->solveStatus)) {
        free_linsys_solver_cudss(s);
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }


    // Symbolic factorization
    s->solveStatus = cudssExecute(
        s->lib_handle,              // Library handle
        CUDSS_PHASE_ANALYSIS,       // Phase (Reordering + symbolic all in 1)
        s->config_handle,           // Config handle
        s->data_handle,             // Data handle
        s->d_cudss_kkt,             // KKT matrix
        s->d_cudss_x,               // x (unknown) vector
        s->d_cudss_b                // b (RHS) vector
        );

    checkCudaErrors(cudaStreamSynchronize(s->cudss_stream));

    if (checkCudssErrors(s->solveStatus)) {
        free_linsys_solver_cudss(s);
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }

    int info;
    size_t sizeWritten = 0;
    checkCudssErrors(cudssDataGet(s->lib_handle, s->data_handle, CUDSS_DATA_INFO, &info, sizeof(info), &sizeWritten));

    if (info != 0) {
        fprintf(stderr, "Solve failed: %d\n", info);
    }

    // Numerical factorization
    s->solveStatus = cudssExecute(
        s->lib_handle,              // Library handle
        CUDSS_PHASE_FACTORIZATION,  // Phase (Numerical factorization)
        s->config_handle,           // Config handle
        s->data_handle,             // Data handle
        s->d_cudss_kkt,             // KKT matrix
        s->d_cudss_x,               // x (unknown) vector
        s->d_cudss_b                // b (RHS) vector
        );

    checkCudaErrors(cudaStreamSynchronize(s->cudss_stream));

    if (checkCudssErrors(s->solveStatus)) {
        free_linsys_solver_cudss(s);
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }

    checkCudssErrors(cudssDataGet(s->lib_handle, s->data_handle, CUDSS_DATA_INFO, &info, sizeof(info), &sizeWritten));

    if (info != 0) {
        fprintf(stderr, "Solve failed: %d\n", info);
    }

    // Check the inertia of the KKT matrix
    OSQPInt inertia[2];
    s->solveStatus = cudssDataGet(
        s->lib_handle,          // Library handle
        s->data_handle,         // Data handle
        CUDSS_DATA_INERTIA,     // Get the intertia
        &inertia,               // Vector to save to
        sizeof(inertia),
        &sizeWritten);

    // Number of positive eigenvalues (the +inertia) must the same as the size of P,
    // otherwise the problem is non-convex
    if (inertia[0] < n) {
        free_linsys_solver_cudss(s);
        return OSQP_NONCVX_ERROR;
    }

    /* Link functions */
    s->name            = &name_cudss;
    s->solve           = &solve_linsys_cudss;
    s->warm_start      = &warm_start_linsys_solver_cudss;
    s->free            = &free_linsys_solver_cudss;
    s->update_matrices = &update_linsys_solver_matrices_cudss;
    s->update_rho_vec  = &update_linsys_solver_rho_vec_cudss;
    s->update_settings = &update_settings_linsys_solver_cudss;

    // Setup phase is over, clear the A CSC from the matrix memory, but keep P if polishing is enabled
    // TODO: Do this cleaner, so we don't have to cast away the const
    if (settings->polishing) {
        // We need to keep the copy of P ourselves and not rely on the user keeping it alive
        OSQPCscMatrix* Ptmp = csc_copy(P->h_csc);
        ((OSQPMatrix*)P)->h_csc = Ptmp;
    } else {
        ((OSQPMatrix*)P)->h_csc = OSQP_NULL;
    }

    ((OSQPMatrix*)A)->h_csc = OSQP_NULL;

    /* No error */
    return 0;
}


const char* name_cudss(cudss_solver* s) {
    int major = 0;
    int minor = 0;
    int patch = 0;

    // Keep a static string to allow us to return it with no problems
    int nameLen = 255;
    static char name[255];

    checkCudssErrors(cudssGetProperty(MAJOR_VERSION, &major));
    checkCudssErrors(cudssGetProperty(MINOR_VERSION, &minor));
    checkCudssErrors(cudssGetProperty(PATCH_LEVEL,   &patch));

    snprintf(name, nameLen, "cuDSS %d.%d.%d", major, minor, patch);
    return name;
}


OSQPInt solve_linsys_cudss(cudss_solver* s,
                           OSQPVectorf*  b,
                           OSQPInt       admm_iter) {

    OSQPInt retval = OSQP_NO_ERROR;
    OSQPInt n = s->n;
    OSQPInt m = s->m;

    // Direct solver doesn't care about the ADMM iteration
    OSQP_UnusedVar(admm_iter);

    // Update the RHS (the cuDSS matrix for b is a thin wrapper, so this is enough to update b)
    cuda_vec_copy_d2d(s->d_b, b->d_val, b->length);

/*    if (s->polishing) {
        OSQPFloat* tmp_vec = (OSQPFloat*) c_malloc(b->length*sizeof(OSQPFloat));

        cuda_vec_copy_d2h(tmp_vec, b->d_val, b->length);
        print_vec(tmp_vec, b->length, "b");
    }
*/
    osqp_profiler_sec_push(OSQP_PROFILER_SEC_LINSYS_SOLVE);

    // Actually do the solve
    s->solveStatus = cudssExecute(
        s->lib_handle,              // Library handle
        CUDSS_PHASE_SOLVE,          // Phase (Solve)
        s->config_handle,           // Config handle
        s->data_handle,             // Data handle
        s->d_cudss_kkt,             // KKT matrix
        s->d_cudss_x,               // x (unknown) vector
        s->d_cudss_b                // b (RHS) vector
        );

    checkCudaErrors(cudaStreamSynchronize(s->cudss_stream));

    if (checkCudssErrors(s->solveStatus)) {
        retval = OSQP_RUNTIME_ERROR;
    }

    int info;
    size_t sizeWritten = 0;
    checkCudssErrors(cudssDataGet(s->lib_handle, s->data_handle, CUDSS_DATA_INFO, &info, sizeof(info), &sizeWritten));

    if (info != 0) {
        fprintf(stderr, "Solve failed: %d\n", info);
        retval = OSQP_RUNTIME_ERROR;
    }

    // Return the solution through the RHS vector (the cuDSS matrix is a thin wrapper)
    if (s->polishing) {
        // Polishing uses the entire x vector for the solution
        cuda_vec_copy_d2d(b->d_val, s->d_x, b->length);

        //OSQPFloat* tmp_vec = (OSQPFloat*) c_malloc(b->length*sizeof(OSQPFloat));

//        cuda_vec_copy_d2h(tmp_vec, s->d_x, b->length);
//        print_vec(tmp_vec, b->length, "x");
    } else {
        OSQPInt number_of_blocks = (m / THREADS_PER_BLOCK) + 1;

        // Copy x_tilde into the first n variables in the output
        cuda_vec_copy_d2d(b->d_val, s->d_x, n);

        // Compute z_tilde in-place in b
        if (s->rho_is_vec) {
            /* b->d_val[j+n] += rho_inv_vec[j] * s->d_x[j+n] */
            vec_ew_fma_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(&(b->d_val[n]),      // Accumulate starting at the nth element
                                                                       s->d_rho_inv_vec,    // This runs 0-m
                                                                       &(s->d_x[n]),        // Start at the nth element
                                                                       m);                  // Only operate on the last m elements
        } else {
            /* b->d_val[j+n] += rho_inv * s->d_x[j+n] */
            checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle,
                                        m,                      // Only operate on the m last elements
                                        &(s->rho_inv),          // Scalar we multiply d_x by
                                        &(s->d_x[n]), 1,        // Start at the nth element
                                        &(b->d_val[n]), 1));    // Accumulate starting at the nth element
        }
    }

    osqp_profiler_sec_pop(OSQP_PROFILER_SEC_LINSYS_SOLVE);

    return retval;
}


void update_settings_linsys_solver_cudss(      cudss_solver* s,
                                         const OSQPSettings* settings) {
    /* No settings to update */
    OSQP_UnusedVar(s);
    OSQP_UnusedVar(settings);
    return;
}


void warm_start_linsys_solver_cudss(      cudss_solver* s,
                                    const OSQPVectorf*  x) {
    /* Warm starting not used by direct solvers */
    OSQP_UnusedVar(s);
    OSQP_UnusedVar(x);
    return;
}


void free_linsys_solver_cudss(cudss_solver* s) {
    if (s) {
        cuda_free((void**) &s->d_rho_inv_vec);
        cuda_free((void**) &s->d_PtoKKT);
        cuda_free((void**) &s->d_AtoKKT);
        cuda_free((void**) &s->d_rhotoKKT);

        // Host CSC matrix
        csc_spfree(s->h_kkt_csr);

        // cuDSS matrices
        cudssMatrixDestroy(s->d_cudss_kkt);
        cudssMatrixDestroy(s->d_cudss_x);
        cudssMatrixDestroy(s->d_cudss_b);

        // Device vectors
        cuda_free((void**) &s->d_kkt_p);
        cuda_free((void**) &s->d_kkt_i);
        cuda_free((void**) &s->d_kkt_x);
        cuda_free((void**) &s->d_x);
        cuda_free((void**) &s->d_b);

        // Library handles
        cudssDataDestroy(s->lib_handle, s->data_handle);
        cudssConfigDestroy(s->config_handle);
        cudssDestroy(s->lib_handle);
        checkCudaErrors(cudaStreamDestroy(s->cudss_stream));

        c_free(s);
    }
}


OSQPInt update_linsys_solver_matrices_cudss(      cudss_solver* s,
                                            const OSQPMatrix*   P,
                                            const OSQPInt*      Px_new_idx,
                                                  OSQPInt       P_new_n,
                                            const OSQPMatrix*   A,
                                            const OSQPInt*      Ax_new_idx,
                                                  OSQPInt       A_new_n) {
    /* Update the KKT matrix with the new data values (structure is constant)
     * cuDSS matrices are thin wrappers over the data, so updating the vector immediately
     * updates the data held in the matrix object.
     * We ignore the ability to update individual elements here, because that will be two
     * levels of indirection in the indexing, not just one. This is all on device, so the data
     * transfer should be faster.
     */
    OSQPInt m = s->m;
    OSQPInt number_of_blocks = (m / THREADS_PER_BLOCK) + 1;

    /* Ensure the upper-triangular part of P is updated before we use it */
    triu_update_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(P->S->val,                  // Copy from the full values of P
                                                                P->d_P_triu_val,            // Store in the triu part
                                                                P->d_P_triu_to_full_ind,    // Mapping from triangular to full
                                                                P->S->nnz);                 // Number of non-zeros in the triangular portion

    // Scatter the new values from the P matrix into the KKT matrix
    scatter_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(s->d_kkt_x, P->d_P_triu_val, s->d_PtoKKT, P->P_triu_nnz);

    // Scatter the new values from the A matrix into the KKT matrix.
    scatter_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(s->d_kkt_x, A->At->val, s->d_AtoKKT, A->At->nnz);

    // Add sigma to the P diagonal entries
    kkt_add_sigma<<<number_of_blocks, THREADS_PER_BLOCK>>>(s->d_kkt_x, s->d_kkt_p, s->sigma, s->n);

    osqp_profiler_sec_push(OSQP_PROFILER_SEC_LINSYS_NUM_FAC);

    s->solveStatus = cudssExecute(
        s->lib_handle,                  // Library handle
        CUDSS_PHASE_REFACTORIZATION,    // Phase (factorization)
        s->config_handle,               // Config handle
        s->data_handle,                 // Data handle
        s->d_cudss_kkt,                 // KKT matrix
        s->d_cudss_x,                   // x (unknown) vector
        s->d_cudss_b                    // b (RHS) vector
        );

    checkCudaErrors(cudaStreamSynchronize(s->cudss_stream));

    if (checkCudssErrors(s->solveStatus)) {
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }

    int info;
    size_t sizeWritten = 0;
    checkCudssErrors(cudssDataGet(s->lib_handle, s->data_handle, CUDSS_DATA_INFO, &info, sizeof(info), &sizeWritten));

    if (info != 0) {
        fprintf(stderr, "Solve failed: %d\n", info);
    }

    // Check the inertia of the KKT matrix
    OSQPInt inertia[2];
    s->solveStatus = cudssDataGet(
        s->lib_handle,          // Library handle
        s->data_handle,         // Data handle
        CUDSS_DATA_INERTIA,     // Get the intertia
        &inertia,               // Vector to save to
        sizeof(inertia),
        &sizeWritten);

    // Number of positive eigenvalues (the +inertia) must the same as the size of P,
    // otherwise the problem is non-convex
    if (inertia[0] < s->n) {
        return OSQP_NONCVX_ERROR;
    }

    return OSQP_NO_ERROR;
}


OSQPInt update_linsys_solver_rho_vec_cudss(      cudss_solver* s,
                                           const OSQPVectorf*  rho_vec,
                                                 OSQPFloat     rho_sc) {
    OSQPInt m = s->m;

    // Copy the new data values (structure is constant)
    // cuDSS matrices are thin wrappers over the data, so updating the vector immediately
    // updates the data held in the matrix object.
    // These kernels are fused operations that do the inverse when the update happens
    OSQPInt number_of_blocks = (m / THREADS_PER_BLOCK) + 1;

    if (s->rho_is_vec) {
        vec_reciprocal_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(s->d_rho_inv_vec, rho_vec->d_val, m);
        kkt_update_rho_vec<<<number_of_blocks, THREADS_PER_BLOCK>>>(s->d_kkt_x, rho_vec->d_val, s->d_rhotoKKT, m);
    } else {
        s->rho_inv = 1. / rho_sc;
        kkt_update_rho_sca<<<number_of_blocks, THREADS_PER_BLOCK>>>(s->d_kkt_x, rho_sc, s->d_rhotoKKT, m);
    }

    osqp_profiler_sec_push(OSQP_PROFILER_SEC_LINSYS_NUM_FAC);

    s->solveStatus = cudssExecute(
        s->lib_handle,                  // Library handle
        CUDSS_PHASE_REFACTORIZATION,    // Phase (factorization)
        s->config_handle,               // Config handle
        s->data_handle,                 // Data handle
        s->d_cudss_kkt,                 // KKT matrix
        s->d_cudss_x,                   // x (unknown) vector
        s->d_cudss_b                    // b (RHS) vector
        );

    checkCudaErrors(cudaStreamSynchronize(s->cudss_stream));

    if (checkCudssErrors(s->solveStatus)) {
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }

    int info;
    size_t sizeWritten = 0;
    checkCudssErrors(cudssDataGet(s->lib_handle, s->data_handle, CUDSS_DATA_INFO, &info, sizeof(info), &sizeWritten));

    if (info != 0) {
        fprintf(stderr, "Solve failed: %d\n", info);
    }

    osqp_profiler_sec_pop(OSQP_PROFILER_SEC_LINSYS_NUM_FAC);

    return 0;
}
