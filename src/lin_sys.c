#include "lin_sys.h"

#include "suitesparse_ldl.h"   // Include only this solver in the same directory

#ifdef ENABLE_MKL_PARDISO
#include "pardiso.h"
#include "pardiso_loader.h"
#endif


#ifndef EMBEDDED

// Load linear system solver shared library
c_int load_linsys_solver(enum linsys_solver_type linsys_solver) {

    switch(linsys_solver){
        case SUITESPARSE_LDL_SOLVER:
            // We do not laod SuiteSparse LDL solver. We have the source.
            return 0;
        #ifdef ENABLE_MKL_PARDISO
        case MKL_PARDISO_SOLVER:
            // Load Pardiso library
            return lh_load_pardiso(OSQP_NULL);
        #endif
        default:  // SuiteSparse LDL
            return 0;
    }
}

// Unload linear system solver shared library
c_int unload_linsys_solver(enum linsys_solver_type linsys_solver) {

    switch(linsys_solver){
        case SUITESPARSE_LDL_SOLVER:
            // We do not laod SuiteSparse LDL solver. We have the source.
            return 0;
        #ifdef ENABLE_MKL_PARDISO
        case MKL_PARDISO_SOLVER:
            // Unload Pardiso library
             return lh_unload_pardiso();
        #endif
        default:  // SuiteSparse LDL
            return 0;
    }
}


// Initialize linear system solver structure
// NB: Only the upper triangular part of P is stuffed!
LinSysSolver * init_linsys_solver(const csc * P, const csc * A,
                                  c_float sigma, c_float * rho_vec,
                                  enum linsys_solver_type linsys_solver, c_int polish){

    switch(linsys_solver){
    		case SUITESPARSE_LDL_SOLVER:
    			 return (LinSysSolver *) init_linsys_solver_suitesparse_ldl(P, A, sigma, rho_vec, polish);
    		#ifdef ENABLE_MKL_PARDISO
    		case MKL_PARDISO_SOLVER:
    		    return (LinSysSolver *) init_linsys_solver_pardiso(P, A, sigma, rho_vec, polish);
    		#endif
    	 	default:  // SuiteSparse LDL
            return (LinSysSolver *) init_linsys_solver_suitesparse_ldl(P, A, sigma, rho_vec, polish);
    }
}

#endif
