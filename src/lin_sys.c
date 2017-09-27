#include "lin_sys.h"


#include "suitesparse_ldl.h"   // Include only this solver in the same directory

#ifdef ENABLE_MKL_PARDISO
#include "pardiso.h"
#endif


#ifndef EMBEDDED
// Initialize linear system solver structure
// NB: Only the upper triangular part of P is stuffed!
LinSysSolver * init_linsys_solver(const csc * P, const csc * A,
                                  c_float sigma, c_float * rho_vec,
                                  enum linsys_solver_type linsys_solver, c_int polish){

    switch(linsys_solver){
    		case SUITESPARSE_LDL_SOLVER:
    			 return (LinSysSolver *) init_linsys_solver_suitesparse_ldl(P, A, sigma, rho_vec, polish);
    		#ifdef ENABLE_MKL_PARDISO
    		case PARDISO_SOLVER:
    		    return (LinSysSolver *) init_linsys_solver_pardiso(P, A, sigma, rho_vec, polish);
    		#endif
    	 	default:
            return (LinSysSolver *) init_linsys_solver_suitesparse_ldl(P, A, sigma, rho_vec, polish);
        }
}

#endif
