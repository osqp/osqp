#include "lin_sys.h"



#ifndef EMBEDDED
// Initialize linear system solver structure
// NB: Only the upper triangular part of P is stuffed!
LinSysSolver * init_linsys_solver(const csc * P, const csc * A,
                const OSQPSettings *settings, c_int polish){

	switch(settings->linsys_solver){
		case SUITESPARSE_LDL: 
			return (LinSysSolver *) init_linsys_solver_suitesparse_ldl(P, A, settings, polish);
	 	default:
			return (LinSysSolver *) init_linsys_solver_suitesparse_ldl(P, A, settings, polish);
        }
}

#endif
