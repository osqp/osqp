#include "scaling.h"

#if EMBEDDED != 1


// Set values lower than threshold SCALING_REG to 1
void set_to_one_zero_values(c_float * D, c_int n){
	c_int i;
	for (i = 0; i < n; i++){
		D[i] = D[i] < SCALING_REG ? 1.0 : D[i];
	}
}




c_int scale_data(OSQPWorkspace * work){
	// Scale KKT matrix
	//
	//    [ P   A']
	//    [ A   0 ]
	//
	// with diagonal matrix
	//  
	//  S = [ D    ]
	//      [    E ]
	//


	c_int i;  // Iterations index	
	c_int n, m; // Number of constraints and variables

	n = work->data->n;
	m = work->data->m;
		
	// Initialize scaling vectors to 1
	vec_set_scalar(work->scaling->D, 1., work->data->n);
	vec_set_scalar(work->scaling->Dinv, 1., work->data->n);
	vec_set_scalar(work->scaling->E, 1., work->data->m);
	vec_set_scalar(work->scaling->Einv, 1., work->data->m);


	for (i = 0; i < work->settings->scaling_iter; i++){
		// Compute infiniy norm of KKT columns
		// First half
		//  [ P ]
		//  [ A ]
		mat_inf_norm_cols_sym_triu(work->data->P, work->D_temp);
		mat_inf_norm_cols(work->data->A, work->D_temp_A);
		vec_ew_max_vec(work->D_temp, work->D_temp_A, work->D_temp, n);	
		// Second half
		//  [ A']
		//  [ 0 ]
		mat_inf_norm_rows(work->data->A, work->E_temp);
		
		// Set to 1 values with 0 norms (avoid crazy scaling)
		set_to_one_zero_values(work->D_temp, n);
		set_to_one_zero_values(work->E_temp, m);

		// Take square root of norms
		vec_ew_sqrt(work->D_temp, n);
		vec_ew_sqrt(work->E_temp, m);

		// Divide scalings D and E by themselves
		vec_ew_recipr(work->D_temp, work->D_temp, n);
		vec_ew_recipr(work->E_temp, work->E_temp, m);
		
		// Equilibrate matrices P and A
		// P <- DPD		
		mat_premult_diag(work->data->P, work->D_temp);
		mat_postmult_diag(work->data->P, work->D_temp);
		// A <- EAD
		mat_premult_diag(work->data->A, work->E_temp);
		mat_postmult_diag(work->data->A, work->D_temp);

		// Update equilibration matrices D and E
		vec_ew_prod(work->scaling->D, work->D_temp, work->scaling->D, n);
		vec_ew_prod(work->scaling->E, work->E_temp, work->scaling->E, m);
		

	} 


	// Store Dinv, Einv in workspace
	vec_ew_recipr(work->scaling->D, work->scaling->Dinv, work->data->n);
	vec_ew_recipr(work->scaling->E, work->scaling->Einv, work->data->m);


	// Scale problem vectors q, l, u
	vec_ew_prod(work->scaling->D, work->data->q, work->data->q, work->data->n);
	vec_ew_prod(work->scaling->E, work->data->l, work->data->l, work->data->m);
	vec_ew_prod(work->scaling->E, work->data->u, work->data->u, work->data->m);


	// DEBUG
	/* c_print("n = %i\n", n); */
	/* print_vec(work->scaling->D, n, "D"); */
	/* print_vec(work->scaling->Dinv, n, "Dinv"); */
	/* print_vec(work->scaling->E, m, "E"); */
	/* print_vec(work->scaling->Einv, m, "Einv"); */


    return 0;
}


#endif  // EMBEDDED

c_int unscale_data(OSQPWorkspace * work){

    mat_premult_diag(work->data->P, work->scaling->Dinv);
    mat_postmult_diag(work->data->P, work->scaling->Dinv);
    vec_ew_prod(work->scaling->Dinv, work->data->q, work->data->q, work->data->n);

    mat_premult_diag(work->data->A, work->scaling->Einv);
    mat_postmult_diag(work->data->A, work->scaling->Dinv);
    vec_ew_prod(work->scaling->Einv, work->data->l, work->data->l, work->data->m);
    vec_ew_prod(work->scaling->Einv, work->data->u, work->data->u, work->data->m);

    return 0;
}



// // Scale solution
// c_int scale_solution(OSQPWorkspace * work){
//
//     // primal
//     vec_ew_prod(work->scaling->Dinv, work->solution->x, work->data->n);
//
//     // dual
//     vec_ew_prod(work->scaling->Einv, work->solution->y, work->data->m);
//
//     return 0;
// }


c_int unscale_solution(OSQPWorkspace * work){
    // primal
    vec_ew_prod(work->scaling->D, work->solution->x, work->solution->x, work->data->n);

    // dual
    vec_ew_prod(work->scaling->E, work->solution->y, work->solution->y, work->data->m);

    return 0;
}
