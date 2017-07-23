#include "scaling.h"

#if EMBEDDED != 1


// Set values lower than threshold SCALING_REG to 1
void set_to_one_zero_values(c_float * D, c_int n){
	c_int i;
	for (i = 0; i < n; i++){
		D[i] = D[i] < SCALING_REG ? 1.0 : D[i];
	}
}


/**
 * Compute infinite norm of the colums of the KKT matrix without forming it
 *
 * The norm is stored in the vector v = (D, E)
 *
 * @param P        Cost matrix
 * @param A        Contraints matrix
 * @param D        Norm of columns related to variables
 * @param D_temp_A Temporary vector for norm of columns of A
 * @param E        Norm of columns related to constraints
 * @param n        Dimension of KKT matrix
 */
void compute_inf_norm_cols_KKT(const csc * P, const csc * A,
							   c_float * D, c_float * D_temp_A,
						       c_float * E, c_int n){
	// First half
	//  [ P ]
	//  [ A ]
	mat_inf_norm_cols_sym_triu(P, D);
	mat_inf_norm_cols(A, D_temp_A);
	vec_ew_max_vec(D, D_temp_A, D, n);
	// Second half
	//  [ A']
	//  [ 0 ]
	mat_inf_norm_rows(A, E);

}


/**
 * Compute 1-norm of the colums of the KKT matrix without forming it
 *
 * The norm is stored in the vector v = (D, E)
 *
 * @param P        Cost matrix
 * @param A        Contraints matrix
 * @param D        Norm of columns related to variables
 * @param D_temp_A Temporary vector for norm of columns of A
 * @param E        Norm of columns related to constraints
 * @param n        Dimension of KKT matrix
 */
void compute_1_norm_cols_KKT(const csc * P, const csc * A,
							   c_float * D, c_float * D_temp_A,
						       c_float * E, c_int n){
	// First half
	//  [ P ]
	//  [ A ]
	mat_1_norm_cols_sym_triu(P, D);
	mat_1_norm_cols(A, D_temp_A);
	vec_ew_sum_vec(D, D_temp_A, D, n);
	// Second half
	//  [ A']
	//  [ 0 ]
	mat_1_norm_rows(A, E);

}


/**
 * Compute 2-norm of the colums of the KKT matrix without forming it
 *
 * The norm is stored in the vector v = (D, E)
 *
 * @param P        Cost matrix
 * @param A        Contraints matrix
 * @param D        Norm of columns related to variables
 * @param D_temp_A Temporary vector for norm of columns of A
 * @param E        Norm of columns related to constraints
 * @param n        Dimension of KKT matrix
 */
void compute_2_norm_cols_KKT(const csc * P, const csc * A,
							   c_float * D, c_float * D_temp_A,
						       c_float * E, c_int n){
	// First half
	//  [ P ]
	//  [ A ]
	mat_2_norm_cols_sym_triu(P, D);
	mat_2_norm_cols(A, D_temp_A);
	vec_ew_sqrt_sos_vec(D, D_temp_A, D, n);
	// Second half
	//  [ A']
	//  [ 0 ]
	mat_2_norm_rows(A, E);

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
	void (*compute_norm_cols_KKT)(const csc *, const csc *, c_float *, c_float *,
								  c_float *, c_int); // Function pointer to compute the norm of columns of KKT matrix

	n = work->data->n;
	m = work->data->m;

	// Initialize scaling vectors to 1
	vec_set_scalar(work->scaling->D, 1., work->data->n);
	vec_set_scalar(work->scaling->Dinv, 1., work->data->n);
	vec_set_scalar(work->scaling->E, 1., work->data->m);
	vec_set_scalar(work->scaling->Einv, 1., work->data->m);

	// Initialize function pointer for selected norm
	switch (work->settings->scaling_norm){
		case 1:
			compute_norm_cols_KKT = &compute_1_norm_cols_KKT;
			break;
		case 2:
			compute_norm_cols_KKT = &compute_2_norm_cols_KKT;
			break;
		case -1: // Infinity norm
			compute_norm_cols_KKT = &compute_inf_norm_cols_KKT;
			break;
	}


	for (i = 0; i < work->settings->scaling_iter; i++){
		// Compute norm of KKT columns
		compute_norm_cols_KKT(work->data->P, work->data->A,
							  work->D_temp, work->D_temp_A,
							  work->E_temp, n);

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
	// #ifdef PRINTING
	// c_print("n = %i\n", n);
	// print_vec(work->scaling->D, n, "D");
	// print_vec(work->scaling->Dinv, n, "Dinv");
	// print_vec(work->scaling->E, m, "E");
	// print_vec(work->scaling->Einv, m, "Einv");
	// #endif

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
