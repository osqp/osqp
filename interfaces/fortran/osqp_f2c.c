#include "osqp.h"
#include "glob_opts.h"

/* Settings struct (see osqp documentation) */

typedef struct {
    c_float rho;
    c_float sigma;
    c_int scaling;

#if EMBEDDED != 1
    c_int adaptive_rho;
    c_int adaptive_rho_interval;
    c_float adaptive_rho_tolerance;
#ifdef PROFILING
    c_float adaptive_rho_fraction;
#endif // Profiling
#endif // EMBEDDED != 1

    c_int max_iter;
    c_float eps_abs;
    c_float eps_rel;
    c_float eps_prim_inf;
    c_float eps_dual_inf;
    c_float alpha;
  /* no fortran equivalent
    enum linsys_solver_type linsys_solver;
  */

#ifndef EMBEDDED
    c_float delta;
    c_int polish;
    c_int polish_refine_iter;

    c_int verbose;
#endif

    c_int scaled_termination;
    c_int check_termination;
    c_int warm_start;

} OSQPFSettings;

/* Solver return information  (see osqp documentation) */

typedef struct {
	c_int iter;
	char status[32];
	c_int status_val;

#ifndef EMBEDDED
	c_int status_polish;
#endif

	c_float obj_val;
	c_float pri_res;
	c_float dua_res;

#ifdef PROFILING
	c_float setup_time;
	c_float solve_time;
	c_float polish_time;
	c_float run_time;
#endif

#if EMBEDDED != 1
	c_int rho_updates;
	c_float rho_estimate;
#endif
} OSQPFInfo;

/* interface to settings */

c_int osqp_f2c_settings( OSQPFSettings *f_settings, OSQPSettings **c_settings ){

    // allocate space for c settings

    *c_settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
    OSQPSettings *star_c_settings = *c_settings;

    // Define C solver settings as default

    set_default_settings(star_c_settings);

    // Override with fortran settings

    star_c_settings->rho = f_settings->rho;
    star_c_settings->sigma = f_settings->sigma;
    star_c_settings->scaling = f_settings->scaling;
#if EMBEDDED != 1
    star_c_settings->adaptive_rho = f_settings->adaptive_rho;
    star_c_settings->adaptive_rho_interval = f_settings->adaptive_rho_interval;
    star_c_settings->adaptive_rho_tolerance = f_settings->adaptive_rho_tolerance;
#ifdef PROFILING
    star_c_settings->adaptive_rho_fraction = f_settings->adaptive_rho_fraction;
#endif // Profiling
#endif // EMBEDDED != 1
    star_c_settings->max_iter = f_settings->max_iter;
    star_c_settings->eps_abs = f_settings->eps_abs;
    star_c_settings->eps_rel = f_settings->eps_rel;
    star_c_settings->eps_prim_inf = f_settings->eps_prim_inf;
    star_c_settings->eps_dual_inf = f_settings->eps_dual_inf;
    star_c_settings->alpha = f_settings->alpha;
  /* no fortran equivalent
    star_c_settings->linsys_solver = f_settings->linsys_solver;
  */
#ifndef EMBEDDED
    star_c_settings->delta = f_settings->delta;
    star_c_settings->polish = f_settings->polish;
    star_c_settings->polish_refine_iter = f_settings->polish_refine_iter;
    star_c_settings->verbose = f_settings->verbose;
#endif
    star_c_settings->scaled_termination = f_settings->scaled_termination;
    star_c_settings->check_termination = f_settings->check_termination;
    star_c_settings->warm_start = f_settings->warm_start;

    return 0 ;
}

/* interface to solver */

c_int osqp_f2c_solve( c_int m, c_int n,
                      c_int P_nnz, c_float *P_val, c_int *P_row, c_int *P_ptr,
                      c_int A_nnz, c_float *A_val, c_int *A_row, c_int *A_ptr,
                      c_float *q, c_float *l, c_float *u,
                      c_float *x, c_float *y, OSQPFInfo *f_info,
                      OSQPSettings *c_settings, OSQPWorkspace **work,
                      OSQPData **data ){

    c_int i;

    // Move from fortran to c indexing

    for (i = 0 ; i < n+1 ; i++) {
      P_ptr[i] = P_ptr[i]-1;
    }

    for (i = 0 ; i < P_nnz ; i++) {
      P_row[i] = P_row[i]-1;
    }

    for (i = 0 ; i < n+1 ; i++) {
      A_ptr[i] = A_ptr[i]-1;
    }

    for (i = 0 ; i < A_nnz ; i++) {
      A_row[i] = A_row[i]-1;
    }

    // allocate space for c data

    *data = (OSQPData *)c_malloc(sizeof(OSQPData));
    OSQPData *star_data = *data;

    // Populate data

    star_data->n = n;
    star_data->m = m;
    star_data->P = csc_matrix(star_data->n, star_data->n,
                              P_nnz, P_val, P_row, P_ptr);
    star_data->A = csc_matrix(star_data->m, star_data->n,
                              A_nnz, A_val, A_row, A_ptr);
    star_data->q = q;
    star_data->l = l;
    star_data->u = u;

    // Setup workspace

    *work = osqp_setup(star_data, c_settings);
    OSQPWorkspace *star_work = *work;

    // Solve Problem

    osqp_solve(star_work);

    // Record solution and dual variables

    for (i = 0 ; i < n ; i++) {
      x[i] = star_work->x[i];
    }

    for (i = 0 ; i < m ; i++) {
      y[i] = star_work->y[i];
    }

    // Record remaining output information

    f_info->iter = star_work->info->iter;
    update_status( f_info, star_work->info->status_val );
    f_info->status_val = star_work->info->status_val;
#ifndef EMBEDDED
    f_info->status_polish = star_work->info->status_polish;
#endif
    f_info->obj_val = star_work->info->obj_val;
    f_info->pri_res = star_work->info->pri_res;
    f_info->dua_res = star_work->info->dua_res;
#ifdef PROFILING
    f_info->setup_time = star_work->info->setup_time;
    f_info->solve_time = star_work->info->solve_time;
    f_info->polish_time = star_work->info->polish_time;
    f_info->run_time = star_work->info->run_time;
#endif
#if EMBEDDED != 1
    f_info->rho_updates = star_work->info->rho_updates;
    f_info->rho_estimate = star_work->info->rho_estimate;
#endif

    // Restore fortran indexing

    for (i = 0 ; i < n+1 ; i++) {
      P_ptr[i] = P_ptr[i]+1;
    }

    for (i = 0 ; i < P_nnz ; i++) {
      P_row[i] = P_row[i]+1;
    }

    for (i = 0 ; i < n+1 ; i++) {
      A_ptr[i] = A_ptr[i]+1;
    }

    for (i = 0 ; i < A_nnz ; i++) {
      A_row[i] = A_row[i]+1;
    }

    // Cleanup

    /*
    osqp_cleanup(work);

    c_free(star_data->A);
    c_free(star_data->P);
    c_free(star_data);
    c_free(c_settings);
    */

    return f_info->status_val ;
}

/* interface to cleanup */

c_int osqp_f2c_cleanup( OSQPSettings *c_settings, OSQPWorkspace *work,
                        OSQPData *data ){

    // Cleanup

    osqp_cleanup(work);

    c_free(data->A);
    c_free(data->P);
    c_free(data);
    c_free(c_settings);

    return 0 ;
}
