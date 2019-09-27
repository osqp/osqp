#include "glob_opts.h"
#include "osqp.h"
#include "auxil.h"
#include "util.h"
#include "scaling.h"
#include "error.h"

#ifndef EMBEDDED
# include "polish.h"
#endif /* ifndef EMBEDDED */

#ifdef CTRLC
# include "ctrlc.h"
#endif /* ifdef CTRLC */

#ifndef EMBEDDED
# include "lin_sys.h"
#endif /* ifndef EMBEDDED */




/**********************
* Main API Functions *
**********************/
const char* osqp_version(void) {
  return OSQP_VERSION;
}


void osqp_get_dimensions(OSQPSolver *solver,
                         c_int      *m,
                         c_int      *n) {

  /* Check if the solver has been initialized */
  if (!solver || !solver->work || !solver->work->data) {
    *m = -1;
    *n = -1;
  }
  else {
    *m = solver->work->data->m;
    *n = solver->work->data->n;
  }
}


void osqp_set_default_settings(OSQPSettings *settings) {

  settings->rho           = (c_float)RHO;            /* ADMM step */
  settings->sigma         = (c_float)SIGMA;          /* ADMM step */
  settings->scaling       = SCALING;                 /* heuristic problem scaling */
#if EMBEDDED != 1
  settings->adaptive_rho           = ADAPTIVE_RHO;
  settings->adaptive_rho_interval  = ADAPTIVE_RHO_INTERVAL;
  settings->adaptive_rho_tolerance = (c_float)ADAPTIVE_RHO_TOLERANCE;

# ifdef PROFILING
  settings->adaptive_rho_fraction = (c_float)ADAPTIVE_RHO_FRACTION;
# endif /* ifdef PROFILING */
#endif  /* if EMBEDDED != 1 */

  settings->max_iter      = MAX_ITER;                /* maximum iterations */
  settings->eps_abs       = (c_float)EPS_ABS;        /* absolute convergence tolerance */
  settings->eps_rel       = (c_float)EPS_REL;        /* relative convergence tolerance */
  settings->eps_prim_inf  = (c_float)EPS_PRIM_INF;   /* primal infeasibility tolerance */
  settings->eps_dual_inf  = (c_float)EPS_DUAL_INF;   /* dual infeasibility   tolerance */
  settings->alpha         = (c_float)ALPHA;          /* relaxation parameter */
  settings->linsys_solver = LINSYS_SOLVER;           /* relaxation parameter */

#ifndef EMBEDDED
  settings->delta              = DELTA;              /* regularization parameter for polish */
  settings->polish             = POLISH;             /* ADMM solution polish: 1 */
  settings->polish_refine_iter = POLISH_REFINE_ITER; /* iterative refinement steps in polish */
  settings->verbose            = VERBOSE;            /* print output */
#endif /* ifndef EMBEDDED */

  settings->scaled_termination = SCALED_TERMINATION; /* Evaluate scaled termination criteria*/
  settings->check_termination  = CHECK_TERMINATION;  /* Interval for evaluating termination criteria */
  settings->warm_start         = WARM_START;         /* warm starting */

#ifdef PROFILING
  settings->time_limit = TIME_LIMIT;
#endif /* ifdef PROFILING */
}

#ifndef EMBEDDED


c_int osqp_setup(OSQPSolver** solverp,
                 const csc* P,
                 const c_float* q,
                 const csc* A,
                 const c_float* l,
                 const c_float* u,
                 c_int m,
                 c_int n,
                 const OSQPSettings *settings) {

  c_int exitflag;

  OSQPSolver*    solver;
  OSQPWorkspace* work;

  // Validate data
  if (validate_data(P,q,A,l,u,m,n)) return osqp_error(OSQP_DATA_VALIDATION_ERROR);

  // Validate settings
  if (validate_settings(settings)) return osqp_error(OSQP_SETTINGS_VALIDATION_ERROR);

  // Allocate empty solver
  solver = c_calloc(1, sizeof(OSQPSolver));
  if (!(solver)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  *solverp = solver;

  // Allocate empty workspace
  work   = c_calloc(1, sizeof(OSQPWorkspace));
  if (!(work)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  solver->work = work;

  // Start and allocate directly timer
# ifdef PROFILING
  work->timer = c_malloc(sizeof(OSQPTimer));
  if (!(work->timer)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  osqp_tic(work->timer);
# endif /* ifdef PROFILING */

  // Initialize algebra libraries
  exitflag = algebra_init_libs();
  if (exitflag) {
    return osqp_error(OSQP_ALGEBRA_LOAD_ERROR);
  }

  // Copy problem data into workspace
  work->data = c_malloc(sizeof(OSQPData));
  if (!(work->data)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  work->data->m = m;
  work->data->n = n;

  // Cost function
  work->data->P = OSQPMatrix_new_from_csc(P,1);   //copy assuming triu form
  work->data->q = OSQPVectorf_new(q,n);
  if (!(work->data->P) || !(work->data->q)) return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Constraints
  work->data->A = OSQPMatrix_new_from_csc(A,0); //assumes non-triu form (i.e. full)
  if (!(work->data->A)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  work->data->l = OSQPVectorf_new(l,m);
  work->data->u = OSQPVectorf_new(u,m);
  if (!(work->data->l) || !(work->data->u))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Vectorized rho parameter
  work->rho_vec     = OSQPVectorf_malloc(m);
  work->rho_inv_vec = OSQPVectorf_malloc(m);
  if (!(work->rho_vec) || !(work->rho_inv_vec))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Type of constraints
  work->constr_type = OSQPVectori_calloc(m);
  if (!(work->constr_type)) return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Allocate internal solver variables (ADMM steps)
  work->x           = OSQPVectorf_calloc(n);
  work->z           = OSQPVectorf_calloc(m);
  work->xz_tilde    = OSQPVectorf_calloc(n + m);
  work->xtilde_view = OSQPVectorf_view(work->xz_tilde,0,n);
  work->ztilde_view = OSQPVectorf_view(work->xz_tilde,n,m);
  work->x_prev      = OSQPVectorf_calloc(n);
  work->z_prev      = OSQPVectorf_calloc(m);
  work->y           = OSQPVectorf_calloc(m);
  if (!(work->x) || !(work->z) || !(work->xz_tilde))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  if (!(work->xtilde_view) || !(work->ztilde_view))
      return osqp_error(OSQP_MEM_ALLOC_ERROR);
  if (!(work->x_prev) || !(work->z_prev) || !(work->y))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Initialize variables x, y, z to 0
  osqp_cold_start(solver);

  // Primal and dual residuals variables
  work->Ax  = OSQPVectorf_calloc(m);
  work->Px  = OSQPVectorf_calloc(n);
  work->Aty = OSQPVectorf_calloc(n);

  // Primal infeasibility variables
  work->delta_y   = OSQPVectorf_calloc(m);
  work->Atdelta_y = OSQPVectorf_calloc(n);

  // Dual infeasibility variables
  work->delta_x  = OSQPVectorf_calloc(n);
  work->Pdelta_x = OSQPVectorf_calloc(n);
  work->Adelta_x = OSQPVectorf_calloc(m);

  if (!(work->Ax) || !(work->Px) || !(work->Aty))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  if (!(work->delta_y) || !(work->Atdelta_y))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  if (!(work->delta_x) || !(work->Pdelta_x) || !(work->Adelta_x))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Copy settings
  solver->settings = copy_settings(settings);
  if (!(solver->settings)) return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Perform scaling
  if (settings->scaling) {
    // Allocate scaling structure
    work->scaling = c_malloc(sizeof(OSQPScaling));
    if (!(work->scaling)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
    work->scaling->D    = OSQPVectorf_calloc(n);
    work->scaling->Dinv = OSQPVectorf_calloc(n);
    work->scaling->E    = OSQPVectorf_calloc(m);
    work->scaling->Einv = OSQPVectorf_calloc(m);
    if (!(work->scaling->D) || !(work->scaling->Dinv) ||
        !(work->scaling->E) || !(work->scaling->Einv))
      return osqp_error(OSQP_MEM_ALLOC_ERROR);


    // Allocate workspace variables used in scaling
    work->D_temp   = OSQPVectorf_calloc(n);
    work->D_temp_A = OSQPVectorf_calloc(n);
    work->E_temp   = OSQPVectorf_calloc(m);
    if (!(work->D_temp) || !(work->D_temp_A) || !(work->E_temp))
      return osqp_error(OSQP_MEM_ALLOC_ERROR);

    // Scale data
    scale_data(solver);
  } else {
    work->scaling  = OSQP_NULL;
    work->D_temp   = OSQP_NULL;
    work->D_temp_A = OSQP_NULL;
    work->E_temp   = OSQP_NULL;
  }

  // Set type of constraints.  Ignore return value
  // because we will definitely factor KKT.
  set_rho_vec(solver);

  // Load linear system solver
  if (load_linsys_solver(settings->linsys_solver)) return osqp_error(OSQP_LINSYS_SOLVER_LOAD_ERROR);

  // Initialize linear system solver structure
  exitflag = init_linsys_solver(&(work->linsys_solver), work->data->P, work->data->A,
                                settings->sigma,
                                work->rho_vec,
                                settings->linsys_solver, 0);

  if (exitflag) {
    return osqp_error(exitflag);
  }

  // Initialize active constraints structure
  work->pol = c_malloc(sizeof(OSQPPolish));
  if (!(work->pol)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  work->pol->active_flags = OSQPVectori_malloc(m);
  work->pol->x            = OSQPVectorf_malloc(n);
  work->pol->z            = OSQPVectorf_malloc(m);
  work->pol->y            = OSQPVectorf_malloc(m);
  if (!(work->pol->x)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  if (!(work->pol->active_flags) ||
      !(work->pol->z) || !(work->pol->y))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Allocate solution
  solver->solution = c_calloc(1, sizeof(OSQPSolution));
  if (!(solver->solution)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  solver->solution->x             = c_calloc(1, n * sizeof(c_float));
  solver->solution->y             = c_calloc(1, m * sizeof(c_float));
  solver->solution->prim_inf_cert = c_calloc(1, m * sizeof(c_float));
  solver->solution->dual_inf_cert = c_calloc(1, n * sizeof(c_float));
  if ( !(solver->solution->x) || !(solver->solution->dual_inf_cert) )
    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  if ( m && (!(solver->solution->y) || !(solver->solution->prim_inf_cert)) )
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Allocate and initialize information
  solver->info = c_calloc(1, sizeof(OSQPInfo));
  if (!(solver->info)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  solver->info->status_polish = 0;              // Polishing not performed
  update_status(solver->info, OSQP_UNSOLVED);
# ifdef PROFILING
  solver->info->solve_time  = 0.0;                   // Solve time to zero
  solver->info->update_time = 0.0;                   // Update time to zero
  solver->info->polish_time = 0.0;                   // Polish time to zero
  solver->info->run_time    = 0.0;                   // Total run time to zero
  solver->info->setup_time  = osqp_toc(work->timer); // Update timer information

  work->first_run         = 1;
  work->clear_update_time = 0;
  work->rho_update_from_solve = 0;
# endif /* ifdef PROFILING */
  solver->info->rho_updates  = 0;                      // Rho updates set to 0
  solver->info->rho_estimate = solver->settings->rho;  // Best rho estimate

  // Print header
# ifdef PRINTING
  if (solver->settings->verbose) print_setup_header(solver);
  work->summary_printed = 0; // Initialize last summary  to not printed
# endif /* ifdef PRINTING */


  // If adaptive rho and automatic interval, but profiling disabled, we need to
  // set the interval to a default value
# ifndef PROFILING
  if (solver->settings->adaptive_rho && !solver->settings->adaptive_rho_interval) {
    if (solver->settings->check_termination) {
      // If check_termination is enabled, we set it to a multiple of the check
      // termination interval
      solver->settings->adaptive_rho_interval = ADAPTIVE_RHO_MULTIPLE_TERMINATION *
                                              solver->settings->check_termination;
    } else {
      // If check_termination is disabled we set it to a predefined fix number
      solver->settings->adaptive_rho_interval = ADAPTIVE_RHO_FIXED;
    }
  }
# endif /* ifndef PROFILING */

  // Return exit flag
  return 0;
}

#endif // #ifndef EMBEDDED


c_int osqp_solve(OSQPSolver *solver) {

  c_int exitflag;
  c_int iter, max_iter;
  c_int compute_cost_function; // Boolean: compute the cost function in the loop or not
  c_int can_check_termination; // Boolean: check termination or not
  OSQPWorkspace* work;

#ifdef PROFILING
  c_float temp_run_time;       // Temporary variable to store current run time
#endif /* ifdef PROFILING */

#ifdef PRINTING
  c_int can_print;             // Boolean whether you can print
#endif /* ifdef PRINTING */

  // Check if solver has been initialized
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);
  work = solver->work;

#ifdef PROFILING
  if (work->clear_update_time == 1)
    solver->info->update_time = 0.0;
  work->rho_update_from_solve = 1;
#endif /* ifdef PROFILING */

  // Initialize variables
  exitflag              = 0;
  can_check_termination = 0;
#ifdef PRINTING
  can_print = solver->settings->verbose;
#endif /* ifdef PRINTING */
#ifdef PRINTING
  compute_cost_function = solver->settings->verbose;       // Compute cost function only
                                                   // if verbose is on
#else /* ifdef PRINTING */
  compute_cost_function = 0;                       // Never compute cost
                                                   // function during the
                                                   // iterations if no printing
                                                   // enabled
#endif /* ifdef PRINTING */



#ifdef PROFILING
  osqp_tic(work->timer); // Start timer
#endif /* ifdef PROFILING */


#ifdef PRINTING

  if (solver->settings->verbose) {
    // Print Header for every column
    print_header();
  }
#endif /* ifdef PRINTING */

#ifdef CTRLC

  // initialize Ctrl-C support
  osqp_start_interrupt_listener();
#endif /* ifdef CTRLC */

  // Initialize variables (cold start or warm start depending on settings)
  if (!solver->settings->warm_start) osqp_cold_start(solver);  // If not warm start ->
                                                               // set x, z, y to zero

  // Main ADMM algorithm

  max_iter = solver->settings->max_iter;
  for (iter = 1; iter <= max_iter; iter++) {

    // Update x_prev, z_prev (preallocated, no malloc)
    swap_vectors(&(work->x), &(work->x_prev));
    swap_vectors(&(work->z), &(work->z_prev));

    /* ADMM STEPS */
    /* Compute \tilde{x}^{k+1}, \tilde{z}^{k+1} */
    update_xz_tilde(solver);

    /* Compute x^{k+1} */
    update_x(solver);

    /* Compute z^{k+1} */
    update_z(solver);

    /* Compute y^{k+1} */
    update_y(solver);

    /* End of ADMM Steps */

#ifdef CTRLC

    // Check the interrupt signal
    if (osqp_is_interrupted()) {
      update_status(solver->info, OSQP_SIGINT);
# ifdef PRINTING
      c_print("Solver interrupted\n");
# endif /* ifdef PRINTING */
      exitflag = 1;
      goto exit;
    }
#endif /* ifdef CTRLC */

#ifdef PROFILING

    // Check if solver time_limit is enabled. In case, check if the current
    // run time is more than the time_limit option.
    if (work->first_run) {
      temp_run_time = solver->info->setup_time + osqp_toc(work->timer);
    }
    else {
      temp_run_time = solver->info->update_time + osqp_toc(work->timer);
    }

    if (solver->settings->time_limit &&
        (temp_run_time >= solver->settings->time_limit)) {
      update_status(solver->info, OSQP_TIME_LIMIT_REACHED);
# ifdef PRINTING

      if (solver->settings->verbose) c_print("run time limit reached\n");
      can_print = 0;  // Not printing at this iteration
# endif /* ifdef PRINTING */
      break;
    }
#endif /* ifdef PROFILING */


    // Can we check for termination ?
    can_check_termination = solver->settings->check_termination &&
                            (iter % solver->settings->check_termination == 0);

#ifdef PRINTING

    // Can we print ?
    can_print = solver->settings->verbose &&
                ((iter % PRINT_INTERVAL == 0) || (iter == 1));

    if (can_check_termination || can_print) { // Update status in either of
                                              // these cases
      // Update information
      update_info(solver, iter, compute_cost_function, 0);

      if (can_print) {
        // Print summary
        print_summary(solver);
      }

      if (can_check_termination) {
        // Check algorithm termination
        if (check_termination(solver, 0)) {
          // Terminate algorithm
          break;
        }
      }
    }
#else /* ifdef PRINTING */

    if (can_check_termination) {
      // Update information and compute also objective value
      update_info(solver, iter, compute_cost_function, 0);

      // Check algorithm termination
      if (check_termination(solver, 0)) {
        // Terminate algorithm
        break;
      }
    }
#endif /* ifdef PRINTING */


#if EMBEDDED != 1
# ifdef PROFILING

    // If adaptive rho with automatic interval, check if the solve time is a
    // certain fraction
    // of the setup time.
    if (solver->settings->adaptive_rho && !solver->settings->adaptive_rho_interval) {
      // Check time
      if (osqp_toc(work->timer) >
          solver->settings->adaptive_rho_fraction * solver->info->setup_time) {
        // Enough time has passed. We now get the number of iterations between
        // the updates.
        if (solver->settings->check_termination) {
          // If check_termination is enabled, we round the number of iterations
          // between
          // rho updates to the closest multiple of check_termination
          solver->settings->adaptive_rho_interval =
          (c_int)c_roundmultiple(iter, solver->settings->check_termination);
         }
         else {
          // If check_termination is disabled, we round the number of iterations
          // between
          // updates to the closest multiple of the default check_termination
          // interval.
          solver->settings->adaptive_rho_interval = (c_int)c_roundmultiple(iter,
                                                                         CHECK_TERMINATION);
        }

        // Make sure the interval is not 0 and at least check_termination times
          solver->settings->adaptive_rho_interval = c_max(
          solver->settings->adaptive_rho_interval,
          solver->settings->check_termination);
      } // If time condition is met
    }   // If adaptive rho enabled and interval set to auto®
# endif // PROFILING

    // Adapt rho
    if (solver->settings->adaptive_rho &&
        solver->settings->adaptive_rho_interval &&
        (iter % solver->settings->adaptive_rho_interval == 0)) {
      // Update info with the residuals if it hasn't been done before
# ifdef PRINTING

      if (!can_check_termination && !can_print) {
        // Information has not been computed neither for termination or printing
        // reasons
        update_info(solver, iter, compute_cost_function, 0);
      }
# else /* ifdef PRINTING */

      if (!can_check_termination) {
        // Information has not been computed before for termination check
        update_info(solver, iter, compute_cost_function, 0);
      }
# endif /* ifdef PRINTING */

      // Actually update rho
      if (adapt_rho(solver)) {
# ifdef PRINTING
        c_eprint("Failed rho update");
# endif // PRINTING
        exitflag = 1;
        goto exit;
      }
    }
#endif // EMBEDDED != 1

  }        // End of ADMM for loop


  // Update information and check termination condition if it hasn't been done
  // during last iteration (max_iter reached or check_termination disabled)
  if (!can_check_termination) {
    /* Update information */
#ifdef PRINTING

    if (!can_print) {
      // Update info only if it hasn't been updated before for printing
      // reasons
      update_info(solver, iter - 1, compute_cost_function, 0);
    }
#else /* ifdef PRINTING */

    // If no printing is enabled, update info directly
    update_info(solver, iter - 1, compute_cost_function, 0);
#endif /* ifdef PRINTING */

#ifdef PRINTING

    /* Print summary */
    if (solver->settings->verbose && !work->summary_printed) print_summary(solver);
#endif /* ifdef PRINTING */

    /* Check whether a termination criterion is triggered */
    check_termination(solver, 0);

  }

  // Compute objective value in case it was not
  // computed during the iterations
  if (!compute_cost_function && has_solution(solver->info)){
    solver->info->obj_val = compute_obj_val(solver, work->x);
  }


#ifdef PRINTING
  /* Print summary for last iteration */
  if (solver->settings->verbose && !work->summary_printed) {
    print_summary(solver);
  }
#endif /* ifdef PRINTING */

  /* if max iterations reached, change status accordingly */
  if (solver->info->status_val == OSQP_UNSOLVED) {
    if (!check_termination(solver, 1)) { // Try to check for approximate
      update_status(solver->info, OSQP_MAX_ITER_REACHED);
    }
  }

#ifdef PROFILING
  /* if time-limit reached check termination and update status accordingly */
 if (solver->info->status_val == OSQP_TIME_LIMIT_REACHED) {
    if (!check_termination(solver, 1)) { // Try for approximate solutions
      update_status(solver->info, OSQP_TIME_LIMIT_REACHED); /* Change update status back to OSQP_TIME_LIMIT_REACHED */
    }
  }
#endif /* ifdef PROFILING */


#if EMBEDDED != 1
  /* Update rho estimate */
  solver->info->rho_estimate = compute_rho_estimate(solver);
#endif /* if EMBEDDED != 1 */

  /* Update solve time */
#ifdef PROFILING
  solver->info->solve_time = osqp_toc(work->timer);
#endif /* ifdef PROFILING */


#ifndef EMBEDDED
  // Polish the obtained solution
  if (solver->settings->polish && (solver->info->status_val == OSQP_SOLVED))
    polish(solver);
#endif /* ifndef EMBEDDED */

#ifdef PROFILING
  /* Update total time */
  if (work->first_run) {
    // total time: setup + solve + polish
    solver->info->run_time = solver->info->setup_time +
                             solver->info->solve_time +
                             solver->info->polish_time;
  } else {
    // total time: update + solve + polish
    solver->info->run_time = solver->info->update_time +
                             solver->info->solve_time +
                             solver->info->polish_time;
  }

  // Indicate that the solve function has already been executed
  if (work->first_run) work->first_run = 0;

  // Indicate that the update_time should be set to zero
  work->clear_update_time = 1;

  // Indicate that osqp_update_rho is not called from osqp_solve
  work->rho_update_from_solve = 0;
#endif /* ifdef PROFILING */

#ifdef PRINTING
  /* Print final footer */
  if (solver->settings->verbose) print_footer(solver->info, solver->settings->polish);
#endif /* ifdef PRINTING */

  // Store solution
  store_solution(solver);


// Define exit flag for quitting function
#if defined(PROFILING) || defined(CTRLC) || EMBEDDED != 1
exit:
#endif /* if defined(PROFILING) || defined(CTRLC) || EMBEDDED != 1 */

#ifdef CTRLC
  // Restore previous signal handler
  osqp_end_interrupt_listener();
#endif /* ifdef CTRLC */

  return exitflag;
}


#ifndef EMBEDDED

c_int osqp_cleanup(OSQPSolver *solver) {

  c_int exitflag = 0;
  OSQPWorkspace* work;

  if(!solver) return 0;   //exit on null

  work = solver->work;

  if (work) { // If workspace has been allocated
    // Free algebra library handlers
    algebra_free_libs();

    // Free Data
    if (work->data) {
      OSQPMatrix_free(work->data->P);
      OSQPMatrix_free(work->data->A);
      OSQPVectorf_free(work->data->q);
      OSQPVectorf_free(work->data->l);
      OSQPVectorf_free(work->data->u);
      c_free(work->data);
    }

    // Free scaling
    if (work->scaling){
      OSQPVectorf_free(work->scaling->D);
      OSQPVectorf_free(work->scaling->Dinv);
      OSQPVectorf_free(work->scaling->E);
      OSQPVectorf_free(work->scaling->Einv);
    }
    c_free(work->scaling);

    // Free workspace variables
    OSQPVectorf_free(work->D_temp);
    OSQPVectorf_free(work->D_temp_A);
    OSQPVectorf_free(work->E_temp);

    // Free linear system solver structure
    if (work->linsys_solver) {
      if (work->linsys_solver->free) {
        work->linsys_solver->free(work->linsys_solver);
      }
    }

    // Unload linear system solver after free
    if (solver->settings) {
      exitflag = unload_linsys_solver(solver->settings->linsys_solver);
    }

#ifndef EMBEDDED
    // Free active constraints structure
    if (work->pol) {
      OSQPVectori_free(work->pol->active_flags);
      OSQPVectorf_free(work->pol->x);
      OSQPVectorf_free(work->pol->z);
      OSQPVectorf_free(work->pol->y);
      c_free(work->pol);
    }
#endif /* ifndef EMBEDDED */

    // Free other Variables
    OSQPVectorf_free(work->rho_vec);
    OSQPVectorf_free(work->rho_inv_vec);
#if EMBEDDED != 1
    OSQPVectori_free(work->constr_type);
#endif
    OSQPVectorf_free(work->x);
    OSQPVectorf_free(work->z);
    OSQPVectorf_free(work->xz_tilde);
    OSQPVectorf_view_free(work->xtilde_view);
    OSQPVectorf_view_free(work->ztilde_view);
    OSQPVectorf_free(work->x_prev);
    OSQPVectorf_free(work->z_prev);
    OSQPVectorf_free(work->y);
    OSQPVectorf_free(work->Ax);
    OSQPVectorf_free(work->Px);
    OSQPVectorf_free(work->Aty);
    OSQPVectorf_free(work->delta_y);
    OSQPVectorf_free(work->Atdelta_y);
    OSQPVectorf_free(work->delta_x);
    OSQPVectorf_free(work->Pdelta_x);
    OSQPVectorf_free(work->Adelta_x);

    // Free Settings
    if (solver->settings) c_free(solver->settings);

    // Free solution
    if (solver->solution) {
      c_free(solver->solution->x);
      c_free(solver->solution->y);
      c_free(solver->solution->prim_inf_cert);
      c_free(solver->solution->dual_inf_cert);
      c_free(solver->solution);
    }

    // Free information
    if (solver->info) c_free(solver->info);

# ifdef PROFILING
    // Free timer
    if (work->timer) c_free(work->timer);
# endif /* ifdef PROFILING */

    // Free work
    c_free(work);
  }

  // Free solver
  c_free(solver);

  return exitflag;
}

#endif // #ifndef EMBEDDED


/************************
* Update problem data  *
************************/
c_int osqp_update_lin_cost(OSQPSolver *solver, const c_float *q_new) {

  OSQPWorkspace* work;

  // Check if workspace has been initialized
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);
  work = solver->work;

#ifdef PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time = 0;
    solver->info->update_time = 0.0;
  }
  osqp_tic(work->timer); // Start timer
#endif /* ifdef PROFILING */

  // Replace q by the new vector
  OSQPVectorf_from_raw(work->data->q, q_new);

  // Scaling
  if (solver->settings->scaling) {
    OSQPVectorf_ew_prod(work->data->q, work->data->q, work->scaling->D);
    OSQPVectorf_mult_scalar(work->data->q, work->scaling->c);
  }

  // Reset solver information
  reset_info(solver->info);

#ifdef PROFILING
  solver->info->update_time += osqp_toc(work->timer);
#endif /* ifdef PROFILING */

  return 0;
}

c_int osqp_update_bounds(OSQPSolver    *solver,
                         const c_float *l_new,
                         const c_float *u_new) {
  
  c_int i, exitflag = 0;
  OSQPVectorf *l_tmp, *u_tmp;
  OSQPWorkspace* work;

  /* If both l_new and u_new are empty, then return */
  if (!l_new && !u_new) return 0;

  /* Check if workspace has been initialized */
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);
  work = solver->work;

#ifdef PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time    = 0;
    solver->info->update_time  = 0.0;
  }
  osqp_tic(work->timer); /* Start timer */
#endif /* ifdef PROFILING */

  /* Use z_prev and delta_y to store l_new and u_new */
  l_tmp = work->z_prev;
  u_tmp = work->delta_y;

  /* Copy l_new and u_new to l_tmp and u_tmp */
  if (l_new) OSQPVectorf_from_raw(l_tmp, l_new);
  if (u_new) OSQPVectorf_from_raw(u_tmp, u_new);

  /* Scaling */
  if (solver->settings->scaling) {
    if (l_new) OSQPVectorf_ew_prod(l_tmp, l_tmp, work->scaling->E);
    if (u_new) OSQPVectorf_ew_prod(u_tmp, u_tmp, work->scaling->E);
  }

  /* Check if upper bound is greater than lower bound */
  if (l_new && u_new)
    exitflag = !OSQPVectorf_all_leq(l_tmp, u_tmp);
  else if (l_new)
    exitflag = !OSQPVectorf_all_leq(l_tmp, work->data->u);
  else
    exitflag = !OSQPVectorf_all_leq(work->data->l, u_tmp);

  if (exitflag) {
#ifdef PRINTING
    c_eprint("lower bound must be lower than or equal to upper bound");
#endif /* ifdef PRINTING */
    return 1;
  }

  /*  Swap vectors l_tmp and l.
   *  NB: Use work->z_prev rather than l_tmp.
   */
  if (l_new) swap_vectors(&work->z_prev, &work->data->l);

  /*  Swap vectors u_tmp and u.
   *  NB: Use work->delta_y rather than u_tmp.
   */
  if (u_new) swap_vectors(&work->delta_y, &work->data->u);

  /* Reset solver information */
  reset_info(solver->info);

#if EMBEDDED != 1
  /* Update rho_vec and refactor if constraints type changes */
  exitflag = update_rho_vec(solver);
#endif /* EMBEDDED != 1 */

#ifdef PROFILING
  solver->info->update_time += osqp_toc(work->timer);
#endif /* ifdef PROFILING */

  return exitflag;
}

void osqp_cold_start(OSQPSolver *solver) {
  OSQPWorkspace* work  = solver->work;
  OSQPVectorf_set_scalar(work->x, 0.);
  OSQPVectorf_set_scalar(work->z, 0.);
  OSQPVectorf_set_scalar(work->y, 0.);
}

c_int osqp_warm_start(OSQPSolver    *solver,
                      const c_float *x,
                      const c_float *y) {

  OSQPWorkspace* work;

  /* Check if workspace has been initialized */
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);
  work = solver->work;

  /* Update warm_start setting to true */
  if (!solver->settings->warm_start) solver->settings->warm_start = 1;

  /* Copy primal and dual variables into the iterates */
  if (x) OSQPVectorf_from_raw(work->x, x);
  if (y) OSQPVectorf_from_raw(work->y, y);

  /* Scale iterates */
  if (solver->settings->scaling) {
    if (x) OSQPVectorf_ew_prod(work->x, work->x, work->scaling->Dinv);
    if (y) {
      OSQPVectorf_ew_prod(work->y, work->y, work->scaling->Einv);
      OSQPVectorf_mult_scalar(work->y, work->scaling->c);
    }
  }

  /* Compute Ax = z and store it in z */
  if (x) OSQPMatrix_Axpy(work->data->A, work->x, work->z, 1.0, 0.0);

  return 0;
}


#if EMBEDDED != 1

c_int osqp_update_P(OSQPSolver    *solver,
                    const c_float *Px_new,
                    const c_int   *Px_new_idx,
                    c_int          P_new_n) {

  c_int exitflag; // Exit flag
  c_int nnzP;     // Number of nonzeros in P
  OSQPWorkspace* work;

  // Check if workspace has been initialized
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);
  work = solver->work;

#ifdef PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time = 0;
    solver->info->update_time = 0.0;
  }
  osqp_tic(work->timer); // Start timer
#endif /* ifdef PROFILING */

  nnzP = OSQPMatrix_get_nz(work->data->P);

  if (Px_new_idx) { // Passing the index of elements changed
    // Check if number of elements is less or equal than the total number of
    // nonzeros in P
    if (P_new_n > nnzP) {
# ifdef PRINTING
      c_eprint("new number of elements (%i) greater than elements in P (%i)",
               (int)P_new_n,
               (int)nnzP);
# endif /* ifdef PRINTING */
      return 1;
    }
  }

  if (solver->settings->scaling) {
    // Unscale data
    unscale_data(solver);
  }

  OSQPMatrix_update_values(work->data->P,Px_new,Px_new_idx,P_new_n);

  if (solver->settings->scaling) {
    // Scale data
    scale_data(solver);
  }

  // Update linear system structure with new data
  exitflag = work->linsys_solver->update_matrices(work->linsys_solver,
                                                  work->data->P,
                                                  work->data->A);

  // Reset solver information
  reset_info(solver->info);

# ifdef PRINTING

  if (exitflag < 0) {
    c_eprint("new KKT matrix is not quasidefinite");
  }
# endif /* ifdef PRINTING */

#ifdef PROFILING
  solver->info->update_time += osqp_toc(work->timer);
#endif /* ifdef PROFILING */

  return exitflag;
}


c_int osqp_update_A(OSQPSolver *solver,
                    const c_float *Ax_new,
                    const c_int   *Ax_new_idx,
                    c_int          A_new_n) {

  c_int exitflag; // Exit flag
  c_int nnzA;     // Number of nonzeros in A
  OSQPWorkspace* work;

  // Check if workspace has been initialized
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);
  work = solver->work;

#ifdef PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time = 0;
    solver->info->update_time = 0.0;
  }
  osqp_tic(work->timer); // Start timer
#endif /* ifdef PROFILING */

  nnzA = OSQPMatrix_get_nz(work->data->A);

  if (Ax_new_idx) { // Passing the index of elements changed
    // Check if number of elements is less or equal than the total number of
    // nonzeros in A
    if (A_new_n > nnzA) {
# ifdef PRINTING
      c_eprint("new number of elements (%i) greater than elements in A (%i)",
               (int)A_new_n,
               (int)nnzA);
# endif /* ifdef PRINTING */
      return 1;
    }
  }

  if (solver->settings->scaling) {
    // Unscale data
    unscale_data(solver);
  }

  // Update A elements
  OSQPMatrix_update_values(work->data->A,Ax_new,Ax_new_idx,A_new_n);

  if (solver->settings->scaling) {
    // Scale data
    scale_data(solver);
  }

  // Update linear system structure with new data
  exitflag = work->linsys_solver->update_matrices(work->linsys_solver,
                                                  work->data->P,
                                                  work->data->A);

  // Reset solver information
  reset_info(solver->info);

# ifdef PRINTING

  if (exitflag < 0) {
    c_eprint("new KKT matrix is not quasidefinite");
  }
# endif /* ifdef PRINTING */

#ifdef PROFILING
  solver->info->update_time += osqp_toc(work->timer);
#endif /* ifdef PROFILING */

  return exitflag;
}


c_int osqp_update_P_A(OSQPSolver    *solver,
                      const c_float *Px_new,
                      const c_int   *Px_new_idx,
                      c_int          P_new_n,
                      const c_float *Ax_new,
                      const c_int   *Ax_new_idx,
                      c_int          A_new_n) {

  c_int exitflag;   // Exit flag
  c_int nnzP, nnzA; // Number of nonzeros in P and A
  OSQPWorkspace* work;

  // Check if workspace has been initialized
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);
  work = solver->work;

#ifdef PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time = 0;
    solver->info->update_time = 0.0;
  }
  osqp_tic(work->timer); // Start timer
#endif /* ifdef PROFILING */

  nnzP = OSQPMatrix_get_nz(work->data->P);
  nnzA = OSQPMatrix_get_nz(work->data->A);


  if (Px_new_idx) { // Passing the index of elements changed
    // Check if number of elements is less or equal than the total number of
    // nonzeros in P
    if (P_new_n > nnzP) {
# ifdef PRINTING
      c_eprint("new number of elements (%i) greater than elements in P (%i)",
               (int)P_new_n,
               (int)nnzP);
# endif /* ifdef PRINTING */
      return 1;
    }
  }


  if (Ax_new_idx) { // Passing the index of elements changed
    // Check if number of elements is less or equal than the total number of
    // nonzeros in A
    if (A_new_n > nnzA) {
# ifdef PRINTING
      c_eprint("new number of elements (%i) greater than elements in A (%i)",
               (int)A_new_n,
               (int)nnzA);
# endif /* ifdef PRINTING */
      return 2;
    }
  }

  if (solver->settings->scaling) {
    // Unscale data
    unscale_data(solver);
  }

  // Update P elements
    OSQPMatrix_update_values(work->data->P,Px_new,Px_new_idx,P_new_n);

  // Update A elements
    OSQPMatrix_update_values(work->data->A,Ax_new,Ax_new_idx,A_new_n);

  if (solver->settings->scaling) {
    // Scale data
    scale_data(solver);
  }

  // Update linear system structure with new data
  exitflag = work->linsys_solver->update_matrices(work->linsys_solver,
                                                  work->data->P,
                                                  work->data->A);

  // Reset solver information
  reset_info(solver->info);

# ifdef PRINTING

  if (exitflag < 0) {
    c_eprint("new KKT matrix is not quasidefinite");
  }
# endif /* ifdef PRINTING */

#ifdef PROFILING
  solver->info->update_time += osqp_toc(work->timer);
#endif /* ifdef PROFILING */

  return exitflag;
}

c_int osqp_update_rho(OSQPSolver *solver, c_float rho_new) {

    c_int exitflag;
    OSQPWorkspace* work;

    // Check if workspace has been initialized
    if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);
    work = solver->work;

  // Check value of rho
  if (rho_new <= 0) {
# ifdef PRINTING
    c_eprint("rho must be positive");
# endif /* ifdef PRINTING */
    return 1;
  }

#ifdef PROFILING
  if (work->rho_update_from_solve == 0) {
    if (work->clear_update_time == 1) {
      work->clear_update_time = 0;
      solver->info->update_time = 0.0;
    }
    osqp_tic(work->timer); // Start timer
  }
#endif /* ifdef PROFILING */

  // Update rho in settings
  solver->settings->rho = c_min(c_max(rho_new, RHO_MIN), RHO_MAX);

  // Update rho_vec and rho_inv_vec
  OSQPVectorf_set_scalar_conditional(work->rho_vec,
                                     work->constr_type,
                                     RHO_MIN,                                     //const  == -1
                                     solver->settings->rho,                       //constr == 0
                                     RHO_EQ_OVER_RHO_INEQ*solver->settings->rho); //constr == 1

  OSQPVectorf_ew_reciprocal(work->rho_inv_vec, work->rho_vec);

  // Update rho_vec in KKT matrix
  exitflag = work->linsys_solver->update_rho_vec(work->linsys_solver, work->rho_vec);

#ifdef PROFILING
  if (work->rho_update_from_solve == 0)
    solver->info->update_time += osqp_toc(work->timer);
#endif /* ifdef PROFILING */

  return exitflag;
}

#endif // EMBEDDED != 1

/****************************
* Update problem settings  *
****************************/
c_int osqp_update_max_iter(OSQPSolver *solver, c_int max_iter_new) {

  // Check if workspace has been initialized
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that max_iter is positive
  if (max_iter_new <= 0) {
#ifdef PRINTING
    c_eprint("max_iter must be positive");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update max_iter
  solver->settings->max_iter = max_iter_new;

  return 0;
}

c_int osqp_update_eps_abs(OSQPSolver*solver, c_float eps_abs_new) {

  // Check if workspace has been initialized
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that eps_abs is positive
  if (eps_abs_new < 0.) {
#ifdef PRINTING
    c_eprint("eps_abs must be nonnegative");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update eps_abs
  solver->settings->eps_abs = eps_abs_new;

  return 0;
}

c_int osqp_update_eps_rel(OSQPSolver *solver, c_float eps_rel_new) {

  // Check if workspace has been initialized
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that eps_rel is positive
  if (eps_rel_new < 0.) {
#ifdef PRINTING
    c_eprint("eps_rel must be nonnegative");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update eps_rel
  solver->settings->eps_rel = eps_rel_new;

  return 0;
}

c_int osqp_update_eps_prim_inf(OSQPSolver *solver, c_float eps_prim_inf_new) {

  // Check if workspace has been initialized

  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that eps_prim_inf is positive
  if (eps_prim_inf_new < 0.) {
#ifdef PRINTING
    c_eprint("eps_prim_inf must be nonnegative");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update eps_prim_inf
  solver->settings->eps_prim_inf = eps_prim_inf_new;

  return 0;
}

c_int osqp_update_eps_dual_inf(OSQPSolver *solver, c_float eps_dual_inf_new) {

  // Check if workspace has been initialized
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that eps_dual_inf is positive
  if (eps_dual_inf_new < 0.) {
#ifdef PRINTING
    c_eprint("eps_dual_inf must be nonnegative");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update eps_dual_inf
  solver->settings->eps_dual_inf = eps_dual_inf_new;


  return 0;
}

c_int osqp_update_alpha(OSQPSolver*solver, c_float alpha_new) {

  // Check if workspace has been initialized
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that alpha is between 0 and 2
  if ((alpha_new <= 0.) || (alpha_new >= 2.)) {
#ifdef PRINTING
    c_eprint("alpha must be between 0 and 2");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update alpha
  solver->settings->alpha = alpha_new;

  return 0;
}

c_int osqp_update_warm_start(OSQPSolver *solver, c_int warm_start_new) {

  // Check if workspace has been initialized
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that warm_start is either 0 or 1
  if ((warm_start_new != 0) && (warm_start_new != 1)) {
#ifdef PRINTING
    c_eprint("warm_start should be either 0 or 1");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update warm_start
  solver->settings->warm_start = warm_start_new;

  return 0;
}

c_int osqp_update_scaled_termination(OSQPSolver *solver, c_int scaled_termination_new) {

  // Check if workspace has been initialized
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that scaled_termination is either 0 or 1
  if ((scaled_termination_new != 0) && (scaled_termination_new != 1)) {
#ifdef PRINTING
    c_eprint("scaled_termination should be either 0 or 1");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update scaled_termination
  solver->settings->scaled_termination = scaled_termination_new;

  return 0;
}

c_int osqp_update_check_termination(OSQPSolver *solver, c_int check_termination_new) {

  // Check if workspace has been initialized
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that check_termination is nonnegative
  if (check_termination_new < 0) {
#ifdef PRINTING
    c_eprint("check_termination should be nonnegative");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update check_termination
  solver->settings->check_termination = check_termination_new;

  return 0;
}

#ifndef EMBEDDED

c_int osqp_update_delta(OSQPSolver *solver, c_float delta_new) {

  // Check if workspace has been initialized
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that delta is positive
  if (delta_new <= 0.) {
# ifdef PRINTING
    c_eprint("delta must be positive");
# endif /* ifdef PRINTING */
    return 1;
  }

  // Update delta
  solver->settings->delta = delta_new;

  return 0;
}

c_int osqp_update_polish(OSQPSolver *solver, c_int polish_new) {

  // Check if workspace has been initialized
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that polish is either 0 or 1
  if ((polish_new != 0) && (polish_new != 1)) {
# ifdef PRINTING
    c_eprint("polish should be either 0 or 1");
# endif /* ifdef PRINTING */
    return 1;
  }

  // Update polish
  solver->settings->polish = polish_new;

# ifdef PROFILING

  // Reset polish time to zero
  solver->info->polish_time = 0.0;
# endif /* ifdef PROFILING */

  return 0;
}

c_int osqp_update_polish_refine_iter(OSQPSolver *solver, c_int polish_refine_iter_new) {

  // Check if workspace has been initialized
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that polish_refine_iter is nonnegative
  if (polish_refine_iter_new < 0) {
# ifdef PRINTING
    c_eprint("polish_refine_iter must be nonnegative");
# endif /* ifdef PRINTING */
    return 1;
  }

  // Update polish_refine_iter
  solver->settings->polish_refine_iter = polish_refine_iter_new;

  return 0;
}

c_int osqp_update_verbose(OSQPSolver *solver, c_int verbose_new) {

  // Check if workspace has been initialized
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that verbose is either 0 or 1
  if ((verbose_new != 0) && (verbose_new != 1)) {
# ifdef PRINTING
    c_eprint("verbose should be either 0 or 1");
# endif /* ifdef PRINTING */
    return 1;
  }

  // Update verbose
  solver->settings->verbose = verbose_new;

  return 0;
}

#endif // EMBEDDED

#ifdef PROFILING
c_int osqp_update_time_limit(OSQPSolver *solver, c_float time_limit_new) {

  // Check if workspace has been initialized
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that time_limit is nonnegative
  if (time_limit_new < 0.) {
# ifdef PRINTING
    c_print("time_limit must be nonnegative\n");
# endif /* ifdef PRINTING */
    return 1;
  }

  // Update time_limit
  solver->settings->time_limit = time_limit_new;

  return 0;
}
#endif /* ifdef PROFILING */

/****************************
* User API Helper functions
****************************/


void csc_set_data(csc* M, c_int m, c_int n, c_int nzmax, c_float *x, c_int *i, c_int *p)
{
  M->m     = m;
  M->n     = n;
  M->nz   = -1;
  M->nzmax = nzmax;
  M->x     = x;
  M->i     = i;
  M->p     = p;
}
