#include "osqp.h"
#include "auxil.h"
#include "util.h"
#include "scaling.h"
#include "glob_opts.h"
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
void osqp_set_default_settings(OSQPSettings *settings) {

  settings->rho           = (c_float)RHO;            /* ADMM step */
  settings->sigma         = (c_float)SIGMA;          /* ADMM step */
  settings->scaling = SCALING;                       /* heuristic problem scaling */
#if EMBEDDED != 1
  settings->adaptive_rho           = ADAPTIVE_RHO;
  settings->adaptive_rho_interval  = ADAPTIVE_RHO_INTERVAL;
  settings->adaptive_rho_tolerance = (c_float)ADAPTIVE_RHO_TOLERANCE;

# ifdef PROFILING
  settings->adaptive_rho_fraction = (c_float)ADAPTIVE_RHO_FRACTION;
# endif /* ifdef PROFILING */
#endif  /* if EMBEDDED != 1 */

  settings->max_iter      = MAX_ITER;                /* maximum iterations to
                                                        take */
  settings->eps_abs       = (c_float)EPS_ABS;        /* absolute convergence
                                                        tolerance */
  settings->eps_rel       = (c_float)EPS_REL;        /* relative convergence
                                                        tolerance */
  settings->eps_prim_inf  = (c_float)EPS_PRIM_INF;   /* primal infeasibility
                                                        tolerance */
  settings->eps_dual_inf  = (c_float)EPS_DUAL_INF;   /* dual infeasibility
                                                        tolerance */
  settings->alpha         = (c_float)ALPHA;          /* relaxation parameter */
  settings->linsys_solver = LINSYS_SOLVER;           /* relaxation parameter */

#ifndef EMBEDDED
  settings->delta              = DELTA;              /* regularization parameter
                                                        for polish */
  settings->polish             = POLISH;             /* ADMM solution polish: 1
                                                      */
  settings->polish_refine_iter = POLISH_REFINE_ITER; /* iterative refinement
                                                        steps in polish */
  settings->verbose            = VERBOSE;            /* print output */
#endif /* ifndef EMBEDDED */

  settings->scaled_termination = SCALED_TERMINATION; /* Evaluate scaled
                                                        termination criteria*/
  settings->check_termination  = CHECK_TERMINATION;  /* Interval for evaluating
                                                        termination criteria */
  settings->warm_start         = WARM_START;         /* warm starting */

#ifdef PROFILING
  settings->time_limit = TIME_LIMIT;
#endif /* ifdef PROFILING */
}

#ifndef EMBEDDED


c_int osqp_setup(OSQPWorkspace** workp, const OSQPData *data, const OSQPSettings *settings) {
  c_int exitflag;

  OSQPWorkspace * work;

  // Validate data
  if (validate_data(data)) return osqp_error(OSQP_DATA_VALIDATION_ERROR);

  // Validate settings
  if (validate_settings(settings)) return osqp_error(OSQP_SETTINGS_VALIDATION_ERROR);

  // Allocate empty workspace
  work = c_calloc(1, sizeof(OSQPWorkspace));
  if (!(work)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  *workp = work;

  // Start and allocate directly timer
# ifdef PROFILING
  work->timer = c_malloc(sizeof(OSQPTimer));
  if (!(work->timer)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  osqp_tic(work->timer);
# endif /* ifdef PROFILING */

  // Copy problem data into workspace
  work->data = c_malloc(sizeof(OSQPData));
  if (!(work->data)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  work->data->n = data->n;
  work->data->m = data->m;

  // Cost function
  work->data->P = copy_csc_mat(data->P);
  work->data->q = vec_copy(data->q, data->n);
  if (!(work->data->P) || !(work->data->q)) return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Constraints
  work->data->A = copy_csc_mat(data->A);
  if (!(work->data->A)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  work->data->l = vec_copy(data->l, data->m);
  work->data->u = vec_copy(data->u, data->m);
  if ( data->m && (!(work->data->l) || !(work->data->u)) )
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Vectorized rho parameter
  work->rho_vec     = c_malloc(data->m * sizeof(c_float));
  work->rho_inv_vec = c_malloc(data->m * sizeof(c_float));
  if ( data->m && (!(work->rho_vec) || !(work->rho_inv_vec)) )
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Type of constraints
  work->constr_type = c_calloc(data->m, sizeof(c_int));
  if (data->m && !(work->constr_type)) return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Allocate internal solver variables (ADMM steps)
  work->x        = c_calloc(data->n, sizeof(c_float));
  work->z        = c_calloc(data->m, sizeof(c_float));
  work->xz_tilde = c_calloc(data->n + data->m, sizeof(c_float));
  work->x_prev   = c_calloc(data->n, sizeof(c_float));
  work->z_prev   = c_calloc(data->m, sizeof(c_float));
  work->y        = c_calloc(data->m, sizeof(c_float));
  if (!(work->x) || !(work->xz_tilde) || !(work->x_prev))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  if ( data->m && (!(work->z) || !(work->z_prev) || !(work->y)) )
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Initialize variables x, y, z to 0
  cold_start(work);

  // Primal and dual residuals variables
  work->Ax  = c_calloc(data->m, sizeof(c_float));
  work->Px  = c_calloc(data->n, sizeof(c_float));
  work->Aty = c_calloc(data->n, sizeof(c_float));

  // Primal infeasibility variables
  work->delta_y   = c_calloc(data->m, sizeof(c_float));
  work->Atdelta_y = c_calloc(data->n, sizeof(c_float));

  // Dual infeasibility variables
  work->delta_x  = c_calloc(data->n, sizeof(c_float));
  work->Pdelta_x = c_calloc(data->n, sizeof(c_float));
  work->Adelta_x = c_calloc(data->m, sizeof(c_float));

  if (!(work->Px) || !(work->Aty) || !(work->Atdelta_y) ||
      !(work->delta_x) || !(work->Pdelta_x))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  if ( data->m && (!(work->Ax) || !(work->delta_y) || !(work->Adelta_x)) )
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Copy settings
  work->settings = copy_settings(settings);
  if (!(work->settings)) return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Perform scaling
  if (settings->scaling) {
    // Allocate scaling structure
    work->scaling = c_malloc(sizeof(OSQPScaling));
    if (!(work->scaling)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
    work->scaling->D    = c_malloc(data->n * sizeof(c_float));
    work->scaling->Dinv = c_malloc(data->n * sizeof(c_float));
    work->scaling->E    = c_malloc(data->m * sizeof(c_float));
    work->scaling->Einv = c_malloc(data->m * sizeof(c_float));
    if (!(work->scaling->D) || !(work->scaling->Dinv))
      return osqp_error(OSQP_MEM_ALLOC_ERROR);
    if ( data->m && (!(work->scaling->E) || !(work->scaling->Einv)) )
      return osqp_error(OSQP_MEM_ALLOC_ERROR);


    // Allocate workspace variables used in scaling
    work->D_temp   = c_malloc(data->n * sizeof(c_float));
    work->D_temp_A = c_malloc(data->n * sizeof(c_float));
    work->E_temp   = c_malloc(data->m * sizeof(c_float));
    // if (!(work->D_temp) || !(work->D_temp_A) || !(work->E_temp))
    //   return osqp_error(OSQP_MEM_ALLOC_ERROR);
    if (!(work->D_temp) || !(work->D_temp_A)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
    if (data->m && !(work->E_temp))           return osqp_error(OSQP_MEM_ALLOC_ERROR);

    // Scale data
    scale_data(work);
  } else {
    work->scaling  = OSQP_NULL;
    work->D_temp   = OSQP_NULL;
    work->D_temp_A = OSQP_NULL;
    work->E_temp   = OSQP_NULL;
  }

  // Set type of constraints
  set_rho_vec(work);

  // Load linear system solver
  if (load_linsys_solver(work->settings->linsys_solver)) return osqp_error(OSQP_LINSYS_SOLVER_LOAD_ERROR);

  // Initialize linear system solver structure
  exitflag = init_linsys_solver(&(work->linsys_solver), work->data->P, work->data->A,
                                work->settings->sigma, work->rho_vec,
                                work->settings->linsys_solver, 0);

  if (exitflag) {
    return osqp_error(exitflag);
  }

  // Initialize active constraints structure
  work->pol = c_malloc(sizeof(OSQPPolish));
  if (!(work->pol)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  work->pol->Alow_to_A = c_malloc(data->m * sizeof(c_int));
  work->pol->Aupp_to_A = c_malloc(data->m * sizeof(c_int));
  work->pol->A_to_Alow = c_malloc(data->m * sizeof(c_int));
  work->pol->A_to_Aupp = c_malloc(data->m * sizeof(c_int));
  work->pol->x         = c_malloc(data->n * sizeof(c_float));
  work->pol->z         = c_malloc(data->m * sizeof(c_float));
  work->pol->y         = c_malloc(data->m * sizeof(c_float));
  if (!(work->pol->x)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  if ( data->m && (!(work->pol->Alow_to_A) || !(work->pol->Aupp_to_A) ||
      !(work->pol->A_to_Alow) || !(work->pol->A_to_Aupp) ||
      !(work->pol->z) || !(work->pol->y)) )
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Allocate solution
  work->solution = c_calloc(1, sizeof(OSQPSolution));
  if (!(work->solution)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  work->solution->x = c_calloc(1, data->n * sizeof(c_float));
  work->solution->y = c_calloc(1, data->m * sizeof(c_float));
  if (!(work->solution->x))            return osqp_error(OSQP_MEM_ALLOC_ERROR);
  if (data->m && !(work->solution->y)) return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Allocate and initialize information
  work->info = c_calloc(1, sizeof(OSQPInfo));
  if (!(work->info)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  work->info->status_polish = 0;              // Polishing not performed
  update_status(work->info, OSQP_UNSOLVED);
# ifdef PROFILING
  work->info->solve_time  = 0.0;                   // Solve time to zero
  work->info->update_time = 0.0;                   // Update time to zero
  work->info->polish_time = 0.0;                   // Polish time to zero
  work->info->run_time    = 0.0;                   // Total run time to zero
  work->info->setup_time  = osqp_toc(work->timer); // Update timer information

  work->first_run         = 1;
  work->clear_update_time = 0;
  work->rho_update_from_solve = 0;
# endif /* ifdef PROFILING */
  work->info->rho_updates  = 0;                    // Rho updates set to 0
  work->info->rho_estimate = work->settings->rho;  // Best rho estimate

  // Print header
# ifdef PRINTING
  if (work->settings->verbose) print_setup_header(work);
  work->summary_printed = 0; // Initialize last summary  to not printed
# endif /* ifdef PRINTING */


  // If adaptive rho and automatic interval, but profiling disabled, we need to
  // set the interval to a default value
# ifndef PROFILING
  if (work->settings->adaptive_rho && !work->settings->adaptive_rho_interval) {
    if (work->settings->check_termination) {
      // If check_termination is enabled, we set it to a multiple of the check
      // termination interval
      work->settings->adaptive_rho_interval = ADAPTIVE_RHO_MULTIPLE_TERMINATION *
                                              work->settings->check_termination;
    } else {
      // If check_termination is disabled we set it to a predefined fix number
      work->settings->adaptive_rho_interval = ADAPTIVE_RHO_FIXED;
    }
  }
# endif /* ifndef PROFILING */

  // Return exit flag
  return 0;
}

#endif // #ifndef EMBEDDED


c_int osqp_solve(OSQPWorkspace *work) {

  c_int exitflag;
  c_int iter;
  c_int compute_cost_function; // Boolean: compute the cost function in the loop or not
  c_int can_check_termination; // Boolean: check termination or not

#ifdef PROFILING
  c_float temp_run_time;       // Temporary variable to store current run time
#endif /* ifdef PROFILING */

#ifdef PRINTING
  c_int can_print;             // Boolean whether you can print
#endif /* ifdef PRINTING */

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

#ifdef PROFILING
  if (work->clear_update_time == 1)
    work->info->update_time = 0.0;
  work->rho_update_from_solve = 1;
#endif /* ifdef PROFILING */

  // Initialize variables
  exitflag              = 0;
  can_check_termination = 0;
#ifdef PRINTING
  can_print = work->settings->verbose;
#endif /* ifdef PRINTING */
#ifdef PRINTING
  compute_cost_function = work->settings->verbose; // Compute cost function only
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

  if (work->settings->verbose) {
    // Print Header for every column
    print_header();
  }
#endif /* ifdef PRINTING */

#ifdef CTRLC

  // initialize Ctrl-C support
  osqp_start_interrupt_listener();
#endif /* ifdef CTRLC */

  // Initialize variables (cold start or warm start depending on settings)
  if (!work->settings->warm_start) cold_start(work);  // If not warm start ->
                                                      // set x, z, y to zero

  // Main ADMM algorithm
  for (iter = 1; iter <= work->settings->max_iter; iter++) {
    // Update x_prev, z_prev (preallocated, no malloc)
    swap_vectors(&(work->x), &(work->x_prev));
    swap_vectors(&(work->z), &(work->z_prev));

    /* ADMM STEPS */
    /* Compute \tilde{x}^{k+1}, \tilde{z}^{k+1} */
    update_xz_tilde(work);

    /* Compute x^{k+1} */
    update_x(work);

    /* Compute z^{k+1} */
    update_z(work);

    /* Compute y^{k+1} */
    update_y(work);

    /* End of ADMM Steps */

#ifdef CTRLC

    // Check the interrupt signal
    if (osqp_is_interrupted()) {
      update_status(work->info, OSQP_SIGINT);
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
      temp_run_time = work->info->setup_time + osqp_toc(work->timer);
    }
    else {
      temp_run_time = work->info->update_time + osqp_toc(work->timer);
    }

    if (work->settings->time_limit &&
        (temp_run_time >= work->settings->time_limit)) {
      update_status(work->info, OSQP_TIME_LIMIT_REACHED);
# ifdef PRINTING
      if (work->settings->verbose) c_print("run time limit reached\n");
      can_print = 0;  // Not printing at this iteration
# endif /* ifdef PRINTING */
      break;
    }
#endif /* ifdef PROFILING */


    // Can we check for termination ?
    can_check_termination = work->settings->check_termination &&
                            (iter % work->settings->check_termination == 0);

#ifdef PRINTING

    // Can we print ?
    can_print = work->settings->verbose &&
                ((iter % PRINT_INTERVAL == 0) || (iter == 1));

    if (can_check_termination || can_print) { // Update status in either of
                                              // these cases
      // Update information
      update_info(work, iter, compute_cost_function, 0);

      if (can_print) {
        // Print summary
        print_summary(work);
      }

      if (can_check_termination) {
        // Check algorithm termination
        if (check_termination(work, 0)) {
          // Terminate algorithm
          break;
        }
      }
    }
#else /* ifdef PRINTING */

    if (can_check_termination) {
      // Update information and compute also objective value
      update_info(work, iter, compute_cost_function, 0);

      // Check algorithm termination
      if (check_termination(work, 0)) {
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
    if (work->settings->adaptive_rho && !work->settings->adaptive_rho_interval) {
      // Check time
      if (osqp_toc(work->timer) >
          work->settings->adaptive_rho_fraction * work->info->setup_time) {
        // Enough time has passed. We now get the number of iterations between
        // the updates.
        if (work->settings->check_termination) {
          // If check_termination is enabled, we round the number of iterations
          // between
          // rho updates to the closest multiple of check_termination
          work->settings->adaptive_rho_interval = (c_int)c_roundmultiple(iter,
                                                                         work->settings->check_termination);
        } else {
          // If check_termination is disabled, we round the number of iterations
          // between
          // updates to the closest multiple of the default check_termination
          // interval.
          work->settings->adaptive_rho_interval = (c_int)c_roundmultiple(iter,
                                                                         CHECK_TERMINATION);
        }

        // Make sure the interval is not 0 and at least check_termination times
        work->settings->adaptive_rho_interval = c_max(
          work->settings->adaptive_rho_interval,
          work->settings->check_termination);
      } // If time condition is met
    }   // If adaptive rho enabled and interval set to auto
# endif // PROFILING

    // Adapt rho
    if (work->settings->adaptive_rho &&
        work->settings->adaptive_rho_interval &&
        (iter % work->settings->adaptive_rho_interval == 0)) {
      // Update info with the residuals if it hasn't been done before
# ifdef PRINTING

      if (!can_check_termination && !can_print) {
        // Information has not been computed neither for termination or printing
        // reasons
        update_info(work, iter, compute_cost_function, 0);
      }
# else /* ifdef PRINTING */

      if (!can_check_termination) {
        // Information has not been computed before for termination check
        update_info(work, iter, compute_cost_function, 0);
      }
# endif /* ifdef PRINTING */

      // Actually update rho
      if (adapt_rho(work)) {
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
      update_info(work, iter - 1, compute_cost_function, 0);
    }
#else /* ifdef PRINTING */

    // If no printing is enabled, update info directly
    update_info(work, iter - 1, compute_cost_function, 0);
#endif /* ifdef PRINTING */

#ifdef PRINTING

    /* Print summary */
    if (work->settings->verbose && !work->summary_printed) print_summary(work);
#endif /* ifdef PRINTING */

    /* Check whether a termination criterion is triggered */
    check_termination(work, 0);
  }

  // Compute objective value in case it was not
  // computed during the iterations
  if (!compute_cost_function && has_solution(work->info)){
    work->info->obj_val = compute_obj_val(work, work->x);
  }


#ifdef PRINTING
  /* Print summary for last iteration */
  if (work->settings->verbose && !work->summary_printed) {
    print_summary(work);
  }
#endif /* ifdef PRINTING */

  /* if max iterations reached, change status accordingly */
  if (work->info->status_val == OSQP_UNSOLVED) {
    if (!check_termination(work, 1)) { // Try to check for approximate
      update_status(work->info, OSQP_MAX_ITER_REACHED);
    }
  }

#ifdef PROFILING
  /* if time-limit reached check termination and update status accordingly */
 if (work->info->status_val == OSQP_TIME_LIMIT_REACHED) {
    if (!check_termination(work, 1)) { // Try for approximate solutions
      update_status(work->info, OSQP_TIME_LIMIT_REACHED); /* Change update status back to OSQP_TIME_LIMIT_REACHED */
    }
  }
#endif /* ifdef PROFILING */


#if EMBEDDED != 1
  /* Update rho estimate */
  work->info->rho_estimate = compute_rho_estimate(work);
#endif /* if EMBEDDED != 1 */

  /* Update solve time */
#ifdef PROFILING
  work->info->solve_time = osqp_toc(work->timer);
#endif /* ifdef PROFILING */


#ifndef EMBEDDED
  // Polish the obtained solution
  if (work->settings->polish && (work->info->status_val == OSQP_SOLVED))
    polish(work);
#endif /* ifndef EMBEDDED */

#ifdef PROFILING
  /* Update total time */
  if (work->first_run) {
    // total time: setup + solve + polish
    work->info->run_time = work->info->setup_time +
                           work->info->solve_time +
                           work->info->polish_time;
  } else {
    // total time: update + solve + polish
    work->info->run_time = work->info->update_time +
                           work->info->solve_time +
                           work->info->polish_time;
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
  if (work->settings->verbose) print_footer(work->info, work->settings->polish);
#endif /* ifdef PRINTING */

  // Store solution
  store_solution(work);


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

c_int osqp_cleanup(OSQPWorkspace *work) {
  c_int exitflag = 0;

  if (work) { // If workspace has been allocated
    // Free Data
    if (work->data) {
      if (work->data->P) csc_spfree(work->data->P);
      if (work->data->A) csc_spfree(work->data->A);
      if (work->data->q) c_free(work->data->q);
      if (work->data->l) c_free(work->data->l);
      if (work->data->u) c_free(work->data->u);
      c_free(work->data);
    }

    // Free scaling variables
    if (work->scaling){
      if (work->scaling->D)    c_free(work->scaling->D);
      if (work->scaling->Dinv) c_free(work->scaling->Dinv);
      if (work->scaling->E)    c_free(work->scaling->E);
      if (work->scaling->Einv) c_free(work->scaling->Einv);
      c_free(work->scaling);
    }

    // Free temp workspace variables for scaling
    if (work->D_temp)   c_free(work->D_temp);
    if (work->D_temp_A) c_free(work->D_temp_A);
    if (work->E_temp)   c_free(work->E_temp);

    // Free linear system solver structure
    if (work->linsys_solver) {
      if (work->linsys_solver->free) {
        work->linsys_solver->free(work->linsys_solver);
      }
    }

    // Unload linear system solver after free
    if (work->settings) {
      exitflag = unload_linsys_solver(work->settings->linsys_solver);
    }

#ifndef EMBEDDED
    // Free active constraints structure
    if (work->pol) {
      if (work->pol->Alow_to_A) c_free(work->pol->Alow_to_A);
      if (work->pol->Aupp_to_A) c_free(work->pol->Aupp_to_A);
      if (work->pol->A_to_Alow) c_free(work->pol->A_to_Alow);
      if (work->pol->A_to_Aupp) c_free(work->pol->A_to_Aupp);
      if (work->pol->x)         c_free(work->pol->x);
      if (work->pol->z)         c_free(work->pol->z);
      if (work->pol->y)         c_free(work->pol->y);
      c_free(work->pol);
    }
#endif /* ifndef EMBEDDED */

    // Free other Variables
    if (work->rho_vec)     c_free(work->rho_vec);
    if (work->rho_inv_vec) c_free(work->rho_inv_vec);
#if EMBEDDED != 1
    if (work->constr_type) c_free(work->constr_type);
#endif
    if (work->x)           c_free(work->x);
    if (work->z)           c_free(work->z);
    if (work->xz_tilde)    c_free(work->xz_tilde);
    if (work->x_prev)      c_free(work->x_prev);
    if (work->z_prev)      c_free(work->z_prev);
    if (work->y)           c_free(work->y);
    if (work->Ax)          c_free(work->Ax);
    if (work->Px)          c_free(work->Px);
    if (work->Aty)         c_free(work->Aty);
    if (work->delta_y)     c_free(work->delta_y);
    if (work->Atdelta_y)   c_free(work->Atdelta_y);
    if (work->delta_x)     c_free(work->delta_x);
    if (work->Pdelta_x)    c_free(work->Pdelta_x);
    if (work->Adelta_x)    c_free(work->Adelta_x);

    // Free Settings
    if (work->settings) c_free(work->settings);

    // Free solution
    if (work->solution) {
      if (work->solution->x) c_free(work->solution->x);
      if (work->solution->y) c_free(work->solution->y);
      c_free(work->solution);
    }

    // Free information
    if (work->info) c_free(work->info);

# ifdef PROFILING
    // Free timer
    if (work->timer) c_free(work->timer);
# endif /* ifdef PROFILING */

    // Free work
    c_free(work);
  }

  return exitflag;
}

#endif // #ifndef EMBEDDED


/************************
* Update problem data  *
************************/
c_int osqp_update_lin_cost(OSQPWorkspace *work, const c_float *q_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

#ifdef PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time = 0;
    work->info->update_time = 0.0;
  }
  osqp_tic(work->timer); // Start timer
#endif /* ifdef PROFILING */

  // Replace q by the new vector
  prea_vec_copy(q_new, work->data->q, work->data->n);

  // Scaling
  if (work->settings->scaling) {
    vec_ew_prod(work->scaling->D, work->data->q, work->data->q, work->data->n);
    vec_mult_scalar(work->data->q, work->scaling->c, work->data->n);
  }

  // Reset solver information
  reset_info(work->info);

#ifdef PROFILING
  work->info->update_time += osqp_toc(work->timer);
#endif /* ifdef PROFILING */

  return 0;
}

c_int osqp_update_bounds(OSQPWorkspace *work,
                         const c_float *l_new,
                         const c_float *u_new) {
  c_int i, exitflag = 0;

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

#ifdef PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time = 0;
    work->info->update_time = 0.0;
  }
  osqp_tic(work->timer); // Start timer
#endif /* ifdef PROFILING */

  // Check if lower bound is smaller than upper bound
  for (i = 0; i < work->data->m; i++) {
    if (l_new[i] > u_new[i]) {
#ifdef PRINTING
      c_eprint("lower bound must be lower than or equal to upper bound");
#endif /* ifdef PRINTING */
      return 1;
    }
  }

  // Replace l and u by the new vectors
  prea_vec_copy(l_new, work->data->l, work->data->m);
  prea_vec_copy(u_new, work->data->u, work->data->m);

  // Scaling
  if (work->settings->scaling) {
    vec_ew_prod(work->scaling->E, work->data->l, work->data->l, work->data->m);
    vec_ew_prod(work->scaling->E, work->data->u, work->data->u, work->data->m);
  }

  // Reset solver information
  reset_info(work->info);

#if EMBEDDED != 1
  // Update rho_vec and refactor if constraints type changes
  exitflag = update_rho_vec(work);
#endif // EMBEDDED != 1

#ifdef PROFILING
  work->info->update_time += osqp_toc(work->timer);
#endif /* ifdef PROFILING */

  return exitflag;
}

c_int osqp_update_lower_bound(OSQPWorkspace *work, const c_float *l_new) {
  c_int i, exitflag = 0;

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

#ifdef PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time = 0;
    work->info->update_time = 0.0;
  }
  osqp_tic(work->timer); // Start timer
#endif /* ifdef PROFILING */

  // Replace l by the new vector
  prea_vec_copy(l_new, work->data->l, work->data->m);

  // Scaling
  if (work->settings->scaling) {
    vec_ew_prod(work->scaling->E, work->data->l, work->data->l, work->data->m);
  }

  // Check if lower bound is smaller than upper bound
  for (i = 0; i < work->data->m; i++) {
    if (work->data->l[i] > work->data->u[i]) {
#ifdef PRINTING
      c_eprint("upper bound must be greater than or equal to lower bound");
#endif /* ifdef PRINTING */
      return 1;
    }
  }

  // Reset solver information
  reset_info(work->info);

#if EMBEDDED != 1
  // Update rho_vec and refactor if constraints type changes
  exitflag = update_rho_vec(work);
#endif // EMBEDDED ! =1

#ifdef PROFILING
  work->info->update_time += osqp_toc(work->timer);
#endif /* ifdef PROFILING */

  return exitflag;
}

c_int osqp_update_upper_bound(OSQPWorkspace *work, const c_float *u_new) {
  c_int i, exitflag = 0;

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

#ifdef PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time = 0;
    work->info->update_time = 0.0;
  }
  osqp_tic(work->timer); // Start timer
#endif /* ifdef PROFILING */

  // Replace u by the new vector
  prea_vec_copy(u_new, work->data->u, work->data->m);

  // Scaling
  if (work->settings->scaling) {
    vec_ew_prod(work->scaling->E, work->data->u, work->data->u, work->data->m);
  }

  // Check if upper bound is greater than lower bound
  for (i = 0; i < work->data->m; i++) {
    if (work->data->u[i] < work->data->l[i]) {
#ifdef PRINTING
      c_eprint("lower bound must be lower than or equal to upper bound");
#endif /* ifdef PRINTING */
      return 1;
    }
  }

  // Reset solver information
  reset_info(work->info);

#if EMBEDDED != 1
  // Update rho_vec and refactor if constraints type changes
  exitflag = update_rho_vec(work);
#endif // EMBEDDED != 1

#ifdef PROFILING
  work->info->update_time += osqp_toc(work->timer);
#endif /* ifdef PROFILING */

  return exitflag;
}

c_int osqp_warm_start(OSQPWorkspace *work, const c_float *x, const c_float *y) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Update warm_start setting to true
  if (!work->settings->warm_start) work->settings->warm_start = 1;

  // Copy primal and dual variables into the iterates
  prea_vec_copy(x, work->x, work->data->n);
  prea_vec_copy(y, work->y, work->data->m);

  // Scale iterates
  if (work->settings->scaling) {
    vec_ew_prod(work->scaling->Dinv, work->x, work->x, work->data->n);
    vec_ew_prod(work->scaling->Einv, work->y, work->y, work->data->m);
    vec_mult_scalar(work->y, work->scaling->c, work->data->m);
  }

  // Compute Ax = z and store it in z
  mat_vec(work->data->A, work->x, work->z, 0);

  return 0;
}

c_int osqp_warm_start_x(OSQPWorkspace *work, const c_float *x) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Update warm_start setting to true
  if (!work->settings->warm_start) work->settings->warm_start = 1;

  // Copy primal variable into the iterate x
  prea_vec_copy(x, work->x, work->data->n);

  // Scale iterate
  if (work->settings->scaling) {
    vec_ew_prod(work->scaling->Dinv, work->x, work->x, work->data->n);
  }

  // Compute Ax = z and store it in z
  mat_vec(work->data->A, work->x, work->z, 0);

  return 0;
}

c_int osqp_warm_start_y(OSQPWorkspace *work, const c_float *y) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Update warm_start setting to true
  if (!work->settings->warm_start) work->settings->warm_start = 1;

  // Copy primal variable into the iterate y
  prea_vec_copy(y, work->y, work->data->m);

  // Scale iterate
  if (work->settings->scaling) {
    vec_ew_prod(work->scaling->Einv, work->y, work->y, work->data->m);
    vec_mult_scalar(work->y, work->scaling->c, work->data->m);
  }

  return 0;
}


#if EMBEDDED != 1

c_int osqp_update_P(OSQPWorkspace *work,
                    const c_float *Px_new,
                    const c_int   *Px_new_idx,
                    c_int          P_new_n) {
  c_int i;        // For indexing
  c_int exitflag; // Exit flag
  c_int nnzP;     // Number of nonzeros in P

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

#ifdef PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time = 0;
    work->info->update_time = 0.0;
  }
  osqp_tic(work->timer); // Start timer
#endif /* ifdef PROFILING */

  nnzP = work->data->P->p[work->data->P->n];

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

  if (work->settings->scaling) {
    // Unscale data
    unscale_data(work);
  }

  // Update P elements
  if (Px_new_idx) { // Change only Px_new_idx
    for (i = 0; i < P_new_n; i++) {
      work->data->P->x[Px_new_idx[i]] = Px_new[i];
    }
  }
  else // Change whole P
  {
    for (i = 0; i < nnzP; i++) {
      work->data->P->x[i] = Px_new[i];
    }
  }

  if (work->settings->scaling) {
    // Scale data
    scale_data(work);
  }

  // Update linear system structure with new data
  exitflag = work->linsys_solver->update_matrices(work->linsys_solver,
                                                  work->data->P,
                                                  work->data->A);

  // Reset solver information
  reset_info(work->info);

# ifdef PRINTING

  if (exitflag < 0) {
    c_eprint("new KKT matrix is not quasidefinite");
  }
# endif /* ifdef PRINTING */

#ifdef PROFILING
  work->info->update_time += osqp_toc(work->timer);
#endif /* ifdef PROFILING */

  return exitflag;
}


c_int osqp_update_A(OSQPWorkspace *work,
                    const c_float *Ax_new,
                    const c_int   *Ax_new_idx,
                    c_int          A_new_n) {
  c_int i;        // For indexing
  c_int exitflag; // Exit flag
  c_int nnzA;     // Number of nonzeros in A

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

#ifdef PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time = 0;
    work->info->update_time = 0.0;
  }
  osqp_tic(work->timer); // Start timer
#endif /* ifdef PROFILING */

  nnzA = work->data->A->p[work->data->A->n];

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

  if (work->settings->scaling) {
    // Unscale data
    unscale_data(work);
  }

  // Update A elements
  if (Ax_new_idx) { // Change only Ax_new_idx
    for (i = 0; i < A_new_n; i++) {
      work->data->A->x[Ax_new_idx[i]] = Ax_new[i];
    }
  }
  else { // Change whole A
    for (i = 0; i < nnzA; i++) {
      work->data->A->x[i] = Ax_new[i];
    }
  }

  if (work->settings->scaling) {
    // Scale data
    scale_data(work);
  }

  // Update linear system structure with new data
  exitflag = work->linsys_solver->update_matrices(work->linsys_solver,
                                                  work->data->P,
                                                  work->data->A);

  // Reset solver information
  reset_info(work->info);

# ifdef PRINTING

  if (exitflag < 0) {
    c_eprint("new KKT matrix is not quasidefinite");
  }
# endif /* ifdef PRINTING */

#ifdef PROFILING
  work->info->update_time += osqp_toc(work->timer);
#endif /* ifdef PROFILING */

  return exitflag;
}


c_int osqp_update_P_A(OSQPWorkspace *work,
                      const c_float *Px_new,
                      const c_int   *Px_new_idx,
                      c_int          P_new_n,
                      const c_float *Ax_new,
                      const c_int   *Ax_new_idx,
                      c_int          A_new_n) {
  c_int i;          // For indexing
  c_int exitflag;   // Exit flag
  c_int nnzP, nnzA; // Number of nonzeros in P and A

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

#ifdef PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time = 0;
    work->info->update_time = 0.0;
  }
  osqp_tic(work->timer); // Start timer
#endif /* ifdef PROFILING */

  nnzP = work->data->P->p[work->data->P->n];
  nnzA = work->data->A->p[work->data->A->n];


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

  if (work->settings->scaling) {
    // Unscale data
    unscale_data(work);
  }

  // Update P elements
  if (Px_new_idx) { // Change only Px_new_idx
    for (i = 0; i < P_new_n; i++) {
      work->data->P->x[Px_new_idx[i]] = Px_new[i];
    }
  }
  else // Change whole P
  {
    for (i = 0; i < nnzP; i++) {
      work->data->P->x[i] = Px_new[i];
    }
  }

  // Update A elements
  if (Ax_new_idx) { // Change only Ax_new_idx
    for (i = 0; i < A_new_n; i++) {
      work->data->A->x[Ax_new_idx[i]] = Ax_new[i];
    }
  }
  else { // Change whole A
    for (i = 0; i < nnzA; i++) {
      work->data->A->x[i] = Ax_new[i];
    }
  }

  if (work->settings->scaling) {
    // Scale data
    scale_data(work);
  }

  // Update linear system structure with new data
  exitflag = work->linsys_solver->update_matrices(work->linsys_solver,
                                                  work->data->P,
                                                  work->data->A);

  // Reset solver information
  reset_info(work->info);

# ifdef PRINTING

  if (exitflag < 0) {
    c_eprint("new KKT matrix is not quasidefinite");
  }
# endif /* ifdef PRINTING */

#ifdef PROFILING
  work->info->update_time += osqp_toc(work->timer);
#endif /* ifdef PROFILING */

  return exitflag;
}

c_int osqp_update_rho(OSQPWorkspace *work, c_float rho_new) {
  c_int exitflag, i;

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

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
      work->info->update_time = 0.0;
    }
    osqp_tic(work->timer); // Start timer
  }
#endif /* ifdef PROFILING */

  // Update rho in settings
  work->settings->rho = c_min(c_max(rho_new, RHO_MIN), RHO_MAX);

  // Update rho_vec and rho_inv_vec
  for (i = 0; i < work->data->m; i++) {
    if (work->constr_type[i] == 0) {
      // Inequalities
      work->rho_vec[i]     = work->settings->rho;
      work->rho_inv_vec[i] = 1. / work->settings->rho;
    }
    else if (work->constr_type[i] == 1) {
      // Equalities
      work->rho_vec[i]     = RHO_EQ_OVER_RHO_INEQ * work->settings->rho;
      work->rho_inv_vec[i] = 1. / work->rho_vec[i];
    }
  }

  // Update rho_vec in KKT matrix
  exitflag = work->linsys_solver->update_rho_vec(work->linsys_solver,
                                                 work->rho_vec);

#ifdef PROFILING
  if (work->rho_update_from_solve == 0)
    work->info->update_time += osqp_toc(work->timer);
#endif /* ifdef PROFILING */

  return exitflag;
}

#endif // EMBEDDED != 1

/****************************
* Update problem settings  *
****************************/
c_int osqp_update_max_iter(OSQPWorkspace *work, c_int max_iter_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that max_iter is positive
  if (max_iter_new <= 0) {
#ifdef PRINTING
    c_eprint("max_iter must be positive");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update max_iter
  work->settings->max_iter = max_iter_new;

  return 0;
}

c_int osqp_update_eps_abs(OSQPWorkspace *work, c_float eps_abs_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that eps_abs is positive
  if (eps_abs_new < 0.) {
#ifdef PRINTING
    c_eprint("eps_abs must be nonnegative");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update eps_abs
  work->settings->eps_abs = eps_abs_new;

  return 0;
}

c_int osqp_update_eps_rel(OSQPWorkspace *work, c_float eps_rel_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that eps_rel is positive
  if (eps_rel_new < 0.) {
#ifdef PRINTING
    c_eprint("eps_rel must be nonnegative");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update eps_rel
  work->settings->eps_rel = eps_rel_new;

  return 0;
}

c_int osqp_update_eps_prim_inf(OSQPWorkspace *work, c_float eps_prim_inf_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that eps_prim_inf is positive
  if (eps_prim_inf_new < 0.) {
#ifdef PRINTING
    c_eprint("eps_prim_inf must be nonnegative");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update eps_prim_inf
  work->settings->eps_prim_inf = eps_prim_inf_new;

  return 0;
}

c_int osqp_update_eps_dual_inf(OSQPWorkspace *work, c_float eps_dual_inf_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that eps_dual_inf is positive
  if (eps_dual_inf_new < 0.) {
#ifdef PRINTING
    c_eprint("eps_dual_inf must be nonnegative");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update eps_dual_inf
  work->settings->eps_dual_inf = eps_dual_inf_new;


  return 0;
}

c_int osqp_update_alpha(OSQPWorkspace *work, c_float alpha_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that alpha is between 0 and 2
  if ((alpha_new <= 0.) || (alpha_new >= 2.)) {
#ifdef PRINTING
    c_eprint("alpha must be between 0 and 2");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update alpha
  work->settings->alpha = alpha_new;

  return 0;
}

c_int osqp_update_warm_start(OSQPWorkspace *work, c_int warm_start_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that warm_start is either 0 or 1
  if ((warm_start_new != 0) && (warm_start_new != 1)) {
#ifdef PRINTING
    c_eprint("warm_start should be either 0 or 1");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update warm_start
  work->settings->warm_start = warm_start_new;

  return 0;
}

c_int osqp_update_scaled_termination(OSQPWorkspace *work, c_int scaled_termination_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that scaled_termination is either 0 or 1
  if ((scaled_termination_new != 0) && (scaled_termination_new != 1)) {
#ifdef PRINTING
    c_eprint("scaled_termination should be either 0 or 1");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update scaled_termination
  work->settings->scaled_termination = scaled_termination_new;

  return 0;
}

c_int osqp_update_check_termination(OSQPWorkspace *work, c_int check_termination_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that check_termination is nonnegative
  if (check_termination_new < 0) {
#ifdef PRINTING
    c_eprint("check_termination should be nonnegative");
#endif /* ifdef PRINTING */
    return 1;
  }

  // Update check_termination
  work->settings->check_termination = check_termination_new;

  return 0;
}

#ifndef EMBEDDED

c_int osqp_update_delta(OSQPWorkspace *work, c_float delta_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that delta is positive
  if (delta_new <= 0.) {
# ifdef PRINTING
    c_eprint("delta must be positive");
# endif /* ifdef PRINTING */
    return 1;
  }

  // Update delta
  work->settings->delta = delta_new;

  return 0;
}

c_int osqp_update_polish(OSQPWorkspace *work, c_int polish_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that polish is either 0 or 1
  if ((polish_new != 0) && (polish_new != 1)) {
# ifdef PRINTING
    c_eprint("polish should be either 0 or 1");
# endif /* ifdef PRINTING */
    return 1;
  }

  // Update polish
  work->settings->polish = polish_new;

# ifdef PROFILING

  // Reset polish time to zero
  work->info->polish_time = 0.0;
# endif /* ifdef PROFILING */

  return 0;
}

c_int osqp_update_polish_refine_iter(OSQPWorkspace *work, c_int polish_refine_iter_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that polish_refine_iter is nonnegative
  if (polish_refine_iter_new < 0) {
# ifdef PRINTING
    c_eprint("polish_refine_iter must be nonnegative");
# endif /* ifdef PRINTING */
    return 1;
  }

  // Update polish_refine_iter
  work->settings->polish_refine_iter = polish_refine_iter_new;

  return 0;
}

c_int osqp_update_verbose(OSQPWorkspace *work, c_int verbose_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that verbose is either 0 or 1
  if ((verbose_new != 0) && (verbose_new != 1)) {
# ifdef PRINTING
    c_eprint("verbose should be either 0 or 1");
# endif /* ifdef PRINTING */
    return 1;
  }

  // Update verbose
  work->settings->verbose = verbose_new;

  return 0;
}

#endif // EMBEDDED

#ifdef PROFILING

c_int osqp_update_time_limit(OSQPWorkspace *work, c_float time_limit_new) {

  // Check if workspace has been initialized
  if (!work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  // Check that time_limit is nonnegative
  if (time_limit_new < 0.) {
# ifdef PRINTING
    c_print("time_limit must be nonnegative\n");
# endif /* ifdef PRINTING */
    return 1;
  }

  // Update time_limit
  work->settings->time_limit = time_limit_new;

  return 0;
}
#endif /* ifdef PROFILING */
