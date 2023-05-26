#include "glob_opts.h"
#include "osqp.h"
#include "auxil.h"
#include "osqp_api_constants.h"
#include "util.h"
#include "scaling.h"
#include "error.h"
#include "version.h"
#include "lin_alg.h"
#include "printing.h"
#include "timing.h"


#ifdef OSQP_CODEGEN
  #include "codegen.h"
#endif

#ifndef OSQP_EMBEDDED_MODE
# include "polish.h"
#endif

#ifdef OSQP_ENABLE_DERIVATIVES
# include "derivative.h"
#endif

#ifdef OSQP_ENABLE_INTERRUPT
# include "interrupt.h"
#endif


/**********************
* Main API Functions *
**********************/
OSQPInt osqp_capabilities(void) {
  OSQPInt capabilities = 0;

  capabilities |= osqp_algebra_linsys_supported();

#if OSQP_EMBEDDED_MODE != 1
  capabilities |= OSQP_CAPABILITY_UPDATE_MATRICES;
#endif

#ifdef OSQP_CODEGEN
  capabilities |= OSQP_CAPABILITY_CODEGEN;
#endif

#ifdef OSQP_ENABLE_DERIVATIVES
    capabilities |= OSQP_CAPABILITY_DERIVATIVES;
#endif

  return capabilities;
}

const char* osqp_version(void) {
  return OSQP_VERSION;
}


const char* osqp_error_message(OSQPInt error_flag) {
  if( error_flag >= OSQP_LAST_ERROR_PLACE ) {
    return OSQP_ERROR_MESSAGE[OSQP_LAST_ERROR_PLACE-1];
  }


  return OSQP_ERROR_MESSAGE[error_flag-1];
}


void osqp_get_dimensions(OSQPSolver* solver,
                         OSQPInt*    m,
                         OSQPInt*    n) {

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


void osqp_set_default_codegen_defines(OSQPCodegenDefines* defines) {

  /* Avoid working with a null pointer */
  if (!defines)
    return;

  defines->embedded_mode      = 1;  /* Default to vector updates only */
  defines->float_type         = 0;  /* Default to double */
  defines->printing_enable    = 0;  /* Default to no printing */
  defines->profiling_enable   = 0;  /* Default to no timing */
  defines->interrupt_enable   = 0;  /* Default to no interrupts */
  defines->derivatives_enable = 0;  /* Default to no derivatives */
}


void osqp_set_default_settings(OSQPSettings* settings) {

  /* Avoid working with a null pointer */
  if (!settings)
    return;

  settings->device = 0;                                      /* device identifier */
  settings->linsys_solver  = osqp_algebra_default_linsys();  /* linear system solver */
  settings->verbose        = OSQP_VERBOSE;                   /* print output */
  settings->warm_starting  = OSQP_WARM_STARTING;             /* warm starting */
  settings->scaling        = OSQP_SCALING;                   /* heuristic problem scaling */
  settings->polishing      = OSQP_POLISHING;                 /* ADMM solution polish: 1 */

  settings->rho           = (OSQPFloat)OSQP_RHO;    /* ADMM step */
  settings->rho_is_vec    = OSQP_RHO_IS_VEC;        /* defines whether rho is scalar or vector*/
  settings->sigma         = (OSQPFloat)OSQP_SIGMA;  /* ADMM step */
  settings->alpha         = (OSQPFloat)OSQP_ALPHA;  /* relaxation parameter */

  settings->cg_max_iter      = OSQP_CG_MAX_ITER;             /* maximum number of CG iterations */
  settings->cg_tol_reduction = OSQP_CG_TOL_REDUCTION;        /* CG tolerance parameter */
  settings->cg_tol_fraction  = OSQP_CG_TOL_FRACTION;         /* CG tolerance parameter */
  settings->cg_precond       = OSQP_DIAGONAL_PRECONDITIONER; /* Preconditioner to use in CG */

  settings->adaptive_rho           = OSQP_ADAPTIVE_RHO;
  settings->adaptive_rho_interval  = OSQP_ADAPTIVE_RHO_INTERVAL;
  settings->adaptive_rho_fraction  = (OSQPFloat)OSQP_ADAPTIVE_RHO_FRACTION;
  settings->adaptive_rho_tolerance = (OSQPFloat)OSQP_ADAPTIVE_RHO_TOLERANCE;

  settings->max_iter           = OSQP_MAX_ITER;                 /* maximum number of ADMM iterations */
  settings->eps_abs            = (OSQPFloat)OSQP_EPS_ABS;       /* absolute convergence tolerance */
  settings->eps_rel            = (OSQPFloat)OSQP_EPS_REL;       /* relative convergence tolerance */
  settings->eps_prim_inf       = (OSQPFloat)OSQP_EPS_PRIM_INF;  /* primal infeasibility tolerance */
  settings->eps_dual_inf       = (OSQPFloat)OSQP_EPS_DUAL_INF;  /* dual infeasibility tolerance */
  settings->scaled_termination = OSQP_SCALED_TERMINATION;       /* evaluate scaled termination criteria */
  settings->check_termination  = OSQP_CHECK_TERMINATION;        /* interval for evaluating termination criteria */
  settings->time_limit         = OSQP_TIME_LIMIT;               /* stop the algorithm when time limit is reached */

  settings->delta              = OSQP_DELTA;                    /* regularization parameter for polishing */
  settings->polish_refine_iter = OSQP_POLISH_REFINE_ITER;       /* iterative refinement steps in polish */
}

#ifndef OSQP_EMBEDDED_MODE


OSQPInt osqp_setup(OSQPSolver**         solverp,
                   const OSQPCscMatrix* P,
                   const OSQPFloat*     q,
                   const OSQPCscMatrix* A,
                   const OSQPFloat*     l,
                   const OSQPFloat*     u,
                   OSQPInt              m,
                   OSQPInt              n,
                   const OSQPSettings*  settings) {

  OSQPInt exitflag;

  OSQPSolver*    solver;
  OSQPWorkspace* work;

  // Validate data
  if (validate_data(P,q,A,l,u,m,n)) return osqp_error(OSQP_DATA_VALIDATION_ERROR);

  // Validate settings
  if (validate_settings(settings, 1)) return osqp_error(OSQP_SETTINGS_VALIDATION_ERROR);

  // Allocate empty solver
  solver = c_calloc(1, sizeof(OSQPSolver));
  if (!(solver)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  *solverp = solver;

  // Allocate empty workspace
  work   = c_calloc(1, sizeof(OSQPWorkspace));
  if (!(work)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  solver->work = work;

  // Allocate empty info struct
  solver->info = c_calloc(1, sizeof(OSQPInfo));
  if (!(solver->info)) return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Start and allocate directly timer
# ifdef OSQP_ENABLE_PROFILING
  work->timer = OSQPTimer_new();
  if (!(work->timer)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  osqp_tic(work->timer);
# endif /* ifdef OSQP_ENABLE_PROFILING */

  // Initialize algebra libraries
  exitflag = osqp_algebra_init_libs(settings->device);
  if (exitflag) return osqp_error(OSQP_ALGEBRA_LOAD_ERROR);

  // Copy problem data into workspace
  work->data = c_calloc(1, sizeof(OSQPData));
  if (!(work->data)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  work->data->m = m;
  work->data->n = n;

  // objective function
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

  if (settings->rho_is_vec) {
    // Vectorized rho parameter
    work->rho_vec     = OSQPVectorf_malloc(m);
    work->rho_inv_vec = OSQPVectorf_malloc(m);
    if (!(work->rho_vec) || !(work->rho_inv_vec))
      return osqp_error(OSQP_MEM_ALLOC_ERROR);

    // Type of constraints
    work->constr_type = OSQPVectori_calloc(m);
    if (!(work->constr_type)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  }
  else {
    work->rho_vec     = OSQP_NULL;
    work->rho_inv_vec = OSQP_NULL;
  }

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

  if (settings->rho_is_vec) {
    // Set type of constraints.  Ignore return value
    // because we will definitely factor KKT.
    set_rho_vec(solver);
  }
  else {
    solver->settings->rho = c_min(c_max(settings->rho, OSQP_RHO_MIN), OSQP_RHO_MAX);
    work->rho_inv = 1. / settings->rho;
  }

  // Initialize linear system solver structure
  exitflag = osqp_algebra_init_linsys_solver(&(work->linsys_solver), work->data->P, work->data->A,
                                             work->rho_vec, solver->settings,
                                             &work->scaled_prim_res, &work->scaled_dual_res, 0);

  if (exitflag == OSQP_NONCVX_ERROR) {
    update_status(solver->info, OSQP_NON_CVX);
    return osqp_error(exitflag);
  }
  else if (exitflag) {
    return osqp_error(exitflag);
  }

  // Initialize variables x, y, z to 0
  osqp_cold_start(solver);

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
  solver->solution->x             = c_calloc(1, n * sizeof(OSQPFloat));
  solver->solution->y             = c_calloc(1, m * sizeof(OSQPFloat));
  solver->solution->prim_inf_cert = c_calloc(1, m * sizeof(OSQPFloat));
  solver->solution->dual_inf_cert = c_calloc(1, n * sizeof(OSQPFloat));
  if ( !(solver->solution->x) || !(solver->solution->dual_inf_cert) )
    return osqp_error(OSQP_MEM_ALLOC_ERROR);
  if ( m && (!(solver->solution->y) || !(solver->solution->prim_inf_cert)) )
    return osqp_error(OSQP_MEM_ALLOC_ERROR);

  // Initialize information
  solver->info->status_polish = OSQP_POLISH_NOT_PERFORMED; // Polishing not performed
  update_status(solver->info, OSQP_UNSOLVED);
# ifdef OSQP_ENABLE_PROFILING
  solver->info->solve_time  = 0.0;                   // Solve time to zero
  solver->info->update_time = 0.0;                   // Update time to zero
  solver->info->polish_time = 0.0;                   // Polish time to zero
  solver->info->run_time    = 0.0;                   // Total run time to zero
  solver->info->setup_time  = osqp_toc(work->timer); // Update timer information

  work->first_run         = 1;
  work->clear_update_time = 0;
  work->rho_update_from_solve = 0;
# endif /* ifdef OSQP_ENABLE_PROFILING */
  solver->info->rho_updates  = 0;                      // Rho updates set to 0
  solver->info->rho_estimate = solver->settings->rho;  // Best rho estimate
  solver->info->obj_val      = OSQP_INFTY;
  solver->info->prim_res     = OSQP_INFTY;
  solver->info->dual_res     = OSQP_INFTY;

  // Print header
# ifdef OSQP_ENABLE_PRINTING
  if (solver->settings->verbose) print_setup_header(solver);
  work->summary_printed = 0; // Initialize last summary  to not printed
# endif /* ifdef OSQP_ENABLE_PRINTING */


  // If adaptive rho and automatic interval, but profiling disabled, we need to
  // set the interval to a default value
# ifndef OSQP_ENABLE_PROFILING
  if (solver->settings->adaptive_rho && !solver->settings->adaptive_rho_interval) {
    if (solver->settings->check_termination) {
      // If check_termination is enabled, we set it to a multiple of the check
      // termination interval
      solver->settings->adaptive_rho_interval = OSQP_ADAPTIVE_RHO_MULTIPLE_TERMINATION *
                                              solver->settings->check_termination;
    } else {
      // If check_termination is disabled we set it to a predefined fix number
      solver->settings->adaptive_rho_interval = OSQP_ADAPTIVE_RHO_FIXED;
    }
  }
# endif /* ifndef OSQP_ENABLE_PROFILING */

# ifdef OSQP_ENABLE_DERIVATIVES
  work->derivative_data = c_calloc(1, sizeof(OSQPDerivativeData));
  if (!(work->derivative_data)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
  work->derivative_data->y_u = OSQPVectorf_malloc(m);
  work->derivative_data->y_l = OSQPVectorf_malloc(m);
  work->derivative_data->ryl = OSQPVectorf_malloc(m);
  work->derivative_data->ryu = OSQPVectorf_malloc(m);
  work->derivative_data->rhs = OSQPVectorf_malloc(2 * (n + 2*m));
  if (!(work->derivative_data->y_u) || !(work->derivative_data->y_l) ||
    !(work->derivative_data->ryl) || !(work->derivative_data->ryu))
    return osqp_error(OSQP_MEM_ALLOC_ERROR);
# endif /* ifdef OSQP_ENABLE_DERIVATIVES */

  // Return exit flag
  return 0;
}

#endif /* ifndef OSQP_EMBEDDED_MODE */


OSQPInt osqp_solve(OSQPSolver *solver) {

  OSQPInt exitflag;
  OSQPInt iter, max_iter;
  OSQPInt compute_obj;           // boolean: compute objective function in the loop or not
  OSQPInt can_check_termination; // boolean: check termination or not
  OSQPWorkspace* work;

#ifdef OSQP_ENABLE_PROFILING
  OSQPFloat temp_run_time;       // Temporary variable to store current run time
#endif /* ifdef OSQP_ENABLE_PROFILING */

#ifdef OSQP_ENABLE_PRINTING
  OSQPInt can_print;             // Boolean whether you can print
#endif /* ifdef OSQP_ENABLE_PRINTING */

  // Check if solver has been initialized
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);
  work = solver->work;

#ifdef OSQP_ENABLE_PROFILING
  if (work->clear_update_time == 1)
    solver->info->update_time = 0.0;
  work->rho_update_from_solve = 1;
#endif /* ifdef OSQP_ENABLE_PROFILING */

  // Initialize variables
  exitflag              = 0;
  can_check_termination = 0;
#ifdef OSQP_ENABLE_PRINTING
  can_print = solver->settings->verbose;
  // Compute objective function only if verbose is on
  compute_obj = solver->settings->verbose;
#else /* ifdef OSQP_ENABLE_PRINTING */
  compute_obj = 0;
#endif /* ifdef OSQP_ENABLE_PRINTING */

#ifdef OSQP_ENABLE_PROFILING
  osqp_tic(work->timer); // Start timer
#endif /* ifdef OSQP_ENABLE_PROFILING */


#ifdef OSQP_ENABLE_PRINTING
  if (solver->settings->verbose) {
    // Print Header for every column
    print_header();
  }
#endif /* ifdef OSQP_ENABLE_PRINTING */

#ifdef OSQP_ENABLE_INTERRUPT

  // initialize Ctrl-C support
  osqp_start_interrupt_listener();
#endif /* ifdef OSQP_ENABLE_INTERRUPT */

  // Initialize variables (cold start or warm start depending on settings)
  // If not warm start -> set x, z, y to zero
  if (!solver->settings->warm_starting) osqp_cold_start(solver);

  // Main ADMM algorithm

  max_iter = solver->settings->max_iter;
  for (iter = 1; iter <= max_iter; iter++) {

    // Update x_prev, z_prev (preallocated, no malloc)
    swap_vectors(&(work->x), &(work->x_prev));
    swap_vectors(&(work->z), &(work->z_prev));

    /* ADMM STEPS */
    /* Compute \tilde{x}^{k+1}, \tilde{z}^{k+1} */
    update_xz_tilde(solver, iter);

    /* Compute x^{k+1} */
    update_x(solver);

    /* Compute z^{k+1} */
    update_z(solver);

    /* Compute y^{k+1} */
    update_y(solver);

    /* End of ADMM Steps */

#ifdef OSQP_ENABLE_INTERRUPT

    // Check the interrupt signal
    if (osqp_is_interrupted()) {
      update_status(solver->info, OSQP_SIGINT);
      c_print("Solver interrupted\n");
      exitflag = 1;
      goto exit;
    }
#endif /* ifdef OSQP_ENABLE_INTERRUPT */

#ifdef OSQP_ENABLE_PROFILING

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
# ifdef OSQP_ENABLE_PRINTING

      if (solver->settings->verbose) c_print("run time limit reached\n");
      can_print = 0;  // Not printing at this iteration
# endif /* ifdef OSQP_ENABLE_PRINTING */
      break;
    }
#endif /* ifdef OSQP_ENABLE_PROFILING */


    // Can we check for termination ?
    can_check_termination = solver->settings->check_termination &&
                            (iter % solver->settings->check_termination == 0);

#ifdef OSQP_ENABLE_PRINTING

    // Can we print ?
    can_print = solver->settings->verbose &&
                ((iter % OSQP_PRINT_INTERVAL == 0) || (iter == 1));

    // NB: We always update info in the first iteration because indirect solvers
    //     use residual values to compute required accuracy of their solution.
    if (can_check_termination || can_print || iter == 1) { // Update status in either of
                                                           // these cases
      // Update information
      update_info(solver, iter, compute_obj, 0);

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
#else /* ifdef OSQP_ENABLE_PRINTING */

    if (can_check_termination) {
      // Update information and compute also objective value
      update_info(solver, iter, compute_obj, 0);

      // Check algorithm termination
      if (check_termination(solver, 0)) {
        // Terminate algorithm
        break;
      }
    }
#endif /* ifdef OSQP_ENABLE_PRINTING */


#if OSQP_EMBEDDED_MODE != 1
# ifdef OSQP_ENABLE_PROFILING

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
          (OSQPInt)c_roundmultiple(iter, solver->settings->check_termination);
         }
         else {
          // If check_termination is disabled, we round the number of iterations
          // between
          // updates to the closest multiple of the default check_termination
          // interval.
          solver->settings->adaptive_rho_interval = (OSQPInt)c_roundmultiple(iter, OSQP_CHECK_TERMINATION);
        }

        // Make sure the interval is not 0 and at least check_termination times
          solver->settings->adaptive_rho_interval = c_max(
          solver->settings->adaptive_rho_interval,
          solver->settings->check_termination);
      } // If time condition is met
    }   // If adaptive rho enabled and interval set to auto®
# else // OSQP_ENABLE_PROFILING
    if (solver->settings->adaptive_rho && !solver->settings->adaptive_rho_interval) {
      // Set adaptive_rho_interval to constant value
      if (solver->settings->check_termination) {
        // If check_termination is enabled, we set it to a multiple of the check
        // termination interval
        solver->settings->adaptive_rho_interval = OSQP_ADAPTIVE_RHO_MULTIPLE_TERMINATION *
                                                solver->settings->check_termination;
      } else {
        // If check_termination is disabled we set it to a predefined fix number
        solver->settings->adaptive_rho_interval = OSQP_ADAPTIVE_RHO_FIXED;
      }
    }
# endif // OSQP_ENABLE_PROFILING

    // Adapt rho
    if (solver->settings->adaptive_rho &&
        solver->settings->adaptive_rho_interval &&
        (iter % solver->settings->adaptive_rho_interval == 0)) {
      // Update info with the residuals if it hasn't been done before
# ifdef OSQP_ENABLE_PRINTING

      if (!can_check_termination && !can_print) {
        // Information has not been computed neither for termination or printing
        // reasons
        update_info(solver, iter, compute_obj, 0);
      }
# else /* ifdef OSQP_ENABLE_PRINTING */

      if (!can_check_termination) {
        // Information has not been computed before for termination check
        update_info(solver, iter, compute_obj, 0);
      }
# endif /* ifdef OSQP_ENABLE_PRINTING */

      // Actually update rho
      if (adapt_rho(solver)) {
        c_eprint("Failed rho update");
        exitflag = 1;
        goto exit;
      }
    }
#endif // OSQP_EMBEDDED_MODE != 1

  }        // End of ADMM for loop


  // Update information and check termination condition if it hasn't been done
  // during last iteration (max_iter reached or check_termination disabled)
  if (!can_check_termination) {
    /* Update information */
#ifdef OSQP_ENABLE_PRINTING

    if (!can_print) {
      // Update info only if it hasn't been updated before for printing
      // reasons
      update_info(solver, iter - 1, compute_obj, 0);
    }
#else /* ifdef OSQP_ENABLE_PRINTING */

    // If no printing is enabled, update info directly
    update_info(solver, iter - 1, compute_obj, 0);
#endif /* ifdef OSQP_ENABLE_PRINTING */

#ifdef OSQP_ENABLE_PRINTING

    /* Print summary */
    if (solver->settings->verbose && !work->summary_printed) print_summary(solver);
#endif /* ifdef OSQP_ENABLE_PRINTING */

    /* Check whether a termination criterion is triggered */
    check_termination(solver, 0);

  }

  // Compute objective value in case it was not
  // computed during the iterations
  if (!compute_obj && has_solution(solver->info)){
    solver->info->obj_val = compute_obj_val(solver, work->x);
  }


#ifdef OSQP_ENABLE_PRINTING
  /* Print summary for last iteration */
  if (solver->settings->verbose && !work->summary_printed) {
    print_summary(solver);
  }
#endif /* ifdef OSQP_ENABLE_PRINTING */

  /* if max iterations reached, change status accordingly */
  if (solver->info->status_val == OSQP_UNSOLVED) {
    if (!check_termination(solver, 1)) { // Try to check for approximate
      update_status(solver->info, OSQP_MAX_ITER_REACHED);
    }
  }

#ifdef OSQP_ENABLE_PROFILING
  /* if time-limit reached check termination and update status accordingly */
 if (solver->info->status_val == OSQP_TIME_LIMIT_REACHED) {
    if (!check_termination(solver, 1)) { // Try for approximate solutions
      update_status(solver->info, OSQP_TIME_LIMIT_REACHED); /* Change update status back to OSQP_TIME_LIMIT_REACHED */
    }
  }
#endif /* ifdef OSQP_ENABLE_PROFILING */


#if OSQP_EMBEDDED_MODE != 1
  /* Update rho estimate */
  solver->info->rho_estimate = compute_rho_estimate(solver);
#endif /* if OSQP_EMBEDDED_MODE != 1 */

  /* Update solve time */
#ifdef OSQP_ENABLE_PROFILING
  solver->info->solve_time = osqp_toc(work->timer);
#endif /* ifdef OSQP_ENABLE_PROFILING */


#ifndef OSQP_EMBEDDED_MODE
  // Polish the obtained solution
  if (solver->settings->polishing && (solver->info->status_val == OSQP_SOLVED))
    polish(solver);
#endif /* ifndef OSQP_EMBEDDED_MODE */

#ifdef OSQP_ENABLE_PROFILING
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
#endif /* ifdef OSQP_ENABLE_PROFILING */

#ifdef OSQP_ENABLE_PRINTING
  /* Print final footer */
  if (solver->settings->verbose) print_footer(solver->info, solver->settings->polishing);
#endif /* ifdef OSQP_ENABLE_PRINTING */

  // Store solution
  store_solution(solver);


// Define exit flag for quitting function
#if defined(OSQP_ENABLE_PROFILING) || defined(OSQP_ENABLE_INTERRUPT) || OSQP_EMBEDDED_MODE != 1
exit:
#endif /* if defined(OSQP_ENABLE_PROFILING) || defined(OSQP_ENABLE_INTERRUPT) || OSQP_EMBEDDED_MODE != 1 */

#ifdef OSQP_ENABLE_INTERRUPT
  // Restore previous signal handler
  osqp_end_interrupt_listener();
#endif /* ifdef OSQP_ENABLE_INTERRUPT */

  return exitflag;
}


#ifndef OSQP_EMBEDDED_MODE

OSQPInt osqp_cleanup(OSQPSolver* solver) {

  OSQPInt exitflag = 0;
  OSQPWorkspace* work;

  if(!solver) return 0;   //exit on null

  work = solver->work;

  if (work) { // If workspace has been allocated
    // Free algebra library handlers
    osqp_algebra_free_libs();

    // Free Data
    if (work->data) {
      OSQPMatrix_free(work->data->P);
      OSQPMatrix_free(work->data->A);
      OSQPVectorf_free(work->data->q);
      OSQPVectorf_free(work->data->l);
      OSQPVectorf_free(work->data->u);
      c_free(work->data);
    }

    // Free scaling variables
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

#ifndef OSQP_EMBEDDED_MODE
    // Free active constraints structure
    if (work->pol) {
      OSQPVectori_free(work->pol->active_flags);
      OSQPVectorf_free(work->pol->x);
      OSQPVectorf_free(work->pol->z);
      OSQPVectorf_free(work->pol->y);
      c_free(work->pol);
    }
#endif /* ifndef OSQP_EMBEDDED_MODE */

    // Free other Variables
    OSQPVectorf_free(work->rho_vec);
    OSQPVectorf_free(work->rho_inv_vec);
#if OSQP_EMBEDDED_MODE != 1
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

# ifdef OSQP_ENABLE_PROFILING
    // Free timer
    if (work->timer) OSQPTimer_free(work->timer);
# endif /* ifdef OSQP_ENABLE_PROFILING */

# ifdef OSQP_ENABLE_DERIVATIVES
      if (work->derivative_data){
          if (work->derivative_data->y_l) OSQPVectorf_free(work->derivative_data->y_l);
          if (work->derivative_data->y_u) OSQPVectorf_free(work->derivative_data->y_u);
          if (work->derivative_data->ryl) OSQPVectorf_free(work->derivative_data->ryl);
          if (work->derivative_data->ryu) OSQPVectorf_free(work->derivative_data->ryu);
          if (work->derivative_data->rhs) OSQPVectorf_free(work->derivative_data->rhs);
          c_free(work->derivative_data);
      }
#endif /* ifdef OSQP_ENABLE_SCALING */

    // Free work
    c_free(work);
  }

  // Free solver
  c_free(solver);

  return exitflag;
}

#endif /* ifndef OSQP_EMBEDDED_MODE */



/************************
* Update problem data  *
************************/

OSQPInt osqp_update_data_vec(OSQPSolver*      solver,
                             const OSQPFloat* q_new,
                             const OSQPFloat* l_new,
                             const OSQPFloat* u_new) {

  OSQPInt exitflag = 0;

  OSQPVectorf*   l_tmp;
  OSQPVectorf*   u_tmp;
  OSQPWorkspace* work;

  /* Check if workspace has been initialized */
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);
  work = solver->work;

#ifdef OSQP_ENABLE_PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time = 0;
    solver->info->update_time = 0.0;
  }
  /* Start timer */
  osqp_tic(work->timer);
#endif /* ifdef OSQP_ENABLE_PROFILING */

  /* Update constraint bounds */
  if (l_new || u_new) {
    /* Use z_prev and delta_y to store l_new and u_new */
    l_tmp = work->z_prev;
    u_tmp = work->delta_y;

    /* Copy l_new and u_new to l_tmp and u_tmp */
    if (l_new) OSQPVectorf_from_raw(l_tmp, l_new);
    if (u_new) OSQPVectorf_from_raw(u_tmp, u_new);

    if (solver->settings->scaling) {
      if (l_new) OSQPVectorf_ew_prod(l_tmp, l_tmp, work->scaling->E);
      if (u_new) OSQPVectorf_ew_prod(u_tmp, u_tmp, work->scaling->E);
    }

      /* Check if upper bound is greater than lower bound */
      if (l_new && u_new) exitflag = !OSQPVectorf_all_leq(l_tmp, u_tmp);
      else if (l_new)     exitflag = !OSQPVectorf_all_leq(l_tmp, work->data->u);
      else                exitflag = !OSQPVectorf_all_leq(work->data->l, u_tmp);
      if (exitflag) return osqp_error(OSQP_DATA_VALIDATION_ERROR);

      /* Swap vectors.
       * NB: Use work->z_prev & delta_y rather than l_tmp & u_tmp */
      if (l_new) swap_vectors(&work->z_prev,  &work->data->l);
      if (u_new) swap_vectors(&work->delta_y, &work->data->u);

#if OSQP_EMBEDDED_MODE != 1
      /* Update rho_vec and refactor if constraints type changes */
      if (solver->settings->rho_is_vec) exitflag = update_rho_vec(solver);
#endif /* #if OSQP_EMBEDDED_MODE != 1 */
  }

  /* Update linear cost vector */
  if (q_new) {
    OSQPVectorf_from_raw(work->data->q, q_new);
    if (solver->settings->scaling) {
      OSQPVectorf_ew_prod(work->data->q, work->data->q, work->scaling->D);
      OSQPVectorf_mult_scalar(work->data->q, work->scaling->c);
    }
  }

  /* Reset solver information */
  reset_info(solver->info);

#ifdef OSQP_ENABLE_PROFILING
  solver->info->update_time += osqp_toc(work->timer);
#endif /* ifdef OSQP_ENABLE_PROFILING */

  return exitflag;
}


OSQPInt osqp_warm_start(OSQPSolver*      solver,
                        const OSQPFloat* x,
                        const OSQPFloat* y) {

  OSQPWorkspace* work;

  /* Check if workspace has been initialized */
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);
  work = solver->work;

  /* Update warm_start setting to true */
  if (!solver->settings->warm_starting) solver->settings->warm_starting = 1;

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

  /* Warm start the linear system solver */
  work->linsys_solver->warm_start(work->linsys_solver, work->x);

  return 0;
}


void osqp_cold_start(OSQPSolver *solver) {
  OSQPWorkspace *work = solver->work;
  OSQPVectorf_set_scalar(work->x, 0.);
  OSQPVectorf_set_scalar(work->z, 0.);
  OSQPVectorf_set_scalar(work->y, 0.);

  /* Cold start the linear system solver */
  work->linsys_solver->warm_start(work->linsys_solver, work->x);
}


#if OSQP_EMBEDDED_MODE != 1

OSQPInt osqp_update_data_mat(OSQPSolver*      solver,
                             const OSQPFloat* Px_new,
                             const OSQPInt*   Px_new_idx,
                             OSQPInt          P_new_n,
                             const OSQPFloat* Ax_new,
                             const OSQPInt*   Ax_new_idx,
                             OSQPInt          A_new_n) {

  OSQPInt exitflag;   // Exit flag
  OSQPInt nnzP, nnzA; // Number of nonzeros in P and A
  OSQPWorkspace *work;

  // Check if workspace has been initialized
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);
  work = solver->work;

#ifdef OSQP_ENABLE_PROFILING
  if (work->clear_update_time == 1) {
    work->clear_update_time = 0;
    solver->info->update_time = 0.0;
  }
  osqp_tic(work->timer); // Start timer
#endif /* ifdef OSQP_ENABLE_PROFILING */

  nnzP = OSQPMatrix_get_nz(work->data->P);
  nnzA = OSQPMatrix_get_nz(work->data->A);


  // Check if the number of elements to update is valid
  if (P_new_n > nnzP || P_new_n < 0) {
    c_eprint("new number of elements (%i) out of bounds for P (%i max)",
             (int)P_new_n, (int)nnzP);
    return 1;
  }
  //indexing is required if the whole P is not updated
  if(Px_new_idx == OSQP_NULL && P_new_n != 0 && P_new_n != nnzP){
        c_eprint("index vector is required for partial updates of P");
        return 1;
  }

  // Check if the number of elements to update is valid
  if (A_new_n > nnzA || A_new_n < 0) {
    c_eprint("new number of elements (%i) out of bounds for A (%i max)",
             (int)A_new_n,
             (int)nnzA);
    return 2;
  }
  //indexing is required if the whole A is not updated
  if(Ax_new_idx == OSQP_NULL && A_new_n != 0 && A_new_n != nnzA){
    c_eprint("index vector is required for partial updates of A");
    return 2;
  }

  if (solver->settings->scaling) unscale_data(solver);

  if (Px_new){
    OSQPMatrix_update_values(work->data->P, Px_new, Px_new_idx, P_new_n);
  }
  if (Ax_new){
    OSQPMatrix_update_values(work->data->A, Ax_new, Ax_new_idx, A_new_n);
  }

  if (solver->settings->scaling) scale_data(solver);

  // Update linear system structure with new data.
  // If there is scaling, then a full update is needed.
  if(solver->settings->scaling){
    exitflag = work->linsys_solver->update_matrices(
                  work->linsys_solver,
                  work->data->P, OSQP_NULL, nnzP,
                  work->data->A, OSQP_NULL, nnzA);
  }
  else{
    exitflag = work->linsys_solver->update_matrices(
                  work->linsys_solver,
                  work->data->P, Px_new_idx, P_new_n,
                  work->data->A, Ax_new_idx, A_new_n);
  }


  // Reset solver information
  reset_info(solver->info);

  if (exitflag != 0){c_eprint("new KKT matrix is not quasidefinite");}

#ifdef OSQP_ENABLE_PROFILING
  solver->info->update_time += osqp_toc(work->timer);
#endif /* ifdef OSQP_ENABLE_PROFILING */

  return exitflag;
}


OSQPInt osqp_update_rho(OSQPSolver* solver,
                        OSQPFloat     rho_new) {

    OSQPInt exitflag;
    OSQPWorkspace *work;

    // Check if workspace has been initialized
    if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);
    work = solver->work;

  // Check value of rho
  if (rho_new <= 0) {
    c_eprint("rho must be positive");
    return 1;
  }

#ifdef OSQP_ENABLE_PROFILING
  if (work->rho_update_from_solve == 0) {
    if (work->clear_update_time == 1) {
      work->clear_update_time = 0;
      solver->info->update_time = 0.0;
    }
    osqp_tic(work->timer); // Start timer
  }
#endif /* ifdef OSQP_ENABLE_PROFILING */

  // Update rho in settings
  solver->settings->rho = c_min(c_max(rho_new, OSQP_RHO_MIN), OSQP_RHO_MAX);

  if (solver->settings->rho_is_vec) {
    // Update rho_vec and rho_inv_vec
    OSQPVectorf_set_scalar_conditional(work->rho_vec,
                                       work->constr_type,
                                       OSQP_RHO_MIN,                                     //constr == -1
                                       solver->settings->rho,                            //constr == 0
                                       OSQP_RHO_EQ_OVER_RHO_INEQ*solver->settings->rho); //constr == 1

    OSQPVectorf_ew_reciprocal(work->rho_inv_vec, work->rho_vec);
  }
  else {
    work->rho_inv = 1. / solver->settings->rho;
  }

  // Update rho_vec in KKT matrix
  exitflag = work->linsys_solver->update_rho_vec(work->linsys_solver, work->rho_vec, solver->settings->rho);

#ifdef OSQP_ENABLE_PROFILING
  if (work->rho_update_from_solve == 0)
    solver->info->update_time += osqp_toc(work->timer);
#endif /* ifdef OSQP_ENABLE_PROFILING */

  return exitflag;
}

#endif // OSQP_EMBEDDED_MODE != 1



/****************************
* Update problem settings  *
****************************/

OSQPInt osqp_update_settings(OSQPSolver*         solver,
                             const OSQPSettings* new_settings) {

  OSQPSettings* settings = solver->settings;

  /* Check if workspace has been initialized */
  if (!solver || !solver->work) return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);

  /* Validate settings */
  if (validate_settings(new_settings, 0)) return osqp_error(OSQP_SETTINGS_VALIDATION_ERROR);

  /* Update settings */
  // linsys_solver ignored
  settings->verbose       = new_settings->verbose;
  settings->warm_starting = new_settings->warm_starting;
  // scaling ignored
  settings->polishing     = new_settings->polishing;

  // rho        ignored
  // rho_is_vec ignored
  // sigma      ignored
  settings->alpha = new_settings->alpha;

  settings->cg_max_iter      = new_settings->cg_max_iter;
  settings->cg_tol_reduction = new_settings->cg_tol_reduction;
  settings->cg_tol_fraction  = new_settings->cg_tol_fraction;
  settings->cg_precond       = new_settings->cg_precond;

  // adaptive_rho           ignored
  // adaptive_rho_interval  ignored
  // adaptive_rho_fraction  ignored
  // adaptive_rho_tolerance ignored

  settings->max_iter           = new_settings->max_iter;
  settings->eps_abs            = new_settings->eps_abs;
  settings->eps_rel            = new_settings->eps_rel;
  settings->eps_prim_inf       = new_settings->eps_prim_inf;
  settings->eps_dual_inf       = new_settings->eps_dual_inf;
  settings->scaled_termination = new_settings->scaled_termination;
  settings->check_termination  = new_settings->check_termination;
  settings->time_limit         = new_settings->time_limit;

  settings->delta              = new_settings->delta;
  settings->polish_refine_iter = new_settings->polish_refine_iter;

  /* Update settings in the linear system solver */
  solver->work->linsys_solver->update_settings(solver->work->linsys_solver, settings);

  return 0;
}


/**********
* Codegen
**********/

OSQPInt osqp_codegen(OSQPSolver*         solver,
                     const char*         output_dir,
                     const char*         file_prefix,
                     OSQPCodegenDefines* defines){

  OSQPInt exitflag = 0;

#ifdef OSQP_CODEGEN
  if (!solver || !solver->work || !solver->settings || !solver->info) {
    return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);
  }
  /* Don't allow codegen for a non-convex problem. */
  else if (solver->info->status_val == OSQP_NON_CVX) {
    return osqp_error(OSQP_NONCVX_ERROR);
  }
  /* Test after non-convex error to ensure we throw a useful error code*/
  else if (!solver->work->data || !solver->work->linsys_solver) {
    return osqp_error(OSQP_WORKSPACE_NOT_INIT_ERROR);
  }
  else if (!defines || (defines->embedded_mode != 1    && defines->embedded_mode != 2)
                    || (defines->float_type != 0       && defines->float_type != 1)
                    || (defines->printing_enable != 0  && defines->printing_enable != 1)
                    || (defines->profiling_enable != 0 && defines->profiling_enable != 1)
                    || (defines->interrupt_enable != 0 && defines->interrupt_enable != 1)
                    || (defines->derivatives_enable != 0 && defines->derivatives_enable != 1)) {
    return osqp_error(OSQP_CODEGEN_DEFINES_ERROR);
  }

  exitflag = codegen_inc(solver, output_dir, file_prefix);
  if (!exitflag) exitflag = codegen_src(solver, output_dir, file_prefix, defines->embedded_mode);
  if (!exitflag) exitflag = codegen_example(output_dir, file_prefix);
  if (!exitflag) exitflag = codegen_defines(output_dir, defines);
#else
  exitflag = OSQP_FUNC_NOT_IMPLEMENTED;
#endif /* ifdef OSQP_CODEGEN */

  return exitflag;
}



/****************************
* User API Helper functions
****************************/

void csc_set_data(OSQPCscMatrix* M,
                  OSQPInt        m,
                  OSQPInt        n,
                  OSQPInt        nzmax,
                  OSQPFloat*     x,
                  OSQPInt*       i,
                  OSQPInt*       p) {
  M->m     = m;
  M->n     = n;
  M->nz   = -1;
  M->nzmax = nzmax;
  M->x     = x;
  M->i     = i;
  M->p     = p;
}

/****************************
* Derivative functions
****************************/
OSQPInt osqp_adjoint_derivative_compute(OSQPSolver* solver,
                                        OSQPFloat*  dx,
                                        OSQPFloat*  dy_l,
                                        OSQPFloat*  dy_u) {
  OSQPInt status = 0;

#ifdef OSQP_ENABLE_DERIVATIVES
  status = adjoint_derivative_compute(solver, dx, dy_l, dy_u);
#else
  status = OSQP_FUNC_NOT_IMPLEMENTED;
#endif

  return status;
}

OSQPInt osqp_adjoint_derivative_get_mat(OSQPSolver*    solver,
                                        OSQPCscMatrix* dP,
                                        OSQPCscMatrix* dA) {
  OSQPInt status = 0;

#ifdef OSQP_ENABLE_DERIVATIVES
  status = adjoint_derivative_get_mat(solver, dP, dA);
#else
  status = OSQP_FUNC_NOT_IMPLEMENTED;
#endif

  return status;
}

OSQPInt osqp_adjoint_derivative_get_vec(OSQPSolver* solver,
                                        OSQPFloat*  dq,
                                        OSQPFloat*  dl,
                                        OSQPFloat*  du) {
  OSQPInt status = 0;

#ifdef OSQP_ENABLE_DERIVATIVES
  status = adjoint_derivative_get_vec(solver, dq, dl, du);
#else
  status = OSQP_FUNC_NOT_IMPLEMENTED;
#endif

  return status;
}
