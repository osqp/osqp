#include "osqp.h"
#include "util.h"
#include "algebra_vector.h"
#include "version.h"
#include "printing.h"
#include "lin_alg.h"

/************************************
* Printing Constants to set Layout *
************************************/
#ifdef OSQP_ENABLE_PRINTING
# define HEADER_LINE_LEN 65
#endif /* ifdef OSQP_ENABLE_PRINTING */

/**********************
* Utility Functions  *
**********************/
void c_strcpy(char dest[], const char source[]) {
  int i = 0;

  while (1) {
    dest[i] = source[i];

    if (dest[i] == '\0') break;
    i++;
  }
}

#ifdef OSQP_ENABLE_PRINTING

static void print_line(void) {
  char  the_line[HEADER_LINE_LEN + 1];
  OSQPInt i;

  for (i = 0; i < HEADER_LINE_LEN; ++i) the_line[i] = '-';
  the_line[HEADER_LINE_LEN] = '\0';
  c_print("%s\n", the_line);
}

void print_header(void) {
  // Different indentation required for windows
#if defined(IS_WINDOWS) && !defined(PYTHON)
  c_print("iter  ");
#else
  c_print("iter   ");
#endif

  // Main information
  c_print("objective    prim res   dual res   gap        rel kkt    rho");
# ifdef OSQP_ENABLE_PROFILING
  c_print("         time");
# endif /* ifdef OSQP_ENABLE_PROFILING */
  c_print("\n");
}

void print_setup_header(const OSQPSolver* solver) {

  OSQPWorkspace* work;
  OSQPData*      data;
  OSQPSettings*  settings;

  OSQPInt nnz; // Number of nonzeros in the problem

#define NAMEBUFLEN 30
  char namebuf[NAMEBUFLEN];

/* Disable device printing in embedded mode to save stack space */
#ifndef OSQP_EMBEDDED_MODE
  #define DEVICEBUFLEN 150
  char devicebuf[DEVICEBUFLEN];
#endif

  work     = solver->work;
  data     = solver->work->data;
  settings = solver->settings;

  // Number of nonzeros
  nnz = OSQPMatrix_get_nz(data->P) + OSQPMatrix_get_nz(data->A);

  print_line();
  c_print("           OSQP v%s  -  Operator Splitting QP Solver\n"
          "              (c) The OSQP Developer Team\n",
          OSQP_VERSION);
  print_line();

  // Print variables and constraints
  c_print("problem:  ");
  c_print("variables n = %i, constraints m = %i\n          ",
                                    (int)data->n,
          (int)data->m);
  c_print("nnz(P) + nnz(A) = %i\n", (int)nnz);

  // Print Settings
  c_print("settings: ");

  osqp_algebra_name(namebuf, NAMEBUFLEN);
  c_print("algebra = %s", namebuf);
  c_print(",\n          ");

  c_print("OSQPInt = %i bytes, OSQPFloat = %i bytes,\n          ", (int)sizeof(OSQPInt), (int)sizeof(OSQPFloat));

#ifndef OSQP_EMBEDDED_MODE
  osqp_algebra_device_name(devicebuf, DEVICEBUFLEN);

  if (devicebuf[0] != 0 ) {
    c_print("device = %s", devicebuf);
    c_print(",\n          ");
  }
#endif

  c_print("linear system solver = %s", work->linsys_solver->name(work->linsys_solver));

  if (work->linsys_solver->nthreads != 1) {
    c_print(" (%d threads)", (int)work->linsys_solver->nthreads);
  }
  c_print(",\n          ");

  c_print("eps_abs = %.1e, eps_rel = %.1e,\n          ",
          settings->eps_abs, settings->eps_rel);
  c_print("eps_prim_inf = %.1e, eps_dual_inf = %.1e,\n          ",
          settings->eps_prim_inf, settings->eps_dual_inf);
  c_print("rho = %.2e ", settings->rho);

  switch(settings->adaptive_rho)
  {
  case OSQP_ADAPTIVE_RHO_UPDATE_DISABLED:
    c_print("(adaptive: disabled)");
    break;

  case OSQP_ADAPTIVE_RHO_UPDATE_ITERATIONS:
    c_print("(adaptive: %d iterations)", (int) settings->adaptive_rho_interval);
    break;

  case OSQP_ADAPTIVE_RHO_UPDATE_TIME:
    c_print("(adaptive: time)");
    break;

  case OSQP_ADAPTIVE_RHO_UPDATE_KKT_ERROR:
    c_print("(adaptive: kkt error, interval %d)", (int) settings->adaptive_rho_interval);
    break;
  }

  c_print(",\n          ");
  c_print("sigma = %.2e, alpha = %.2f, ",
          settings->sigma, settings->alpha);
  c_print("max_iter = %i\n", (int)settings->max_iter);

  if (settings->check_termination) {
    if(settings->check_dualgap) {
      c_print("          check_termination: on (interval %i, duality gap: on),\n",
        (int)settings->check_termination);
    }
    else {
      c_print("          check_termination: on (interval %i, duality gap: off),\n",
        (int)settings->check_termination);
    }
  }
  else
    c_print("          check_termination: off,\n");

# ifdef OSQP_ENABLE_PROFILING
  if (settings->time_limit)
    c_print("          time_limit: %.2e sec,\n", settings->time_limit);
# endif /* ifdef OSQP_ENABLE_PROFILING */

  if (settings->scaling) {
    c_print("          scaling: on (%i iterations), ", (int)settings->scaling);
  } else {
    c_print("          scaling: off, ");
  }

  if (settings->scaled_termination) {
    c_print("scaled_termination: on\n");
  } else {
    c_print("scaled_termination: off\n");
  }

  if (settings->warm_starting) {
    c_print("          warm starting: on, ");
  } else {
    c_print("          warm starting: off, ");
  }

  if (settings->polishing) {
    c_print("polishing: on, ");
  } else {
    c_print("polishing: off, ");
  }

  c_print("\n");
}

void print_summary(OSQPSolver* solver) {

  OSQPInfo*      info     = solver->info;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  c_print("%4i",     (int)info->iter);
  c_print(" %12.4e", info->obj_val);
  c_print("  %9.2e", info->prim_res);
  c_print("  %9.2e", info->dual_res);
  c_print("  %9.2e", info->duality_gap);
  c_print("  %9.2e", info->rel_kkt_error);

  /* Specially mark the iterations where we have just adapted rho
   * (Note, we print out the new rho value in this iteration, not the old one) */
  if(solver->work->rho_updated) {
    c_print("  %9.2e*", settings->rho);
  } else {
    c_print("  %9.2e ", settings->rho);
  }


# ifdef OSQP_ENABLE_PROFILING

  if (work->first_run) {
    // total time: setup + solve
    c_print("  %9.2es", info->setup_time + info->solve_time);
  } else {
    // total time: update + solve
    c_print("  %9.2es", info->update_time + info->solve_time);
  }
# endif /* ifdef OSQP_ENABLE_PROFILING */
  c_print("\n");

  work->summary_printed = 1; // Summary has been printed
}

void print_polish(OSQPSolver* solver) {

  OSQPInfo*      info = solver->info;
  OSQPWorkspace* work = solver->work;

  c_print("%4s",     "plsh");
  c_print(" %12.4e", info->obj_val);
  c_print("  %9.2e", info->prim_res);
  c_print("  %9.2e", info->dual_res);
  c_print("  %9.2e", info->duality_gap);
  c_print("  %9.2e", info->rel_kkt_error);

  // Different characters for windows/unix
#if defined(IS_WINDOWS) && !defined(PYTHON)
  c_print("  --------- ");
#else
  c_print("   -------- ");
#endif

# ifdef OSQP_ENABLE_PROFILING
  if (work->first_run) {
    // total time: setup + solve
    c_print("  %9.2es", info->setup_time + info->solve_time +
            info->polish_time);
  } else {
    // total time: update + solve
    c_print("  %9.2es", info->update_time + info->solve_time +
            info->polish_time);
  }
# endif /* ifdef OSQP_ENABLE_PROFILING */
  c_print("\n");
}

void print_footer(OSQPInfo* info,
                  OSQPInt   polishing) {
  c_print("\n"); // Add space after iterations

  c_print("status:               %s\n", info->status);

  if (polishing && (info->status_val == OSQP_SOLVED)) {
    if (info->status_polish == OSQP_POLISH_SUCCESS) {
      c_print("solution polishing:   successful\n");
    } else if (info->status_polish < 0) {
      c_print("solution polishing:   unsuccessful\n");
    } else if (info->status_polish == OSQP_POLISH_NO_ACTIVE_SET_FOUND) {
      c_print("solution polishing:   not needed\n");
    }
  }

  c_print("number of iterations: %i\n", (int)info->iter);

  if ((info->status_val == OSQP_SOLVED) ||
      (info->status_val == OSQP_SOLVED_INACCURATE)) {
    c_print("optimal objective:    %.4f\n", info->obj_val);
    c_print("dual objective:       %.4f\n", info->dual_obj_val);
    c_print("duality gap:          %.4e\n", info->duality_gap);
    c_print("primal-dual integral: %.4e\n", info->primdual_int);
  }

# ifdef OSQP_ENABLE_PROFILING
  c_print("run time:             %.2es\n", info->run_time);
# endif /* ifdef OSQP_ENABLE_PROFILING */

# if OSQP_EMBEDDED_MODE != 1
  c_print("optimal rho estimate: %.2e\n", info->rho_estimate);
# endif /* if OSQP_EMBEDDED_MODE != 1 */
  c_print("\n");
}

#endif /* End #ifdef OSQP_ENABLE_PRINTING */


#ifndef OSQP_EMBEDDED_MODE

OSQPSettings* copy_settings(const OSQPSettings *settings) {

  OSQPSettings *new = c_malloc(sizeof(OSQPSettings));
  if (!new) return OSQP_NULL;

  /* Copy settings
   * NB: Copying them explicitly because memcpy is not
   * defined when OSQP_ENABLE_PRINTING is disabled (appears in string.h)
   */
  new->device        = settings->device;
  new->linsys_solver = settings->linsys_solver;

  new->allocate_solution = settings->allocate_solution;
  new->profiler_level    = settings->profiler_level;
  new->verbose           = settings->verbose;
  new->warm_starting     = settings->warm_starting;
  new->scaling           = settings->scaling;
  new->polishing         = settings->polishing;

  new->rho        = settings->rho;
  new->rho_is_vec = settings->rho_is_vec;
  new->sigma      = settings->sigma;
  new->alpha      = settings->alpha;

  new->cg_max_iter      = settings->cg_max_iter;
  new->cg_tol_reduction = settings->cg_tol_reduction;
  new->cg_tol_fraction  = settings->cg_tol_fraction;
  new->cg_precond       = settings->cg_precond;

  new->adaptive_rho           = settings->adaptive_rho;
  new->adaptive_rho_interval  = settings->adaptive_rho_interval;
  new->adaptive_rho_fraction  = settings->adaptive_rho_fraction;
  new->adaptive_rho_tolerance = settings->adaptive_rho_tolerance;

  new->max_iter           = settings->max_iter;
  new->eps_abs            = settings->eps_abs;
  new->eps_rel            = settings->eps_rel;
  new->eps_prim_inf       = settings->eps_prim_inf;
  new->eps_dual_inf       = settings->eps_dual_inf;
  new->scaled_termination = settings->scaled_termination;
  new->check_termination  = settings->check_termination;
  new->check_dualgap      = settings->check_dualgap;
  new->time_limit         = settings->time_limit;

  new->delta              = settings->delta;
  new->polish_refine_iter = settings->polish_refine_iter;

  return new;
}

#endif /* ifndef OSQP_EMBEDDED_MODE */


/* ==================== DEBUG FUNCTIONS ======================= */


#if defined(OSQP_ENABLE_DEBUG) && defined(OSQP_ENABLE_PRINTING)

void print_csc_matrix(const OSQPCscMatrix* M,
                      const char*          name)
{
  OSQPInt j, i, row_start, row_stop;
  OSQPInt k = 0;

  // Print name
  c_print("%s :\n", name);

  for (j = 0; j < M->n; j++) {
    row_start = M->p[j];
    row_stop  = M->p[j + 1];

    if (row_start == row_stop) continue;
    else {
      for (i = row_start; i < row_stop; i++) {
        c_print("\t[%3u,%3u] = %.3g\n", (int)M->i[i], (int)j, M->x[k++]);
      }
    }
  }
}

void dump_csc_matrix(const OSQPCscMatrix* M,
                     const char*          file_name) {
  OSQPInt j, i, row_strt, row_stop;
  OSQPInt k = 0;
  FILE *f = fopen(file_name, "w");

  if (f) {
    for (j = 0; j < M->n; j++) {
      row_strt = M->p[j];
      row_stop = M->p[j + 1];

      if (row_strt == row_stop) continue;
      else {
        for (i = row_strt; i < row_stop; i++) {
          fprintf(f, "%d\t%d\t%20.18e\n",
                  (int)M->i[i] + 1, (int)j + 1, M->x[k++]);
        }
      }
    }
    fprintf(f, "%d\t%d\t%20.18e\n", (int)M->m, (int)M->n, 0.0);
    fclose(f);
    c_print("File %s successfully written.\n", file_name);
  } else {
    c_eprint("Error during writing file %s.\n", file_name);
  }
}

void print_trip_matrix(const OSQPCscMatrix* M,
                       const char*          name)
{
  OSQPInt k = 0;

  // Print name
  c_print("%s :\n", name);

  for (k = 0; k < M->nz; k++) {
    c_print("\t[%3u, %3u] = %.3g\n", (int)M->i[k], (int)M->p[k], M->x[k]);
  }
}

void print_dns_matrix(const OSQPFloat* M,
                            OSQPInt    m,
                            OSQPInt    n,
                      const char*    name)
{
  OSQPInt i, j;

  c_print("%s : \n\t", name);

  for (i = 0; i < m; i++) {   // Cycle over rows
    for (j = 0; j < n; j++) { // Cycle over columns
      if (j < n - 1) {
        // c_print("% 14.12e,  ", M[j*m+i]);
        c_print("% .3g,  ", M[j * m + i]);

      } else {
        // c_print("% 14.12e;  ", M[j*m+i]);
        c_print("% .3g;  ", M[j * m + i]);
      }
    }

    if (i < m - 1) {
      c_print("\n\t");
    }
  }
  c_print("\n");
}

void print_vec(const OSQPFloat* v,
                     OSQPInt    n,
               const char*      name) {
  print_dns_matrix(v, 1, n, name);
}

void dump_vec(const OSQPFloat* v,
                    OSQPInt    len,
              const char*    file_name) {
  OSQPInt i;
  FILE *f = fopen(file_name, "w");

  if (f) {
    for (i = 0; i < len; i++) {
      fprintf(f, "%20.18e\n", v[i]);
    }
    fclose(f);
    c_print("File %s successfully written.\n", file_name);
  } else {
    c_print("Error during writing file %s.\n", file_name);
  }
}

void print_vec_int(const OSQPInt* x,
                         OSQPInt  n,
                   const char*    name) {
  OSQPInt i;

  c_print("%s = [", name);

  for (i = 0; i < n; i++) {
    c_print(" %i ", (int)x[i]);
  }
  c_print("]\n");
}

#endif /* if defined(OSQP_ENABLE_DEBUG) && defined(OSQP_ENABLE_PRINTING) */
