#include <stdio.h>
#include "osqp_api_constants.h"
#include "osqp_api_types.h"


/*********
* Vectors
**********/

static void write_vecf(FILE          *srcfile,
                       const c_float *vecf,
                       c_int          n,
                       const char    *name){

  c_int i;
  fprintf(srcFile, "c_float %s[%d] = {\n", name, n);
  for (i = 0; i < n; i++) {
    fprintf(srcFile, "  %.20f,\n", vecf[i]);
  }
  fprintf(srcFile, "};\n");
}

static void write_veci(FILE        *srcfile,
                       const c_int *veci,
                       c_int        n,
                       const char  *name){

  c_int i;
  fprintf(srcFile, "c_int %s[%d] = {\n", name, n);
  for (i = 0; i < n; i++) {
    fprintf(srcFile, "  %i,\n", veci[i]);
  }
  fprintf(srcFile, "};\n");
}

static void write_OSQPVectorf(FILE              *srcfile,
                              const OSQPVectorf *vec,
                              const char        *name){
  
  char vecf_name[50];
  sprintf(vecf_name, "%s_val", name);
  write_vecf(srcfile, vec->values, vec->length, vecf_name);
  fprintf(srcFile, "OSQPVectorf %s = {\n  %s,\n  %d\n};\n", name, vecf_name, vec->length);
}

static void write_OSQPVectori(FILE              *srcfile,
                              const OSQPVectori *vec,
                              const char        *name){
  
  char veci_name[50];
  sprintf(veci_name, "%s_val", name);
  write_veci(srcfile, vec->values, vec->length, veci_name);
  fprintf(srcFile, "OSQPVectori %s = {\n  %s,\n  %d\n};\n", name, vecf_name, vec->length);
}



/*********
* Matrix
**********/

void write_OSQPMatrix(FILE             *srcfile,
                      const OSQPMatrix *mat,
                      const char       *name){

  write_csc(srcFile, mat->csc, name);
  fprintf(srcFile, "OSQPMatrix %s = {\n  ", name, n);
  
}

struct OSQPMatrix_ {
  csc*                             csc;
  OSQPMatrix_symmetry_type    symmetry;
};

typedef struct {
  c_int    m;     ///< number of rows
  c_int    n;     ///< number of columns
  c_int   *p;     ///< column pointers (size n+1); col indices (size nzmax) starting from 0 for triplet format
  c_int   *i;     ///< row indices, size nzmax starting from 0
  c_float *x;     ///< numerical values, size nzmax
  c_int    nzmax; ///< maximum number of entries
  c_int    nz;    ///< number of entries in triplet matrix, -1 for csc
} csc;


/**********
* Settings
***********/

// void write_settings_inc(FILE             *incfile,
//                         const OSQPSolver *solver){

//   fprintf(incFile, "/* Settings structure prototype */\n");
//   fprintf(incFile, "extern OSQPSettings settings;\n\n");
// }

void write_settings_src(FILE             *srcfile,
                        const OSQPSolver *solver){

  OSQPSettings *settings = solver->settings;
  
  fprintf(srcFile, "/* Define the settings structure */\n");
  fprintf(srcFile, "OSQPSettings settings = {\n");
  fprintf(srcFile, "  0,\n");                                                 // device
  fprintf(srcFile, "  OSQP_LINSYS_SOLVER,\n");                                // linsys_solver
  fprintf(srcFile, "  0,\n");                                                 // verbose
  fprintf(srcFile, "  %d,\n", settings->warm_starting);                       // warm_starting
  fprintf(srcFile, "  %d,\n", settings->scaling);                             // scaling
  fprintf(srcFile, "  0,\n",);                                                // polishing
  fprintf(srcFile, "  (c_float)%.20f,\n", settings->rho);                     // rho
  fprintf(srcFile, "  %d,\n", settings->rho_is_vec);                          // rho_is_vec
  fprintf(srcFile, "  (c_float)%.20f,\n", settings->sigma);                   // sigma
  fprintf(srcFile, "  (c_float)%.20f,\n", settings->alpha);                   // alpha
  fprintf(srcFile, "  %d,\n", settings->cg_max_iter);                         // cg_max_iter
  fprintf(srcFile, "  %d,\n", settings->cg_tol_reduction);                    // cg_tol_reduction
  fprintf(srcFile, "  (c_float)%.20f,\n", settings->cg_tol_fraction);         // cg_tol_fraction
  fprintf(srcFile, "  %d,\n", settings->adaptive_rho);                        // adaptive_rho
  fprintf(srcFile, "  %d,\n", settings->adaptive_rho_interval);               // adaptive_rho_interval
  fprintf(srcFile, "  (c_float)%.20f,\n", settings->adaptive_rho_fraction);   // adaptive_rho_fraction
  fprintf(srcFile, "  (c_float)%.20f,\n", settings->adaptive_rho_tolerance);  // adaptive_rho_tolerance
  fprintf(srcFile, "  %d,\n", settings->max_iter);                            // max_iter
  fprintf(srcFile, "  (c_float)%.20f,\n", settings->eps_abs);                 // eps_abs
  fprintf(srcFile, "  (c_float)%.20f,\n", settings->eps_rel);                 // eps_rel
  fprintf(srcFile, "  (c_float)%.20f,\n", settings->eps_prim_inf);            // eps_prim_inf
  fprintf(srcFile, "  (c_float)%.20f,\n", settings->eps_dual_inf);            // eps_dual_inf
  fprintf(srcFile, "  %d,\n", settings->scaled_termination);                  // scaled_termination
  fprintf(srcFile, "  %d,\n", settings->check_termination);                   // check_termination
  fprintf(srcFile, "  (c_float)%.20f,\n", settings->time_limit);              // time_limit
  fprintf(srcFile, "  (c_float)%.20f,\n", settings->delta);                   // delta
  fprintf(srcFile, "  %d,\n", settings->polish_refine_iter);                  // polish_refine_iter
  fprintf(srcFile, "};\n\n");
}


/******
* Info
*******/

// void write_info_inc(FILE             *incfile,
//                     const OSQPSolver *solver){

//   fprintf(incFile, "/* Info structure prototype */\n");
//   fprintf(incFile, "extern OSQPInfo info;\n\n");
// }

void write_info_src(FILE             *srcfile,
                    const OSQPSolver *solver){

  OSQPInfo *info = solver->info;
  
  fprintf(srcFile, "/* Define the info structure */\n");
  fprintf(srcFile, "OSQPInfo info = {\n");
  fprintf(srcFile, "  %s,\n", info->status);                    // status
  fprintf(srcFile, "  %d,\n", info->status_val);                // status_val
  fprintf(srcFile, "  %d,\n", info->status_polish);             // status_polish
  fprintf(srcFile, "  (c_float)%.20f,\n", info->obj_val);       // obj_val
  fprintf(srcFile, "  (c_float)%.20f,\n", info->prim_res);      // prim_res
  fprintf(srcFile, "  (c_float)%.20f,\n", info->dual_res);      // dual_res
  fprintf(srcFile, "  %d,\n", info->iter);                      // iter
  fprintf(srcFile, "  %d,\n", info->rho_updates);               // rho_updates
  fprintf(srcFile, "  (c_float)%.20f,\n", info->rho_estimate);  // rho_estimate
  fprintf(srcFile, "  (c_float)%.20f,\n", info->setup_time);    // setup_time
  fprintf(srcFile, "  (c_float)%.20f,\n", info->solve_time);    // solve_time
  fprintf(srcFile, "  (c_float)%.20f,\n", info->update_time);   // update_time
  fprintf(srcFile, "  (c_float)%.20f,\n", info->polish_time);   // polish_time
  fprintf(srcFile, "  (c_float)%.20f,\n", info->run_time);      // run_time
  fprintf(srcFile, "};\n\n");
}


/**********
* Solution
***********/

// void write_solution_inc(FILE             *incfile,
//                         const OSQPSolver *solver){

//   // c_int m = solver->work->data->m;
//   // c_int n = solver->work->data->n;

//   fprintf(incFile, "/* Solution prototype */\n");
//   // fprintf(incFile, "extern c_float xsolution[%d];\n", n);
//   // if (m > 0) fprintf(incFile, "extern c_float ysolution[%d];\n", m);
//   // else       fprintf(incFile, "extern c_float *ysolution;\n");
//   fprintf(incFile, "extern OSQPSolution solution;\n\n");
// }

void write_solution_src(FILE             *srcfile,
                        const OSQPSolver *solver){

  c_int m = solver->work->data->m;
  c_int n = solver->work->data->n;
  
  fprintf(srcFile, "/* Define solution */\n");
  fprintf(srcFile, "c_float xsolution[%d];\n", n);
  if (m > 0) fprintf(srcFile, "c_float ysolution[%d];\n", m);
  else       fprintf(srcFile, "c_float *ysolution = NULL;\n");
  fprintf(srcFile, "OSQPSolution solution = {\n  xsolution,\n  ysolution\n};\n\n");
}


/*********
* Scaling
**********/

// void write_scaling_inc(FILE             *incfile,
//                        const OSQPSolver *solver){

//   // OSQPScaling *scaling = solver->work->scaling;

//   fprintf(incFile, "/* Scaling structure prototype */\n");
//   // write_vecf_extern(incFile, scaling->D,    "Dscaling");
//   // write_vecf_extern(incFile, scaling->E,    "Escaling");
//   // write_vecf_extern(incFile, scaling->Dinv, "Dinvscaling");
//   // write_vecf_extern(incFile, scaling->Einv, "Einvscaling");
//   fprintf(incFile, "extern OSQPScaling scaling;\n\n");
// }

void write_scaling_src(FILE             *srcfile,
                       const OSQPSolver *solver){

  OSQPScaling *scaling = solver->work->scaling;
  
  fprintf(srcFile, "/* Define the scaling structure */\n");
  write_vecf(srcFile, scaling->D,    "Dscaling");
  write_vecf(srcFile, scaling->E,    "Escaling");
  write_vecf(srcFile, scaling->Dinv, "Dinvscaling");
  write_vecf(srcFile, scaling->Einv, "Einvscaling");
  fprintf(srcFile, "OSQPScaling scaling = {\n");
  fprintf(srcFile, "  (c_float)%.20f,\n", scaling->c);      // c
  fprintf(srcFile, "  Dscaling,\n");                        // D
  fprintf(srcFile, "  Escaling,\n");                        // E
  fprintf(srcFile, "  (c_float)%.20f,\n", scaling->cinv);   // cinv
  fprintf(srcFile, "  Dinvscaling,\n");                     // Dinv
  fprintf(srcFile, "  Einvscaling,\n");                     // Einv
  fprintf(srcFile, "};\n\n");
}
