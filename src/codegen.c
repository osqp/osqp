#include <stdio.h>
#include <ctype.h>  /* -> toupper */
#include <time.h>   /* time, ctime */

#include "error.h"
#include "types.h"
#include "algebra_impl.h"
#include "qdldl_interface.h"


/*********
* Vectors
**********/

static void write_vecf(FILE          *f,
                       const c_float *vecf,
                       c_int          n,
                       const char    *name){

  c_int i;
  if (n) {
    fprintf(f, "c_float %s[%d] = {\n", name, n);
    for (i = 0; i < n; i++) {
      fprintf(f, "  (c_float)%.20f,\n", vecf[i]);
    }
    fprintf(f, "};\n");
  }
  else {
    fprintf(f, "c_float *%s = NULL;\n", name);
  }
}

static void write_veci(FILE        *f,
                       const c_int *veci,
                       c_int        n,
                       const char  *name){

  c_int i;
  if (n) {
    fprintf(f, "c_int %s[%d] = {\n", name, n);
    for (i = 0; i < n; i++) {
      fprintf(f, "  %i,\n", veci[i]);
    }
    fprintf(f, "};\n");
  }
  else {
    fprintf(f, "c_int *%s = NULL;\n", name);
  }
}

static void write_OSQPVectorf(FILE              *f,
                              const OSQPVectorf *vec,
                              const char        *name){
  
  char vecf_name[50];
  sprintf(vecf_name, "%s_val", name);
  write_vecf(f, vec->values, vec->length, vecf_name);
  fprintf(f, "OSQPVectorf %s = {\n  %s,\n  %d\n};\n", name, vecf_name, vec->length);
}

static void write_OSQPVectori(FILE              *f,
                              const OSQPVectori *vec,
                              const char        *name){
  
  char veci_name[50];
  sprintf(veci_name, "%s_val", name);
  write_veci(f, vec->values, vec->length, veci_name);
  fprintf(f, "OSQPVectori %s = {\n  %s,\n  %d\n};\n", name, veci_name, vec->length);
}


/*********
* Matrix
**********/

static void write_csc(FILE       *f,
                      const csc  *M,
                      const char *name){

char vec_name[50];
sprintf(vec_name, "%s_p", name);
write_veci(f, M->p, M->n+1, vec_name);
sprintf(vec_name, "%s_i", name);
write_veci(f, M->i, M->nzmax, vec_name);
sprintf(vec_name, "%s_x", name);
write_vecf(f, M->x, M->nzmax, vec_name);
fprintf(f, "csc %s = {\n", name);
fprintf(f, "  %d,\n", M->m);
fprintf(f, "  %d,\n", M->n);
fprintf(f, "  %s_p,\n", name);
fprintf(f, "  %s_i,\n", name);
fprintf(f, "  %s_x,\n", name);
fprintf(f, "  %d,\n", M->nzmax);
fprintf(f, "  %d,\n", M->nz);
fprintf(f, "};\n");
}

static void write_OSQPMatrix(FILE             *f,
                             const OSQPMatrix *mat,
                             const char       *name){

  char csc_name[50];
  sprintf(csc_name, "%s_csc", name);
  write_csc(f, mat->csc, csc_name);
  fprintf(f, "OSQPMatrix %s = {\n", name);
  fprintf(f, "  &%s,\n", csc_name);
  fprintf(f, "  %d\n", mat->symmetry);
  fprintf(f, "};\n");
  
}


/**********
* Settings
***********/

static void write_settings(FILE               *f,
                           const OSQPSettings *settings,
                           const char         *prefix){
  
  fprintf(f, "/* Define the settings structure */\n");
  fprintf(f, "OSQPSettings %ssettings = {\n", prefix);
  fprintf(f, "  0,\n"); // device
  fprintf(f, "  OSQP_LINSYS_SOLVER,\n");
  fprintf(f, "  0,\n"); // verbose
  fprintf(f, "  %d,\n", settings->warm_starting);
  fprintf(f, "  %d,\n", settings->scaling);
  fprintf(f, "  0,\n"); // polishing
  fprintf(f, "  (c_float)%.20f,\n", settings->rho);
  fprintf(f, "  %d,\n", settings->rho_is_vec);
  fprintf(f, "  (c_float)%.20f,\n", settings->sigma);
  fprintf(f, "  (c_float)%.20f,\n", settings->alpha);
  fprintf(f, "  %d,\n", settings->cg_max_iter);
  fprintf(f, "  %d,\n", settings->cg_tol_reduction);
  fprintf(f, "  (c_float)%.20f,\n", settings->cg_tol_fraction);
  fprintf(f, "  %d,\n", settings->adaptive_rho);
  fprintf(f, "  %d,\n", settings->adaptive_rho_interval);
  fprintf(f, "  (c_float)%.20f,\n", settings->adaptive_rho_fraction);
  fprintf(f, "  (c_float)%.20f,\n", settings->adaptive_rho_tolerance);
  fprintf(f, "  %d,\n", settings->max_iter);
  fprintf(f, "  (c_float)%.20f,\n", settings->eps_abs);
  fprintf(f, "  (c_float)%.20f,\n", settings->eps_rel);
  fprintf(f, "  (c_float)%.20f,\n", settings->eps_prim_inf);
  fprintf(f, "  (c_float)%.20f,\n", settings->eps_dual_inf);
  fprintf(f, "  %d,\n", settings->scaled_termination);
  fprintf(f, "  %d,\n", settings->check_termination);
  fprintf(f, "  (c_float)%.20f,\n", settings->time_limit);
  fprintf(f, "  (c_float)%.20f,\n", settings->delta);
  fprintf(f, "  %d,\n", settings->polish_refine_iter);
  fprintf(f, "};\n\n");
}


/******
* Info
*******/

static void write_info(FILE           *f,
                       const OSQPInfo *info,
                       const char     *prefix){
  
  fprintf(f, "/* Define the info structure */\n");
  fprintf(f, "OSQPInfo %sinfo = {\n", prefix);
  fprintf(f, "  \"%s\",\n", OSQP_STATUS_MESSAGE[OSQP_UNSOLVED]);
  fprintf(f, "  %d,\n", OSQP_UNSOLVED);
  fprintf(f, "  0,\n"); // status_polish
  fprintf(f, "  (c_float)%.20f,\n", OSQP_INFTY); // obj_val
  fprintf(f, "  (c_float)%.20f,\n", OSQP_INFTY); // prim_res
  fprintf(f, "  (c_float)%.20f,\n", OSQP_INFTY); // dual_res
  fprintf(f, "  0,\n"); // iter (iteration count)
  fprintf(f, "  0,\n"); // rho_updates
  fprintf(f, "  (c_float)%.20f,\n", info->rho_estimate);
  fprintf(f, "  (c_float)0.0,\n"); // setup_time
  fprintf(f, "  (c_float)0.0,\n"); // solve_time
  fprintf(f, "  (c_float)0.0,\n"); // update_time
  fprintf(f, "  (c_float)0.0,\n"); // polish_time
  fprintf(f, "  (c_float)0.0,\n"); // run_time
  fprintf(f, "};\n\n");
}


/**********
* Solution
***********/

static void write_solution(FILE       *f,
                           c_int       n,
                           c_int       m,
                           const char *prefix){

  fprintf(f, "/* Define the solution structure */\n");
  fprintf(f, "c_float %ssol_x[%d];\n", prefix, n);
  if (m > 0) fprintf(f, "c_float %ssol_y[%d];\n", prefix, m);
  else       fprintf(f, "c_float *%ssol_y = NULL;\n", prefix);
  if (m > 0) fprintf(f, "c_float %ssol_prim_inf_cert[%d];\n", prefix, m);
  else       fprintf(f, "c_float *%ssol_prim_inf_cert = NULL;\n", prefix);
  fprintf(f, "c_float %ssol_dual_inf_cert[%d];\n", prefix, n);
  fprintf(f, "OSQPSolution %ssol = {\n", prefix);
  fprintf(f, "  %ssol_x,\n", prefix);
  fprintf(f, "  %ssol_y,\n", prefix);
  fprintf(f, "  %ssol_prim_inf_cert,\n", prefix);
  fprintf(f, "  %ssol_dual_inf_cert,\n", prefix);
  fprintf(f, "};\n\n");
}


/*********
* Scaling
**********/

static void write_scaling(FILE              *f,
                          const OSQPScaling *scaling,
                          const char        *prefix){

  char name[50];
  fprintf(f, "\n/* Define the scaling structure */\n");
  sprintf(name, "%sscaling_D", prefix);
  write_OSQPVectorf(f, scaling->D,    name);
  sprintf(name, "%sscaling_E", prefix);
  write_OSQPVectorf(f, scaling->E,    name);
  sprintf(name, "%sscaling_Dinv", prefix);
  write_OSQPVectorf(f, scaling->Dinv, name);
  sprintf(name, "%sscaling_Einv", prefix);
  write_OSQPVectorf(f, scaling->Einv, name);
  fprintf(f, "OSQPScaling %sscaling = {\n", prefix);
  fprintf(f, "  (c_float)%.20f,\n", scaling->c);
  fprintf(f, "  &%sscaling_D,\n", prefix);
  fprintf(f, "  &%sscaling_E,\n", prefix);
  fprintf(f, "  (c_float)%.20f,\n", scaling->cinv);
  fprintf(f, "  &%sscaling_Dinv,\n", prefix);
  fprintf(f, "  &%sscaling_Einv\n", prefix);
  fprintf(f, "};\n\n");
}


/******
* Data
*******/

static void write_data(FILE           *f,
                       const OSQPData *data,
                       const char     *prefix){

  char name[50];
  fprintf(f, "/* Define the data structure */\n");
  sprintf(name, "%sdata_P", prefix);
  write_OSQPMatrix(f,  data->P, name);
  sprintf(name, "%sdata_A", prefix);
  write_OSQPMatrix(f,  data->A, name);
  sprintf(name, "%sdata_q", prefix);
  write_OSQPVectorf(f, data->q, name);
  sprintf(name, "%sdata_l", prefix);
  write_OSQPVectorf(f, data->l, name);
  sprintf(name, "%sdata_u", prefix);
  write_OSQPVectorf(f, data->u, name);
  fprintf(f, "OSQPData %sdata = {\n", prefix);
  fprintf(f, "  %d,\n", data->n);
  fprintf(f, "  %d,\n", data->m);
  fprintf(f, "  &%sdata_P,\n", prefix);
  fprintf(f, "  &%sdata_A,\n", prefix);
  fprintf(f, "  &%sdata_q,\n", prefix);
  fprintf(f, "  &%sdata_l,\n", prefix);
  fprintf(f, "  &%sdata_u\n", prefix);
  fprintf(f, "};\n\n");
}


/**********************
* Linear System Solver
***********************/

static void write_linsys(FILE               *f,
                         const qdldl_solver *linsys,
                         const OSQPData     *data,
                         const char         *prefix,
                         c_int               embedded){

  char name[50];
  c_int n = linsys->n;
  c_int m = linsys->m;

  fprintf(f, "/* Define the linear system solver structure */\n");
  sprintf(name, "%slinsys_L", prefix);
  write_csc(f, linsys->L, name);
  sprintf(name, "%slinsys_Dinv", prefix);
  write_vecf(f, linsys->Dinv, n+m, name);
  sprintf(name, "%slinsys_P", prefix);
  write_veci(f, linsys->P, n+m, name);
  fprintf(f, "c_float %slinsys_bp[%d];\n",  prefix, n+m);
  fprintf(f, "c_float %slinsys_sol[%d];\n", prefix, n+m);
  sprintf(name, "%slinsys_rho_inv_vec", prefix);
  write_vecf(f, linsys->rho_inv_vec, n+m, name);
  if (embedded > 1) {
    sprintf(name, "%slinsys_KKT", prefix);
    write_csc(f, linsys->KKT, name);
    sprintf(name, "%slinsys_PtoKKT", prefix);
    write_veci(f, linsys->PtoKKT, data->P->csc->p[n], name);
    sprintf(name, "%slinsys_AtoKKT", prefix);
    write_veci(f, linsys->AtoKKT, data->A->csc->p[n], name);
    sprintf(name, "%slinsys_rhotoKKT", prefix);
    write_veci(f, linsys->rhotoKKT, m, name);
    sprintf(name, "%slinsys_D", prefix);
    write_vecf(f, linsys->D, n+m, name);
    sprintf(name, "%slinsys_etree", prefix);
    write_veci(f, linsys->etree, n+m, name);
    sprintf(name, "%slinsys_Lnz", prefix);
    write_veci(f, linsys->Lnz, n+m, name);
    fprintf(f, "QDLDL_int   %slinsys_iwork[%d];\n", prefix, 3*(n+m));
    fprintf(f, "QDLDL_bool  %slinsys_bwork[%d];\n", prefix, n+m);
    fprintf(f, "QDLDL_float %slinsys_fwork[%d];\n", prefix, n+m);
  }

  fprintf(f, "qdldl_solver %slinsys = {\n", prefix);
  fprintf(f, "  %d,\n", linsys->type);
  fprintf(f, "  &name_qdldl,\n");
  fprintf(f, "  &solve_linsys_qdldl,\n");
  fprintf(f, "  &update_settings_linsys_solver_qdldl,\n");
  fprintf(f, "  &warm_start_linsys_solver_qdldl,\n");
  if (embedded > 1) {
    fprintf(f, "  &update_linsys_solver_matrices_qdldl,\n");
    fprintf(f, "  &update_linsys_solver_rho_vec_qdldl,\n");
  }
  fprintf(f, "  &%slinsys_L,\n", prefix);
  fprintf(f, "  %slinsys_Dinv,\n", prefix);
  fprintf(f, "  %slinsys_P,\n", prefix);
  fprintf(f, "  %slinsys_bp,\n", prefix);
  fprintf(f, "  %slinsys_sol,\n", prefix);
  fprintf(f, "  %slinsys_rho_inv_vec,\n", prefix);
  fprintf(f, "  (c_float)%.20f,\n", linsys->sigma);
  fprintf(f, "  (c_float)%.20f,\n", linsys->rho_inv);
  fprintf(f, "  %d,\n", n);
  fprintf(f, "  %d,\n", m);
  if (embedded > 1) {
    fprintf(f, "  &%slinsys_KKT,\n", prefix);
    fprintf(f, "  %slinsys_PtoKKT,\n", prefix);
    fprintf(f, "  %slinsys_AtoKKT,\n", prefix);
    fprintf(f, "  %slinsys_rhotoKKT,\n", prefix);
    fprintf(f, "  %slinsys_D,\n", prefix);
    fprintf(f, "  %slinsys_etree,\n", prefix);
    fprintf(f, "  %slinsys_Lnz,\n", prefix);
    fprintf(f, "  %slinsys_iwork,\n", prefix);
    fprintf(f, "  %slinsys_bwork,\n", prefix);
    fprintf(f, "  %slinsys_fwork,\n", prefix);
  }
  fprintf(f, "};\n\n");
}


/***********
* Workspace
************/

static void write_workspace(FILE                *f,
                            const OSQPWorkspace *work,
                            c_int                n,
                            c_int                m,
                            const char          *prefix,
                            c_int                embedded){

  char name[50];

  write_data(f, work->data, prefix);
  write_linsys(f, (qdldl_solver *)work->linsys_solver, work->data, prefix, embedded);
  sprintf(name, "%swork_rho_vec", prefix);
  write_OSQPVectorf(f, work->rho_vec, name);
  sprintf(name, "%swork_rho_inv_vec", prefix);
  write_OSQPVectorf(f, work->rho_inv_vec, name);
  if (embedded > 1) {
    sprintf(name, "%swork_constr_type", prefix);
    write_OSQPVectori(f, work->constr_type, name);
  }

  /* Initialize x,y,z as we usually want to warm start the iterates */
  sprintf(name, "%swork_x", prefix);
  write_OSQPVectorf(f, work->x, name);
  sprintf(name, "%swork_y", prefix);
  write_OSQPVectorf(f, work->y, name);
  sprintf(name, "%swork_z", prefix);
  write_OSQPVectorf(f, work->z, name);

  fprintf(f, "c_float %swork_xz_tilde_val[%d];\n", prefix, n+m);
  fprintf(f, "OSQPVectorf %swork_xz_tilde = {\n  %swork_xz_tilde_val,\n  %d\n};\n", prefix, prefix, n+m);
  fprintf(f, "OSQPVectorf %swork_xtilde_view = {\n  %swork_xz_tilde_val,\n  %d\n};\n", prefix, prefix, n);
  fprintf(f, "OSQPVectorf %swork_ztilde_view = {\n  %swork_xz_tilde_val+%d,\n  %d\n};\n", prefix, prefix, n, m);
  fprintf(f, "c_float %swork_x_prev_val[%d];\n", prefix, n);
  fprintf(f, "OSQPVectorf %swork_x_prev = {\n  %swork_x_prev_val,\n  %d\n};\n", prefix, prefix, n);
  if (m > 0) {
    fprintf(f, "c_float %swork_z_prev_val[%d];\n", prefix, m);
    fprintf(f, "OSQPVectorf %swork_z_prev = {\n  %swork_z_prev_val,\n  %d\n};\n", prefix, prefix, m);
    fprintf(f, "c_float %swork_Ax_val[%d];\n", prefix, m);
    fprintf(f, "OSQPVectorf %swork_Ax = {\n  %swork_Ax_val,\n  %d\n};\n", prefix, prefix, m);
  }
  else {
    fprintf(f, "OSQPVectorf %swork_z_prev = { NULL, 0 };\n", prefix);
    fprintf(f, "OSQPVectorf %swork_Ax = { NULL, 0 };\n", prefix);
  }
  fprintf(f, "c_float %swork_Px_val[%d];\n", prefix, n);
  fprintf(f, "OSQPVectorf %swork_Px = {\n  %swork_Px_val,\n  %d\n};\n", prefix, prefix, n);
  fprintf(f, "c_float %swork_Aty_val[%d];\n", prefix, n);
  fprintf(f, "OSQPVectorf %swork_Aty = {\n  %swork_Aty_val,\n  %d\n};\n", prefix, prefix, n);
  if (m > 0) {
    fprintf(f, "c_float %swork_delta_y_val[%d];\n", prefix, m);
    fprintf(f, "OSQPVectorf %swork_delta_y = {\n  %swork_delta_y_val,\n  %d\n};\n", prefix, prefix, m);
  }
  else {
    fprintf(f, "OSQPVectorf %swork_delta_y = { NULL, 0 };\n", prefix);
  }
  fprintf(f, "c_float %swork_Atdelta_y_val[%d];\n", prefix, n);
  fprintf(f, "OSQPVectorf %swork_Atdelta_y = {\n  %swork_Atdelta_y_val,\n  %d\n};\n", prefix, prefix, n);
  fprintf(f, "c_float %swork_delta_x_val[%d];\n", prefix, n);
  fprintf(f, "OSQPVectorf %swork_delta_x = {\n  %swork_delta_x_val,\n  %d\n};\n", prefix, prefix, n);
  fprintf(f, "c_float %swork_Pdelta_x_val[%d];\n", prefix, n);
  fprintf(f, "OSQPVectorf %swork_Pdelta_x = {\n  %swork_Pdelta_x_val,\n  %d\n};\n", prefix, prefix, n);
  if (m > 0) {
    fprintf(f, "c_float %swork_Adelta_x_val[%d];\n", prefix, m);
    fprintf(f, "OSQPVectorf %swork_Adelta_x = {\n  %swork_Adelta_x_val,\n  %d\n};\n", prefix, prefix, m);
  }
  else {
    fprintf(f, "OSQPVectorf %swork_Adelta_x = { NULL, 0 };\n", prefix);
  }
  if (embedded > 1) {
    fprintf(f, "c_float %swork_D_temp_val[%d];\n", prefix, n);
    fprintf(f, "OSQPVectorf %swork_D_temp = {\n  %swork_D_temp_val,\n  %d\n};\n", prefix, prefix, n);
    fprintf(f, "c_float %swork_D_temp_A_val[%d];\n", prefix, n);
    fprintf(f, "OSQPVectorf %swork_D_temp_A = {\n  %swork_D_temp_A_val,\n  %d\n};\n", prefix, prefix, n);
    if (m > 0) {
      fprintf(f, "c_float %swork_E_temp_val[%d];\n", prefix, m);
      fprintf(f, "OSQPVectorf %swork_E_temp = {\n  %swork_E_temp_val,\n  %d\n};\n", prefix, prefix, m);
    }
    else {
      fprintf(f, "OSQPVectorf %swork_E_temp = { NULL, 0 };\n", prefix);
    }
  }
  write_scaling(f, work->scaling, prefix);
  
  fprintf(f, "/* Define the workspace structure */\n");
  fprintf(f, "OSQPWorkspace %swork = {\n", prefix);
  fprintf(f, "  &%sdata,\n", prefix);
  fprintf(f, "  (LinSysSolver *)&%slinsys,\n", prefix);
  fprintf(f, "  &%swork_rho_vec,\n", prefix);
  fprintf(f, "  &%swork_rho_inv_vec,\n", prefix);
  if (embedded > 1) fprintf(f, "  &%swork_constr_type,\n", prefix);
  fprintf(f, "  &%swork_x,\n", prefix);
  fprintf(f, "  &%swork_y,\n", prefix);
  fprintf(f, "  &%swork_z,\n", prefix);
  fprintf(f, "  &%swork_xz_tilde,\n", prefix);
  fprintf(f, "  &%swork_xtilde_view,\n", prefix);
  fprintf(f, "  &%swork_ztilde_view,\n", prefix);
  fprintf(f, "  &%swork_x_prev,\n", prefix);
  fprintf(f, "  &%swork_z_prev,\n", prefix);
  fprintf(f, "  &%swork_Ax,\n", prefix);
  fprintf(f, "  &%swork_Px,\n", prefix);
  fprintf(f, "  &%swork_Aty,\n", prefix);
  fprintf(f, "  &%swork_delta_y,\n", prefix);
  fprintf(f, "  &%swork_Atdelta_y,\n", prefix);
  fprintf(f, "  &%swork_delta_x,\n", prefix);
  fprintf(f, "  &%swork_Pdelta_x,\n", prefix);
  fprintf(f, "  &%swork_Adelta_x,\n", prefix);
  if (embedded > 1) {
    fprintf(f, "  &%swork_D_temp,\n", prefix);
    fprintf(f, "  &%swork_D_temp_A,\n", prefix);
    fprintf(f, "  &%swork_E_temp,\n", prefix);
  }
  fprintf(f, "  &%sscaling,\n", prefix);
  fprintf(f, "  (c_float)0.0,\n"); // scaled_prim_res
  fprintf(f, "  (c_float)0.0,\n"); // scaled_dual_res
  fprintf(f, "  (c_float)%.20f,\n", work->rho_inv);
  fprintf(f, "};\n\n");
}


/*********
* Solver
**********/

static void write_solver(FILE             *f,
                         const OSQPSolver *solver,
                         const char       *prefix,
                         c_int             embedded){

  c_int n = solver->work->data->n;
  c_int m = solver->work->data->m;

  write_settings(f, solver->settings, prefix);
  write_solution(f, n, m, prefix);
  write_info(f, solver->info, prefix);
  write_workspace(f, solver->work, n, m, prefix, embedded);

  fprintf(f, "/* Define the solver structure */\n");
  fprintf(f, "OSQPSolver %ssolver = {\n", prefix);
  fprintf(f, "  &%ssettings,\n", prefix);
  fprintf(f, "  &%ssol,\n", prefix);
  fprintf(f, "  &%sinfo,\n", prefix);
  fprintf(f, "  &%swork\n", prefix);
  fprintf(f, "};\n");
}


/*************
* Codegen API
**************/

/* Define the maximum allowed length of the path (directory + filename + extension) */
#define PATH_LENGTH 1024

/* Define the maximum allowed length of the filename (no extension)*/
#define FILE_LENGTH 100

c_int codegen_inc(OSQPSolver *solver,
                  const char *output_dir,
                  const char *file_prefix){

  char fname[FILE_LENGTH], hfname[PATH_LENGTH], incGuard[FILE_LENGTH];
  FILE *incFile;
  time_t now;
  c_int i = 0;

  sprintf(fname,  "%sworkspace", file_prefix);
  sprintf(hfname, "%s%s.h", output_dir, fname);

  /* Open include file */
  incFile = fopen(hfname, "w");
  if (incFile == NULL)
    return osqp_error(OSQP_FOPEN_ERROR);

  /* Print comment headers containing the generation time into the files */
  time(&now);
  fprintf(incFile, "/*\n");
  fprintf(incFile, " * This file was autogenerated by OSQP on %s", ctime(&now));
  fprintf(incFile, " * \n");
  fprintf(incFile, " * This file contains the prototype for the solver structure needed\n");
  fprintf(incFile, " * by OSQP. The actual data is contained inside %sworkspace.c.\n", file_prefix);
  fprintf(incFile, " */\n\n");

  /* Add an include-guard statement */
  sprintf(incGuard, "%s_H", fname);
  while(incGuard[i]){
    incGuard[i] = toupper(incGuard[i]);
    i++;
  }
  fprintf(incFile, "#ifndef %s\n",   incGuard);
  fprintf(incFile, "#define %s\n\n", incGuard);

  /* Include required headers */
  fprintf(incFile, "#include \"osqp_api_types.h\"\n\n");

  fprintf(incFile, "extern OSQPSolver %ssolver;\n\n", file_prefix);

  /* The endif for the include-guard statement */
  fprintf(incFile, "#endif /* ifndef %s */\n", incGuard);

  /* Close header file */
  fclose(incFile);

  return 0;
}


c_int codegen_src(OSQPSolver *solver,
                  const char *output_dir,
                  const char *file_prefix,
                  c_int       embedded){

  char fname[PATH_LENGTH], cfname[PATH_LENGTH];
  FILE *srcFile;
  time_t now;

  sprintf(fname,  "%s%sworkspace", output_dir, file_prefix);
  sprintf(cfname, "%s.c",        fname);

  /* Open source file */
  srcFile = fopen(cfname, "w");
  if (srcFile == NULL)
    return osqp_error(OSQP_FOPEN_ERROR);

  /* Print comment headers containing the generation time into the files */
  time(&now);
  fprintf(srcFile, "/*\n");
  fprintf(srcFile, " * This file was autogenerated by OSQP on %s", ctime(&now));
  fprintf(srcFile, " * \n");
  fprintf(srcFile, " * This file contains the workspace variables needed by OSQP.\n");
  fprintf(srcFile, " */\n\n");

  /* Include required headers */
  fprintf(srcFile, "#include \"types.h\"\n");
  fprintf(srcFile, "#include \"algebra_impl.h\"\n");
  fprintf(srcFile, "#include \"qdldl_interface.h\"\n\n");

  /* Write the workspace variables to file */
  write_solver(srcFile, solver, file_prefix, embedded);

  /* Close header file */
  fclose(srcFile);

  return 0;
}


c_int codegen_defines(const char *output_dir,
                      OSQPCodegenDefines *defines) {
  char cfname[PATH_LENGTH];
  FILE *incFile;
  time_t now;

  sprintf(cfname,  "%sosqp_configure.h", output_dir);

  /* Open source file */
  incFile = fopen(cfname, "w");
  if (incFile == NULL)
    return osqp_error(OSQP_FOPEN_ERROR);

  /* Print comment headers containing the generation time into the files */
  time(&now);
  fprintf(incFile, "/*\n");
  fprintf(incFile, " * This file was autogenerated by OSQP on %s", ctime(&now));
  fprintf(incFile, " * \n");
  fprintf(incFile, " * This file contains the configuration options needed by OSQP.\n");
  fprintf(incFile, " */\n\n");

  /* Add an include-guard statement */
  fprintf(incFile, "#ifndef OSQP_CONFIGURE_H\n");
  fprintf(incFile, "#define OSQP_CONFIGURE_H\n\n");

  /* Write out the algebra in-use */
  fprintf(incFile, "#define ALGEBRA_DEFAULT\n");
  fprintf(incFile, "#define OSQP_ALGEBRA \"default\"\n\n");

  /* Write out the embedded mode in use */
  fprintf(incFile, "#define EMBEDDED %d\n\n", defines->embedded_mode);

  /* Write out if printing is enabled */
  if (defines->printing_enable == 1) {
    fprintf(incFile, "#define PRINTING\n\n");
  }

  /* Write out if profiling is enabled*/
  if (defines->profiling_enable == 1) {
    fprintf(incFile, "#define PROFILING\n\n");
  }

  /* Write out if interrupts is enabled*/
  if (defines->interrupt_enable == 1) {
    fprintf(incFile, "#define CTRLC\n\n");
  }

  /* Write out the type of floating-point number to use */
  if (defines->float_type == 1) {
    fprintf(incFile, "#define DFLOAT\n\n");
  }

  /* The endif for the include-guard statement */
  fprintf(incFile, "#endif /* ifndef OSQP_CONFIGURE_H */\n");

  /* Close header file */
  fclose(incFile);
}


c_int codegen_example(const char *output_dir,
                      const char *file_prefix){

  char fname[PATH_LENGTH], cfname[PATH_LENGTH];
  FILE *srcFile;
  time_t now;

  sprintf(cfname, "%semosqp.c", output_dir);

  /* Open source file */
  srcFile = fopen(cfname, "w");
  if (srcFile == NULL)
    return osqp_error(OSQP_FOPEN_ERROR);

  /* Print comment headers containing the generation time into the files */
  time(&now);
  fprintf(srcFile, "/*\n");
  fprintf(srcFile, " * This file was autogenerated by OSQP on %s", ctime(&now));
  fprintf(srcFile, " * \n");
  fprintf(srcFile, " * This file contains a sample solver to run the embedded code.\n");
  fprintf(srcFile, " */\n\n");

  /* Include required headers */
  fprintf(srcFile, "#include <stdio.h>\n");
  fprintf(srcFile, "#include \"osqp.h\"\n");
  fprintf(srcFile, "#include \"%sworkspace.h\"\n\n", file_prefix);

  fprintf(srcFile, "int main() {\n");
  fprintf(srcFile, "  c_int exitflag;\n\n");
  fprintf(srcFile, "  printf( \"Embedded test program for vector updates.\\n\");\n\n");

  fprintf(srcFile, "  exitflag = osqp_solve( &%ssolver );\n\n", file_prefix);

  fprintf(srcFile, "  if( exitflag > 0 ) {\n");
  fprintf(srcFile, "    printf( \"  OSQP errored: %%s\\n\", osqp_error_message(exitflag));\n" );
  fprintf(srcFile, "    return (int)exitflag;\n");
  fprintf(srcFile, "  } else {\n");
  fprintf(srcFile, "    printf( \"  Solved workspace with no error.\\n\" );\n");
  fprintf(srcFile, "  }\n");
  fprintf(srcFile, "}\n");

  /* Close header file */
  fclose(srcFile);

  return 0;
}
