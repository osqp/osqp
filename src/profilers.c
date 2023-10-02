
#include "profilers.h"


OSQPProfilerItemInfo osqp_profiler_sections[] = {
    /* Level 1 detail (coarse) */
    {"prob_setup",      "Problem setup",                      1}, /* OSQP_PROFILER_SEC_SETUP */
    {"prob_scale",      "Problem data scaling",               1}, /* OSQP_PROFILER_SEC_SCALE */
    {"solve_opt_prob",  "Solving optimization problem",       1}, /* OSQP_PROFILER_SEC_OPT_SOLVE */
    {"admm_iter",       "ADMM iteration",                     1}, /* OSQP_PROFILER_SEC_ADMM_ITER */
    {"admm_kkt_solve",  "KKT system solve in ADMM iteration", 1}, /* OSQP_PROFILER_SEC_ADMM_KKT_SOLVE */
    {"admm_vec_update", "Vector updates in ADMM iteration",   1}, /* OSQP_PROFILER_SEC_ADMM_UPDATE */
    {"admm_project",    "Projection in ADMM iteration",       1}, /* OSQP_PROFILER_SEC_ADMM_PROJ */
    {"sol_polish",      "Solution polishing",                 1}, /* OSQP_PROFILER_SEC_POLISH */

    /* Level 2 detail (more details) */
    {"linsys_init",      "Initialize linear system solver",         2}, /* OSQP_PROFILER_SEC_LINSYS_INIT */
    {"linsys_solve",     "Solve the linear system",                 2}, /* OSQP_PROFILER_SEC_LINSYS_SOLVE */
    {"linsys_sym_fac",   "Symbolic factorization in direct solver", 2}, /* OSQP_PROFILER_SEC_LINSYS_SYM_FAC */
    {"linsys_num_fac",   "Numeric factorization in direct solver",  2}, /* OSQP_PROFILER_SEC_LINSYS_NUM_FAC */
    {"linsys_backsolve", "Backsolve in direct solver",              2}, /* OSQP_PROFILER_SEC_LINSYS_BACKSOLVE */
    {"linsys_mvm",       "Matrix-vector multiplication",            2}  /* OSQP_PROFILER_SEC_LINSYS_MVM */
};

OSQPProfilerItemInfo osqp_profiler_events[] = {
    /* Level 1 details (coarse) */
    {"rho_update", "Rho update", 1} /* OSQP_PROFILER_EVENT_RHO_UPDATE */
};