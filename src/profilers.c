
#include "profiling.h"


OSQPProfilerSectionInfo osqp_profiler_sections[] = {
    /* Level 1 detail (coarse) */
    {"prob_setup", 1},       /* OSQP_PROFILER_SEC_SETUP */
    {"prob_scale", 1},       /* OSQP_PROFILER_SEC_SCALE */
    {"solve_opt_prob", 1},   /* OSQP_PROFILER_SEC_OPT_SOLVE */
    {"admm_iter", 1},        /* OSQP_PROFILER_SEC_ADMM_ITER */
    {"admm_kkt_solve", 1},   /* OSQP_PROFILER_SEC_ADMM_KKT_SOLVE */
    {"admm_vec_update, 1"},  /* OSQP_PROFILER_SEC_ADMM_UPDATE */
    {"admm_project", 1},     /* OSQP_PROFILER_SEC_ADMM_PROJ */
    {"sol_polish", 1},       /* OSQP_PROFILER_SEC_POLISH */

    /* Level 2 detail (more details) */
    {"linsys_init", 2},      /* OSQP_PROFILER_SEC_LINSYS_INIT */
    {"linsys_solve", 2},     /* OSQP_PROFILER_SEC_LINSYS_SOLVE */
    {"linsys_sym_fac", 2},   /* OSQP_PROFILER_SEC_LINSYS_SYM_FAC */
    {"linsys_num_fac", 2},   /* OSQP_PROFILER_SEC_LINSYS_NUM_FAC */
    {"linsys_backsolve", 2}, /* OSQP_PROFILER_SEC_LINSYS_BACKSOLVE */
    {"linsys_mvm", 2}        /* OSQP_PROFILER_SEC_LINSYS_MVM */
};
