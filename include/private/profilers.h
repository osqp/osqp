#ifndef PROFILERS_H_
#define PROFILERS_H_

// cmake generated compiler flags
#include "osqp_configure.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Profiler section annotations that will be used by certain profilers.
 */
typedef enum {
    /* Level 1 detail (coarse) */
    OSQP_PROFILER_SEC_SETUP,                     /* Workspace setup */
    OSQP_PROFILER_SEC_SCALE,                     /* Problem scaling */
    OSQP_PROFILER_SEC_OPT_SOLVE,                 /* Solving optimization problem */
    OSQP_PROFILER_SEC_ADMM_ITER,                 /* Single ADMM iteration */
    OSQP_PROFILER_SEC_ADMM_KKT_SOLVE,            /* Solve KKT system */
    OSQP_PROFILER_SEC_ADMM_UPDATE,               /* Update the vectors in the ADMM */
    OSQP_PROFILER_SEC_ADMM_PROJ,                 /* Project vector */
    OSQP_PROFILER_SEC_POLISH,                    /* Solution polishing */

    /* Level 2 detail (more details) */
    OSQP_PROFILER_SEC_LINSYS_INIT,               /* Linsys: initialization */
    OSQP_PROFILER_SEC_LINSYS_SOLVE,              /* Linsys: solving */
    OSQP_PROFILER_SEC_LINSYS_SYM_FAC,            /* Linsys: symbolic factorization */
    OSQP_PROFILER_SEC_LINSYS_NUM_FAC,            /* Linsys: numerical factorization */
    OSQP_PROFILER_SEC_LINSYS_BACKSOLVE,          /* Linsys: backsolve */
    OSQP_PROFILER_SEC_LINSYS_MVM,                /* Linsys: matrix-vector multiply */

    /* Sentinel element */
    OSQP_PROFILER_SEC_ARRAY_LAST
} OSQPProfilerSection;


/*
 * Metadata for a profiler section annotation
 */
typedef struct {
    const char* name;       /* Visible name for the section */
    const char* desc;       /* Description of the section */
    int level;              /* Level the section is enabled at */
} OSQPProfilerSectionInfo;

extern OSQPProfilerSectionInfo osqp_profiler_sections[];


/*
 * Opaque objects containing the implementation-specific profiler section objects.
 * Initialized in @c osqp_profiler_init().
 */
typedef struct OSQPProfilerSection_ OSQPProfilerSections;
extern OSQPProfilerSections *osqp_profiler_sec_impl;

/**
 * Initialize the profiler annotations for level @c level.
 *
 * @param level is the level (0, 1, 2) of annotations to enable
 */
void _osqp_profiler_init(int level);

/**
 * Update the active profiler annotation level.
 *
 * @param level is the new level (0, 1, 2) of annotations to enable
 */
void _osqp_profiler_update_level(int level);

/**
 * Push a profiler section annotation onto the stack to show the code is in the section.
 *
 * @param section is the section to push
 */
void _osqp_profiler_sec_push(OSQPProfilerSection section);

/**
 * Pop the most recent profiler section off the stack (when leaving the section).
 */
void _osqp_profiler_sec_pop(OSQPProfilerSection section);


/*
 * Allow disabling the profiler annotations completely with no overhead by just ignoring the call.
 */
#ifdef OSQP_ENABLE_PROFILER_ANNOTATIONS
#define osqp_profiler_init(level)         _osqp_profiler_init(level)
#define osqp_profiler_update_level(level) _osqp_profiler_update_level(level)
#define osqp_profiler_sec_push(sec)       _osqp_profiler_sec_push(sec)
#define osqp_profiler_sec_pop(sec)        _osqp_profiler_sec_pop(sec)
#else
#define osqp_profiler_init(level)
#define osqp_profiler_update_level(level)
#define osqp_profiler_sec_push(sec)
#define osqp_profiler_sec_pop(sec)
#endif


#ifdef __cplusplus
}
#endif

#endif /* PROFILERS_H_ */
