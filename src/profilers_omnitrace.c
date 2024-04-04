
#include <omnitrace/categories.h>
#include <omnitrace/types.h>
#include <omnitrace/user.h>

#include "types.h"
#include "profilers.h"


struct OSQPProfilerSection_ {
    int sec;
    int enabled;
};

struct OSQPProfilerSection_ osqp_profiler_sec_impl[] = {
    /* Level 1 detail (coarse) */
    {OSQP_PROFILER_SEC_SETUP,          0},
    {OSQP_PROFILER_SEC_SCALE,          0},
    {OSQP_PROFILER_SEC_OPT_SOLVE,      0},
    {OSQP_PROFILER_SEC_ADMM_ITER,      0},
    {OSQP_PROFILER_SEC_ADMM_KKT_SOLVE, 0},
    {OSQP_PROFILER_SEC_ADMM_UPDATE,    0},
    {OSQP_PROFILER_SEC_ADMM_PROJ,      0},
    {OSQP_PROFILER_SEC_POLISH,         0},

    /* Level 2 detail (more details) */
    {OSQP_PROFILER_SEC_LINSYS_INIT,      0},
    {OSQP_PROFILER_SEC_LINSYS_SOLVE,     0},
    {OSQP_PROFILER_SEC_LINSYS_SYM_FAC,   0},
    {OSQP_PROFILER_SEC_LINSYS_NUM_FAC,   0},
    {OSQP_PROFILER_SEC_LINSYS_BACKSOLVE, 0},
    {OSQP_PROFILER_SEC_LINSYS_MVM,       0}
};

struct OSQPProfilerEvent_ {
    int sec;
    int enabled;
};

struct OSQPProfilerEvent_ osqp_profiler_event_impl[] = {
    /* Level 1 detail (coarse) */
    {OSQP_PROFILER_EVENT_RHO_UPDATE,     0}
};

void _osqp_profiler_init(int level) {
    
    for(OSQPInt i=0; i < OSQP_PROFILER_SEC_ARRAY_LAST; i++) {
        osqp_profiler_sec_impl[i].enabled = (osqp_profiler_sections[i].level <= level);
    }

    for(OSQPInt i=0; i < OSQP_PROFILER_EVENT_ARRAY_LAST; i++) {
        osqp_profiler_event_impl[i].enabled = (osqp_profiler_events[i].level <= level);
    }
}


void _osqp_profiler_update_level(int level) {
    for(OSQPInt i=0; i < OSQP_PROFILER_SEC_ARRAY_LAST; i++) {
        osqp_profiler_sec_impl[i].enabled = (osqp_profiler_sections[i].level <= level);
    }

    for(OSQPInt i=0; i < OSQP_PROFILER_EVENT_ARRAY_LAST; i++) {
        osqp_profiler_event_impl[i].enabled = (osqp_profiler_events[i].level <= level);
    }
}

void _osqp_profiler_sec_push(OSQPProfilerSection section) {
    // Don't push a section that isn't enabled
    if(osqp_profiler_sec_impl[section].enabled == 0)
        return;
    
    omnitrace_user_push_region(osqp_profiler_sections[section].desc);
}


void _osqp_profiler_sec_pop(OSQPProfilerSection section) {
    // Don't pop a section that isn't enabled
    if(osqp_profiler_sec_impl[section].enabled == 0)
        return;

    omnitrace_user_pop_region(osqp_profiler_sections[section].desc);
}


void _osqp_profiler_event_mark(OSQPProfilerEvent event) {
    // Not implemented on omnitrace
}