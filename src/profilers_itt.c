
#include <ittnotify.h>
#include <string.h>

#include "types.h"
#include "profilers.h"


struct OSQPProfilerSection_ {
    int sec;
    int enabled;
    __itt_string_handle* taskHandle;
};

struct OSQPProfilerSection_ osqp_profiler_sec_impl[] = {
    /* Level 1 detail (coarse) */
    {OSQP_PROFILER_SEC_SETUP,          0, OSQP_NULL},
    {OSQP_PROFILER_SEC_SCALE,          0, OSQP_NULL},
    {OSQP_PROFILER_SEC_OPT_SOLVE,      0, OSQP_NULL},
    {OSQP_PROFILER_SEC_ADMM_ITER,      0, OSQP_NULL},
    {OSQP_PROFILER_SEC_ADMM_KKT_SOLVE, 0, OSQP_NULL},
    {OSQP_PROFILER_SEC_ADMM_UPDATE,    0, OSQP_NULL},
    {OSQP_PROFILER_SEC_ADMM_PROJ,      0, OSQP_NULL},
    {OSQP_PROFILER_SEC_POLISH,         0, OSQP_NULL},

    /* Level 2 detail (more details) */
    {OSQP_PROFILER_SEC_LINSYS_INIT,      0, OSQP_NULL},
    {OSQP_PROFILER_SEC_LINSYS_SOLVE,     0, OSQP_NULL},
    {OSQP_PROFILER_SEC_LINSYS_SYM_FAC,   0, OSQP_NULL},
    {OSQP_PROFILER_SEC_LINSYS_NUM_FAC,   0, OSQP_NULL},
    {OSQP_PROFILER_SEC_LINSYS_BACKSOLVE, 0, OSQP_NULL},
    {OSQP_PROFILER_SEC_LINSYS_MVM,       0, OSQP_NULL}
};

struct OSQPProfilerEvent_ {
    int sec;
    int enabled;
    __itt_event eventHandle;
};

struct OSQPProfilerEvent_ osqp_profiler_event_impl[] = {
    /* Level 1 detail (coarse) */
    {OSQP_PROFILER_EVENT_RHO_UPDATE,     0, 0}
};

/* Global domain to namespace the OSQP profiling from other markers */
static __itt_domain* s_osqp_itt_domain;

void _osqp_profiler_init(int level) {
    // Create the OSQP domain for profiling
    s_osqp_itt_domain = __itt_domain_create("osqp");

    const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
    const int num_colors = sizeof(colors)/sizeof(uint32_t);

    for(OSQPInt i=0; i < OSQP_PROFILER_SEC_ARRAY_LAST; i++) {
        osqp_profiler_sec_impl[i].enabled = (osqp_profiler_sections[i].level <= level);
        osqp_profiler_sec_impl[i].taskHandle = __itt_string_handle_create(osqp_profiler_sections[i].desc);
    }

    for(OSQPInt i=0; i < OSQP_PROFILER_EVENT_ARRAY_LAST; i++) {
        osqp_profiler_event_impl[i].enabled = (osqp_profiler_events[i].level <= level);
        osqp_profiler_event_impl[i].eventHandle = __itt_event_create(osqp_profiler_events[i].desc,
                                                                     strlen(osqp_profiler_events[i].desc));
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
    
    __itt_task_begin(s_osqp_itt_domain, __itt_null, __itt_null, osqp_profiler_sec_impl[section].taskHandle);
}


void _osqp_profiler_sec_pop(OSQPProfilerSection section) {
    // Don't pop a section that isn't enabled
    if(osqp_profiler_sec_impl[section].enabled == 0)
        return;

    __itt_task_end(s_osqp_itt_domain);
}


void _osqp_profiler_event_mark(OSQPProfilerEvent event) {
    // Don't record an event that isn't enabled
    if(osqp_profiler_event_impl[event].enabled == 0)
        return;

    __itt_event_start(osqp_profiler_event_impl[event].eventHandle);
}