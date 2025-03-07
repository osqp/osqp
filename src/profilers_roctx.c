
#include <roctx.h>

#include "types.h"
#include "profilers.h"


struct OSQPProfilerSection_ {
    int sec;
    int enabled;
};

struct OSQPProfilerEvent_ {
    int sec;
    int enabled;
};

#define PROFILER_SEC_IMPL(name)    {name, 0}
#define PROFILER_EVENT_IMPL(name)  {name, 0}

#include "profilers_impl.c"

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

    roctxRangePush(osqp_profiler_sections[section].desc);
}


void _osqp_profiler_sec_pop(OSQPProfilerSection section) {
    // Don't pop a section that isn't enabled
    if(osqp_profiler_sec_impl[section].enabled == 0)
        return;

    roctxRangePop();
}


void _osqp_profiler_event_mark(OSQPProfilerEvent event) {
    // Don't record an event that isn't enabled
    if(osqp_profiler_event_impl[event].enabled == 0)
        return;

    roctxMark(osqp_profiler_events[event].desc);
}