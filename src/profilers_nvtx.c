
#include <nvtx3/nvToolsExt.h>

#include "types.h"
#include "profilers.h"


struct OSQPProfilerItem_ {
    int sec;
    int enabled;
    nvtxEventAttributes_t attr;
};

#define OSQPProfilerSection_ OSQPProfilerItem_
#define OSQPProfilerEvent_   OSQPProfilerItem_

#define PROFILER_SEC_IMPL(name)    {name, 0, {0}}
#define PROFILER_EVENT_IMPL(name)  {name, 0, {0}}

#include "profilers_impl.c"

/* Global domain to namespace the OSQP profiling from other markers */
static nvtxDomainHandle_t s_osqp_nvtx_domain;

void _osqp_profiler_init(int level) {
    // Create the OSQP domain for profiling
    s_osqp_nvtx_domain = nvtxDomainCreateA("osqp");

    const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
    const int num_colors = sizeof(colors)/sizeof(uint32_t);


    for(OSQPInt i=0; i < OSQP_PROFILER_SEC_ARRAY_LAST; i++) {
        osqp_profiler_sec_impl[i].enabled = (osqp_profiler_sections[i].level <= level);

        osqp_profiler_sec_impl[i].attr.version = NVTX_VERSION;
        osqp_profiler_sec_impl[i].attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        osqp_profiler_sec_impl[i].attr.colorType = NVTX_COLOR_ARGB;
        osqp_profiler_sec_impl[i].attr.color = colors[i%num_colors];
        osqp_profiler_sec_impl[i].attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
        osqp_profiler_sec_impl[i].attr.message.ascii = osqp_profiler_sections[i].desc;
    }

    for(OSQPInt i=0; i < OSQP_PROFILER_EVENT_ARRAY_LAST; i++) {
        osqp_profiler_event_impl[i].enabled = (osqp_profiler_events[i].level <= level);

        osqp_profiler_event_impl[i].attr.version = NVTX_VERSION;
        osqp_profiler_event_impl[i].attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        osqp_profiler_event_impl[i].attr.colorType = NVTX_COLOR_ARGB;
        osqp_profiler_event_impl[i].attr.color = colors[i%num_colors];
        osqp_profiler_event_impl[i].attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
        osqp_profiler_event_impl[i].attr.message.ascii = osqp_profiler_events[i].desc;
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

    nvtxDomainRangePushEx(s_osqp_nvtx_domain, &(osqp_profiler_sec_impl[section].attr));
}


void _osqp_profiler_sec_pop(OSQPProfilerSection section) {
    // Don't pop a section that isn't enabled
    if(osqp_profiler_sec_impl[section].enabled == 0)
        return;

    nvtxDomainRangePop(s_osqp_nvtx_domain);
}


void _osqp_profiler_event_mark(OSQPProfilerEvent event) {
    // Don't record an event that isn't enabled
    if(osqp_profiler_event_impl[event].enabled == 0)
        return;

    nvtxDomainMarkEx(s_osqp_nvtx_domain, &(osqp_profiler_event_impl[event].attr));
}