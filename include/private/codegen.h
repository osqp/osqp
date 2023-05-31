#ifndef CODEGEN_H
#define CODEGEN_H

#include "osqp_api_types.h"

#ifdef __cplusplus
extern "C" {
#endif

OSQPInt codegen_inc(OSQPSolver* solver,
                    const char* output_dir,
                    const char* file_prefix);

OSQPInt codegen_src(OSQPSolver* solver,
                    const char* output_dir,
                    const char* file_prefix,
                    OSQPInt     embedded);

OSQPInt codegen_defines(const char*         output_dir,
                        OSQPCodegenDefines* defines);

OSQPInt codegen_example(const char* output_dir,
                        const char* file_prefix);

#ifdef __cplusplus
}
#endif

#endif /* ifndef CODEGEN_H */
