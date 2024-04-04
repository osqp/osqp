#ifndef CODEGEN_H
#define CODEGEN_H

#include "osqp_api_types.h"

#ifdef __cplusplus
extern "C" {
#endif

OSQPInt codegen_inc(const char* output_dir,
                    const char* file_prefix);

OSQPInt codegen_src(const char* output_dir,
                    const char* file_prefix,
                    OSQPSolver* solver,
                    OSQPInt     embedded);

OSQPInt codegen_defines(const char*         output_dir,
                        OSQPCodegenDefines* defines);

OSQPInt codegen_example(const char* output_dir,
                        const char* file_prefix);

#ifdef __cplusplus
}
#endif

#endif /* ifndef CODEGEN_H */
