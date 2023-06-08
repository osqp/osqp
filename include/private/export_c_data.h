#ifndef EXPORT_C_DATA_H_
#define EXPORT_C_DATA_H_

#include "osqp.h"

/*
 * Export the data from the OSQP solver as a C data structure.
 *
 * @param solver     The solver
 * @param output_dir The output directory
 * @param file_name  The filename (without extension)
 * @return Error code
 */
OSQPInt export_c_data(OSQPSolver* solver,
                      const char* output_dir,
                      const char* file_name);

#endif /* EXPORT_C_DATA_H_ */
