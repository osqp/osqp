#ifndef EXPORT_C_HELPERS_H_
#define EXPORT_C_HELPERS_H_

#include "error.h"
#include "types.h"

/* Define the maximum allowed length of a variable name */
#define MAX_VAR_LENGTH 255

/* Define the maximum allowed length of the path (directory + filename + extension) */
#define PATH_LENGTH 1024

/* Define the maximum allowed length of the filename (no extension)*/
#define FILE_LENGTH 100

#define PROPAGATE_ERROR(f) \
  exitflag = f; \
  if (exitflag) { return exitflag; }

#define GENERATE_ERROR(f) \
  exitflag = f; \
  if (exitflag) { return _osqp_error_line(exitflag, __FUNCTION__, __FILE__, __LINE__); }


/*
 * Write a raw float vector to a C array in a file.
 *
 * @param f    File handle to write to
 * @param vecf The vector to write
 * @param n    The length of the vector
 * @param name The name of the array to create
 * @return Error code
 */
OSQPInt write_vecf(FILE*            f,
                   const OSQPFloat* vecf,
                   OSQPInt          n,
                   const char*      name);

/*
 * Write a raw integer vector to a C array in a file.
 *
 * @param f    File handle to write to
 * @param veci The vector to write
 * @param n    The length of the vector
 * @param name The name of the array to create
 * @return Error code
 */
OSQPInt write_veci(FILE*          f,
                   const OSQPInt* veci,
                   OSQPInt        n,
                   const char*    name);

/*
 * Write a float vector to a C array in a file.
 *
 * @param f    File handle to write to
 * @param vec  The OSQP vector to write
 * @param name The name of the array to create
 * @return Error code
 */

OSQPInt write_OSQPVectorf(FILE*              f,
                          const OSQPVectorf* vec,
                          const char*        name);

/*
 * Write an integer vector to a C array in a file.
 *
 * @param f    File handle to write to
 * @param vec  The OSQP vector to write
 * @param name The name of the array to create
 * @return Error code
 */

OSQPInt write_OSQPVectori(FILE*              f,
                          const OSQPVectori* vec,
                          const char*        name);

/*
 * Write a CSC matrix to a struct in a C file.
 *
 * @param f    File handle to write to
 * @param M    The matrix to write
 * @param name The name of the array to create
 * @return Error code
 */
OSQPInt write_csc(FILE*                f,
                  const OSQPCscMatrix* M,
                  const char*          name);

#endif /* EXPORT_C_HELPERS_H_ */
