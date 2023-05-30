#ifndef ERROR_H
#define ERROR_H

/* OSQP error handling */

#include "osqp.h"

#ifdef __cplusplus
extern "C" {
#endif

/* OSQP error macro */
#if __STDC_VERSION__ >= 199901L
/* The C99 standard gives the __func__ macro, which is preferred over __FUNCTION__ */
#  define osqp_error(error_code) _osqp_error(error_code, __func__);
#else
#  define osqp_error(error_code) _osqp_error(error_code, __FUNCTION__);
#endif



/**
 * Internal function to print error description and return error code.
 * @param  error_code    Error code
 * @param  function_name Function name
 * @return               Error code
 */
OSQPInt _osqp_error(enum osqp_error_type  error_code,
                    const char*           function_name);

/**
 * Internal function to print error description, location and return error code.
 * @param  error_code    Error code
 * @param  function_name Function name
 * @param  filename      Filename
 * @param  line_number   Line number of error
 * @return               Error code
 */
OSQPInt _osqp_error_line(enum osqp_error_type error_code,
                         const char*          function_name,
                         const char*          filename,
                         OSQPInt              line_number);

#ifdef __cplusplus
}
#endif

#endif /* ifndef ERROR_H */
