#ifndef ERROR_H
#define ERROR_H

/* OSQP error handling */

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

# include "osqp.h"


/* OSQP error macro */
#  define osqp_error(error_code) _osqp_error(error_code, __FUNCTION__);


/**
 * Internal function to print error description and return error code.
 * @param  Error code
 * @param  Function name
 * @return Error code
 */
  c_int _osqp_error(enum osqp_error_type error_code,
		    const char * function_name);



# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef ERROR_H
