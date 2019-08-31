#ifndef ERROR_H
# define ERROR_H

/* OSQP error handling */

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

# include "types.h"


/* OSQP error macro */
# if __STDC_VERSION__ >= 199901L
/* The C99 standard gives the __func__ macro, which is preferred over __FUNCTION__ */
#  define osqp_error(error_code) _osqp_error(error_code, __func__);
#else
#  define osqp_error(error_code) _osqp_error(error_code, __FUNCTION__);
#endif



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
