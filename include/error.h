#ifndef ERROR_H
# define ERROR_H

/* OSQP error handling */

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

# include "types.h"


/**
 * Print error description and return error code
 * @param  Error code
 * @return Error code.
 */
c_int osqp_error(enum osqp_error_type error_code);


# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef ERROR_H
