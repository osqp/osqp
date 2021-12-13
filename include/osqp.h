#ifndef OSQP_H
#define OSQP_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

/* Types, functions etc required by the OSQP API */
# include "csc_type.h"
# include "osqp_configure.h"
# include "osqp_api_constants.h"
# include "osqp_api_types.h"
# include "osqp_api_functions.h"

#ifndef EMBEDDED
# include "osqp_api_utils.h"
#endif

# ifdef __cplusplus
}
# endif /* ifdef __cplusplus */

#endif /* ifndef OSQP_H */
