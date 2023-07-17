/* Header file defining printing functions */
#ifndef PRINTING_H_
#define PRINTING_H_

/* cmake generated compiler flags */
#include "osqp_configure.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Format specifier to use for the OSQP integers */
# ifdef OSQP_USE_LONG            /* Long integers */
#define OSQP_INT_FMT "lld"
# else                           /* Standard integers */
#define OSQP_INT_FMT "d"
# endif

/* Error printing function */
/* Always define this, and let implementations undefine if they want to change it */
# if __STDC_VERSION__ >= 199901L
/* The C99 standard gives the __func__ macro, which is preferred over __FUNCTION__ */
# define c_eprint(...) c_print("ERROR in %s: ", __func__); \
         c_print(__VA_ARGS__); c_print("\n");
#else
# define c_eprint(...) c_print("ERROR in %s: ", __FUNCTION__); \
         c_print(__VA_ARGS__); c_print("\n");
#endif

#ifdef OSQP_CUSTOM_PRINTING
/* Use user-provided printing functions */
# include OSQP_CUSTOM_PRINTING

#elif defined(OSQP_ENABLE_PRINTING)
/* Use standard printing routine */
# include <stdio.h>
# include <string.h>

# define c_print printf

#else
/* No printing is desired, so turn the two functions into NOPs */
# undef c_eprint
# define c_print(...)
# define c_eprint(...)

#endif  /* OSQP_CUSTOM_PRINTING */

#ifdef __cplusplus
}
#endif

#endif /* PRINTING_H_ */
