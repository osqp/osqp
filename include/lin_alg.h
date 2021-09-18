#ifndef LIN_ALG_H
#define LIN_ALG_H


# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

# include "algebra_vector.h"
# include "algebra_matrix.h"


/* Initialize libraries that implement algebra. */
c_int osqp_algebra_init_libs(void);

/* Free libraries that implement algebra. */
void osqp_algebra_free_libs(void);


# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef LIN_ALG_H
