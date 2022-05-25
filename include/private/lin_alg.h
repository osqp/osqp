#ifndef LIN_ALG_H
#define LIN_ALG_H

# include "algebra_vector.h"
# include "algebra_matrix.h"

# ifdef __cplusplus
extern "C" {
# endif

/* Initialize libraries that implement algebra. */
c_int osqp_algebra_init_libs(c_int device);

/* Free libraries that implement algebra. */
void osqp_algebra_free_libs(void);


# ifdef __cplusplus
}
# endif

#endif /* ifndef LIN_ALG_H */
