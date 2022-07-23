#ifndef LIN_ALG_H
#define LIN_ALG_H

# include "algebra_vector.h"
# include "algebra_matrix.h"

# ifdef __cplusplus
extern "C" {
# endif

/* Return which linear system solvers are supported */
c_int osqp_algebra_linsys_supported(void);

/* Return the default linear system the algebra backend prefers */
enum osqp_linsys_solver_type osqp_algebra_default_linsys(void);

/* Initialize libraries that implement algebra. */
c_int osqp_algebra_init_libs(c_int device);

/* Free libraries that implement algebra. */
void osqp_algebra_free_libs(void);


# ifdef __cplusplus
}
# endif

#endif /* ifndef LIN_ALG_H */
