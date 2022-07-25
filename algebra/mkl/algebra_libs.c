#include "osqp_api_types.h"
#include <mkl.h>

c_int osqp_algebra_init_libs(void) {
    c_int retval = 0;

#ifdef DLONG
    retval = mkl_set_interface_layer(MKL_INTERFACE_ILP64);
#else
    retval = mkl_set_interface_layer(MKL_INTERFACE_LP64);
#endif

    // Positive value is the interface chosen, so -1 is the error condition
    if(retval == -1)
        return 1;

    return 0;
}

void osqp_algebra_free_libs(void) {return;}
