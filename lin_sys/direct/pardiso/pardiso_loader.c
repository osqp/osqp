#include "osqp.h"
#include "mkl_pardiso.h"

// Wrappers for loaded Pardiso function handlers
void mypardiso(void** pt, const c_int* maxfct, const c_int* mnum,
                  const c_int* mtype, const c_int* phase, const c_int* n,
                  const c_float* a, const c_int* ia, const c_int* ja,
                  c_int* perm, const c_int* nrhs, c_int* iparm,
                  const c_int* msglvl, c_float* b, c_float* x,
                  c_int* error) {
    PARDISO (pt, maxfct, mnum, mtype, phase,
             n, a, ia, ja, perm, nrhs, iparm, msglvl, b, x, error);
}
