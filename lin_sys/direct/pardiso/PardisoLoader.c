#include "LibraryHandler.h"
#include "PardisoLoader.h"

#define PARDISOLIBNAME "mkl_rt." SHAREDLIBEXT
typedef void (*voidfun)(void);

voidfun LSL_loadSym (soHandle_t h, const char *symName);


// Pardiso function interfaces
typedef void (*pardiso_t)(void**, const c_int*, const c_int*, const c_int*,
                          const c_int*, const c_int*, const c_float*,
                          const c_int*, const c_int*, const c_int*,
                          const c_int*, c_int*, const c_int*, c_float*,
                          c_float*, const c_int*);
typedef int (*mkl_set_ifl_t)(int);

static soHandle_t Pardiso_handle = OSQP_NULL;
static pardiso_t func_pardiso = OSQP_NULL;
static mkl_set_ifl_t func_mkl_set_interface_layer = OSQP_NULL;


// Wrappers for loaded Pardiso functions
void pardiso(void** pt, const c_int* maxfct, const c_int* mnum,
                  const c_int* mtype, const c_int* phase, const c_int* n,
                  const c_float* a, const c_int* ia, const c_int* ja,
                  const c_int* perm, const c_int* nrhs, c_int* iparm,
                  const c_int* msglvl, c_float* b, c_float* x,
                  const c_int* error) {
    func_pardiso(pt, maxfct, mnum, mtype, phase, n, a, ia, ja,
                 perm, nrhs, iparm, msglvl, b, x, error);
}
c_int mkl_set_interface_layer(c_int code) {
    return (c_int)func_mkl_set_interface_layer((int)code);
}



int LSL_loadPardisoLib(const char* libname) {
    // Load Pardiso library
    if (libname) {
        Pardiso_handle = LSL_loadLib(libname);
    } else { /* try a default library name */
        Pardiso_handle = LSL_loadLib(PARDISOLIBNAME);
    }
    if (!Pardiso_handle) return 1;

    // Load Pardiso functions
    func_pardiso = (pardiso_t)LSL_loadSym(Pardiso_handle, "pardiso");
    if (!func_pardiso) return 1;

    func_mkl_set_interface_layer = (mkl_set_ifl_t)LSL_loadSym(Pardiso_handle,
                                                    "MKL_Set_Interface_Layer");
    if (!func_mkl_set_interface_layer) return 1;

    return 0;
}

int LSL_unloadPardisoLib() {
    int rc;

    if (Pardiso_handle == OSQP_NULL)
      return 0;

    rc = LSL_unloadLib(Pardiso_handle);
    Pardiso_handle = OSQP_NULL;
    func_pardiso = OSQP_NULL;

    return rc;
}
