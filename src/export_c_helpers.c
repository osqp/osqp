#include <stdio.h>

#include "error.h"
#include "export_c_helpers.h"
#include "osqp_api_constants.h"
#include "types.h"

#include "algebra_impl.h"


/*********
* Vectors
**********/

OSQPInt write_vecf(FILE*            f,
                   const OSQPFloat* vecf,
                   OSQPInt          n,
                   const char*      name) {

  OSQPInt i;

  if (n && vecf) {
    fprintf(f, "OSQPFloat %s[%d] = {\n", name, n);
    for (i = 0; i < n; i++) {
      fprintf(f, "  (OSQPFloat)%.20f,\n", vecf[i]);
    }
    fprintf(f, "};\n");
  }
  else {
    fprintf(f, "#define %s (OSQP_NULL)\n", name);
  }

  return OSQP_NO_ERROR;
}

OSQPInt write_veci(FILE*          f,
                   const OSQPInt* veci,
                   OSQPInt        n,
                   const char*    name) {

  OSQPInt i;

  if (n && veci) {
    fprintf(f, "OSQPInt %s[%d] = {\n", name, n);
    for (i = 0; i < n; i++) {
      fprintf(f, "  %i,\n", veci[i]);
    }
    fprintf(f, "};\n");
  }
  else {
    fprintf(f, "#define %s (OSQP_NULL)\n", name);
  }

  return OSQP_NO_ERROR;
}

OSQPInt write_OSQPVectorf(FILE*              f,
                          const OSQPVectorf* vec,
                          const char*        name) {
  
  OSQPInt exitflag = OSQP_NO_ERROR;
  char vecf_name[MAX_VAR_LENGTH];

  if (!vec) return OSQP_DATA_NOT_INITIALIZED;

  sprintf(vecf_name, "%s_val", name);
  PROPAGATE_ERROR(write_vecf(f, vec->values, vec->length, vecf_name))
  fprintf(f, "OSQPVectorf %s = {\n  %s,\n  %d\n};\n", name, vecf_name, vec->length);

  return exitflag;
}

OSQPInt write_OSQPVectori(FILE*              f,
                          const OSQPVectori* vec,
                          const char*        name) {
  
  OSQPInt exitflag = OSQP_NO_ERROR;
  char veci_name[MAX_VAR_LENGTH];

  if (!vec) return OSQP_DATA_NOT_INITIALIZED;

  sprintf(veci_name, "%s_val", name);
  PROPAGATE_ERROR(write_veci(f, vec->values, vec->length, veci_name))
  fprintf(f, "OSQPVectori %s = {\n  %s,\n  %d\n};\n", name, veci_name, vec->length);

  return exitflag;
}


/*********
* CSC Matrix
**********/

OSQPInt write_csc(FILE*                f,
                  const OSQPCscMatrix* M,
                  const char*          name) {

  OSQPInt exitflag = OSQP_NO_ERROR;
  char vec_name[MAX_VAR_LENGTH];

  if (!M) return OSQP_DATA_NOT_INITIALIZED;

  sprintf(vec_name, "%s_p", name);
  PROPAGATE_ERROR(write_veci(f, M->p, M->n+1, vec_name))
  sprintf(vec_name, "%s_i", name);
  PROPAGATE_ERROR(write_veci(f, M->i, M->nzmax, vec_name))
  sprintf(vec_name, "%s_x", name);
  PROPAGATE_ERROR(write_vecf(f, M->x, M->nzmax, vec_name))
  fprintf(f, "OSQPCscMatrix %s = {\n", name);
  fprintf(f, "  %d,\n", M->m);
  fprintf(f, "  %d,\n", M->n);
  fprintf(f, "  %s_p,\n", name);
  fprintf(f, "  %s_i,\n", name);
  fprintf(f, "  %s_x,\n", name);
  fprintf(f, "  %d,\n", M->nzmax);
  fprintf(f, "  %d,\n", M->nz);
  fprintf(f, "};\n");

  return exitflag;
}
