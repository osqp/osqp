#include "osqp.h"
#include "algebra_vector.h"
#include "algebra_impl.h"

#include <GraphBLAS.h>

/* VECTOR FUNCTIONS ----------------------------------------------------------*/

OSQPVectorf* OSQPVectorf_new(const c_float *a,
                             c_int          length){
  OSQPVectorf* out = OSQPVectorf_malloc(length);

  if(!out) return OSQP_NULL;

  out->length = length;

  if (length > 0) {
    OSQPVectorf_from_raw(out, a);
  }

  return out;
}

OSQPVectori* OSQPVectori_new(const c_int *a,
                             c_int       length) {
  OSQPVectori* out = OSQPVectori_malloc(length);

  if(!out) return OSQP_NULL;

  out->length = length;

  if (length > 0) {
    OSQPVectori_from_raw(out, a);
  }
  return out;
}

OSQPVectorf* OSQPVectorf_malloc(c_int length) {
  OSQPVectorf *out = c_malloc(sizeof(OSQPVectorf));

  if (!out) return OSQP_NULL;

  out->length = length;

  if (GrB_Vector_new(&(out->vec), GRFLOAT, length) != GrB_SUCCESS) {
    c_free(out);
    return OSQP_NULL;
  }

  return out;
}

OSQPVectori* OSQPVectori_malloc(c_int length) {
  OSQPVectori *out = c_malloc(sizeof(OSQPVectori));

  if (!out) return OSQP_NULL;

  out->length = length;

  if (GrB_Vector_new(&(out->vec), GRINT, length) != GrB_SUCCESS) {
    c_free(out);
    return OSQP_NULL;
  }

  return out;
}

OSQPVectorf* OSQPVectorf_calloc(c_int length) {
  c_int retval = 0;
  OSQPVectorf* out = OSQPVectorf_malloc(length);

  if(!out) return OSQP_NULL;

  /* Assign all entries 0.0 using the constant vector variant of the method */
  retval = GrB_assign(out->vec,  /* w = Vector to operate on */
                      GrB_NULL,  /* mask = Don't prevent writing to any elements of w */
                      GrB_NULL,  /* accum = Don't accumulate into w */
                      0.0,       /* val = Assign 0.0 to the element */
                      GrB_ALL,   /* indices = Assign to all indices */
                      length,    /* nindices = The total length of the vector */
                      GrB_NULL); /* desc = Option descriptor for the operation (TODO: Use this for performance) */

  if (retval != GrB_SUCCESS) {
    GrB_free(&(out->vec));
    c_free(out);
    return OSQP_NULL;
  }

  return out;
}

OSQPVectori* OSQPVectori_calloc(c_int length) {
  c_int retval = 0;
  OSQPVectori* out = OSQPVectori_malloc(length);

  if(!out) return OSQP_NULL;

  /* Assign all entries 0 using the constant vector variant of the method */
  retval = GrB_assign(out->vec,  /* w = Vector to operate on */
                      GrB_NULL,  /* mask = Don't prevent writing to any elements of w */
                      GrB_NULL,  /* accum = Don't accumulate into w */
                      0,         /* val = Assign 0 to the element */
                      GrB_ALL,   /* indices = Assign to all indices */
                      length,    /* nindices = The total length of the vector */
                      GrB_NULL); /* desc = Option descriptor for the operation (TODO: Use this for performance) */

  if (retval != GrB_SUCCESS) {
    GrB_free(&(out->vec));
    c_free(out);
    return OSQP_NULL;
  }

  return out;
}

OSQPVectorf* OSQPVectorf_copy_new(const OSQPVectorf *a) {
  OSQPVectorf *out = c_malloc(sizeof(OSQPVectorf));

  if (!out) return OSQP_NULL;

  out->length = a->length;

  if (GrB_Vector_dup(&(out->vec), a->vec) != GrB_SUCCESS) {
    c_free(out);
    return OSQP_NULL;
  }

  return out;
}

OSQPVectori* OSQPVectori_copy_new(const OSQPVectori *a) {
  OSQPVectori *out = c_malloc(sizeof(OSQPVectori));

  if (!out) return OSQP_NULL;

  out->length = a->length;

  if (GrB_Vector_dup(&(out->vec), a->vec) != GrB_SUCCESS) {
    c_free(out);
    return OSQP_NULL;
  }

  return out;
}

void OSQPVectorf_free(OSQPVectorf *a){
  if (a) GrB_free(&(a->vec));
  c_free(a);
}

void OSQPVectori_free(OSQPVectori *a){
  if (a) GrB_free(&(a->vec));
  c_free(a);
}

/* TODO */
OSQPVectorf* OSQPVectorf_view(const OSQPVectorf *a,
                              c_int              head,
                              c_int              length) {
  OSQPVectorf* view = c_malloc(sizeof(OSQPVectorf));
  if (view) {
      OSQPVectorf_view_update(view, a, head, length);
  }
  return view;
}

/* TODO */
void OSQPVectorf_view_update(OSQPVectorf       *a,
                             const OSQPVectorf *b,
                             c_int             head,
                             c_int             length) {
}

/* TODO */
void OSQPVectorf_view_free(OSQPVectorf *a){
}

c_int OSQPVectorf_length(const OSQPVectorf *a) {
  GrB_Index size;

  GrB_Vector_size(&size, a->vec);
  return size;
}

c_int OSQPVectori_length(const OSQPVectori *a) {
  GrB_Index size;

  GrB_Vector_size(&size, a->vec);
  return size;
}

/* Pointer to vector data (floats) */
/* TODO */
c_float* OSQPVectorf_data(const OSQPVectorf *a){return 0;}

/* TODO */
c_int*   OSQPVectori_data(const OSQPVectori *a){return 0;}

void OSQPVectorf_copy(OSQPVectorf       *b,
                      const OSQPVectorf *a) {
  /* Assign a to b using the standard vector variant */
  GrB_assign(b->vec,    /* w = Write output to vector b */
             GrB_NULL,  /* mask = Don't block writing to any elements in w */
             GrB_NULL,  /* assum = Overwrite existing elements in w */
             a->vec,    /* u = Use a as a source */
             GrB_ALL,   /* indices = Assign all indices */
             b->length, /* nindices = The number of indices */
             GrB_NULL); /* descr = Options for the operation (TODO: Use this for performance) */
}

void OSQPVectori_copy(OSQPVectori      *b,
                     const OSQPVectori *a) {
  /* Assign a to b using the standard vector variant */
  GrB_assign(b->vec,    /* w = Write output to vector b */
             GrB_NULL,  /* mask = Don't block writing to any elements in w */
             GrB_NULL,  /* assum = Overwrite existing elements in w */
             a->vec,    /* u = Use a as a source */
             GrB_ALL,   /* indices = Assign all indices */
             b->length, /* nindices = The number of indices */
             GrB_NULL); /* descr = Options for the operation (TODO: Use this for performance) */
}

void OSQPVectorf_from_raw(OSQPVectorf   *b,
                          const c_float *av) {
  GrB_Vector_build(b->vec,    /* w = Store the items into vector b */
                   GrB_ALL,   /* indices = Store into all indices in w */
                   av,        /* values = Use av as the source of values */
                   b->length, /* n = The length of the av array */
                   GrB_NULL); /* dup = Error if the indices are duplicated (not an issue here) */
}

void OSQPVectori_from_raw(OSQPVectori *b,
                          const c_int *av) {
  GrB_Vector_build(b->vec,        /* w = Store the items into vector b */
                   GrB_ALL,       /* indices = Store into all indices in w */
                   (gb_int*) av,  /* values = Use av as the source of values */
                   b->length,     /* n = The length of the av array */
                   GrB_NULL);     /* dup = Error if the indices are duplicated (not an issue here) */
}

void OSQPVectorf_to_raw(c_float           *bv,
                        const OSQPVectorf *a) {
  c_int i;
  c_int length = a->length;

  for (i = 0; i < length; i++) {
    GrB_Vector_extractElement(&bv[i], a->vec, i);
  }
}

void OSQPVectori_to_raw(c_int *bv, const OSQPVectori *a) {
  c_int i;
  c_int length = a->length;

  for (i = 0; i < length; i++) {
    GrB_Vector_extractElement((gb_int*) &bv[i], a->vec, i);
  }
}

void OSQPVectorf_set_scalar(OSQPVectorf *a,
                            c_float      sc) {
  /* Use the constant vector variant to assign a scalar to the entire vector */
  GrB_assign(a->vec,    /* w = Vector to operate on */
             GrB_NULL,  /* mask = Don't prevent writing to any elements of w */
             GrB_NULL,  /* accum = Don't accumulate into w */
             sc,        /* val = Assign sc to the elements */
             GrB_ALL,   /* indices = Assign to all indices */
             a->length, /* nindices = The total length of the vector */
             GrB_NULL); /* desc = Option descriptor for the operation (TODO: Use this for performance) */
}

/* TODO */
void OSQPVectorf_set_scalar_conditional(OSQPVectorf       *a,
                                        const OSQPVectori *test,
                                        c_float            sc_if_neg,
                                        c_float            sc_if_zero,
                                        c_float            sc_if_pos){
  c_int     i;
  c_int     length = a->length;
  c_float*  av     = a->values;
  c_int*    testv  = test->values;

  for (i = 0; i < length; i++) {
      if (testv[i] == 0)      av[i] = sc_if_zero;
      else if (testv[i] > 0)  av[i] = sc_if_pos;
      else                    av[i] = sc_if_neg;
  }
}

void OSQPVectorf_mult_scalar(OSQPVectorf *a,
                             c_float      sc) {
  /* Use the Vector-BinaryOp variant to apply the times operator to each element with a given scalar */
  GrB_apply(a->vec,         /* w = Output vector */
            GrB_NULL,       /* mask = Write to all entries */
            GrB_NULL,       /* accum = Perform no operation on the output vector */
            GR_FLOAT_TIMES, /* op = Multiplication binary operator */
            sc,             /* val = Scalar to multiply by */
            a->vec,         /* u = Vector to multiply each element */
            GrB_NULL);      /* desc = Descriptor for settings (TODO: Use this for performance) */
}

void OSQPVectorf_plus(OSQPVectorf      *x,
                     const OSQPVectorf *a,
                     const OSQPVectorf *b) {
  if (x == a){
    /* Accumulate b onto x */
    GrB_eWiseAdd(x->vec,              /* w = Output vector (that can be accumulated onto) */
                 GrB_NULL,            /* mask = Don't block writing to any elements of w */
                 GR_FLOAT_PLUS,       /* accum = Accumulate onto w using the PLUS operator */
                 GR_FLOAT_SEMIRING,   /* op = Use the PLUS_TIMES semiring for the algebraic operations between (plus)*/
                 b->vec,              /* u = Add this vector */
                 GrB_NULL,            /* v = No second vector in original operation*/
                 GrB_NULL);           /* desc = Descriptor for settings (TODO: Use this for performance) */
  }
  else {
    /* Replace x with a+b */
    GrB_eWiseAdd(x->vec,              /* w = Output vector (that can be accumulated onto) */
                 GrB_NULL,            /* mask = Don't block writing to any elements of w */
                 GrB_NULL,            /* accum = Don't accumulate onto w, just overwrite it */
                 GR_FLOAT_SEMIRING,   /* op = Use the PLUS_TIMES semiring for the algebraic operations between u and v */
                 a->vec,              /* u = First vector in operation */
                 b->vec,              /* v = Second vector in operation*/
                 GrB_NULL);           /* desc = Descriptor for settings (TODO: Use this for performance) */
  }
}

void OSQPVectorf_minus(OSQPVectorf       *x,
                       const OSQPVectorf *a,
                       const OSQPVectorf *b) {
  if (x == a){
    /* Subtract the elements of b from those of x x */
    GrB_eWiseAdd(x->vec,              /* w = Output vector (that can be accumulated onto) */
                 GrB_NULL,            /* mask = Don't block writing to any elements of w */
                 GR_FLOAT_MINUS,      /* accum = Accumulate onto w using the MINUS operator */
                 GR_FLOAT_SEMIRING,   /* op = Use the PLUS_TIMES semiring for the algebraic operations between u and v */
                 b->vec,              /* u = Subtract this vector */
                 GrB_NULL,            /* v = No second vector in original operation*/
                 GrB_NULL);           /* desc = Descriptor for settings (TODO: Use this for performance) */
  }
  else {
    /* Replace x with a-b */
    GrB_eWiseAdd(x->vec,              /* w = Output vector (that can be accumulated onto) */
                 GrB_NULL,            /* mask = Don't block writing to any elements of w */
                 GrB_NULL,            /* accum = Don't accumulate onto w, just overwrite it */
                 GR_FLOAT_MINUS,      /* op = Use the minus operator for the operation between each element of u and v */
                 a->vec,              /* u = First vector in operation */
                 b->vec,              /* v = Second vector in operation*/
                 GrB_NULL);           /* desc = Descriptor for settings (TODO: Use this for performance) */
  }
}

/* TODO */
void OSQPVectorf_add_scaled(OSQPVectorf       *x,
                            c_float            sca,
                            const OSQPVectorf *a,
                            c_float            scb,
                            const OSQPVectorf *b){
  c_int i;
  c_int length = x->length;
  c_float*  av = a->values;
  c_float*  bv = b->values;
  c_float*  xv = x->values;

  /* shorter version when incrementing */
  if (x == a && sca == 1.){
    for (i = 0; i < length; i++) {
      xv[i] += scb * bv[i];
    }
  }
  else {
    for (i = 0; i < length; i++) {
      xv[i] = sca * av[i] + scb * bv[i];
    }
  }
}

/* TODO */
void OSQPVectorf_add_scaled3(OSQPVectorf       *x,
                             c_float            sca,
                             const OSQPVectorf *a,
                             c_float            scb,
                             const OSQPVectorf *b,
                             c_float            scc,
                             const OSQPVectorf *c){
  c_int i;
  c_int length = x->length;
  c_float*  av = a->values;
  c_float*  bv = b->values;
  c_float*  cv = c->values;
  c_float*  xv = x->values;

  /* shorter version when incrementing */
  if (x == a && sca == 1.){
    for (i = 0; i < length; i++) {
      xv[i] += scb * bv[i] + scc * cv[i];
    }
  }
  else {
    for (i = 0; i < length; i++) {
      xv[i] =  sca * av[i] + scb * bv[i] + scc * cv[i];
    }
  }
}


c_float OSQPVectorf_norm_inf(const OSQPVectorf *v){
  c_float normval = 0.0;

  GrB_Scalar normaccum;
  GrB_Scalar_new(&normaccum, GRFLOAT);

  /* Use the vector->scalar variant to use a custom binary operator for the reduction.
     Using the vector->float variant would require the use of a monoid instead of a binary operator. */
  GrB_reduce(normaccum,     /* val = Store result in normaccum */
             GrB_NULL,      /* accum = Don't accumulate onto the value already in normval */
             maxabs,        /* op = Use the custom maxabs operator to find the item with the largest absolute value*/
             v->vec,        /* u = Operate on this vector */
             GrB_NULL);     /* desc = Descriptor for setting, not used in this function */

  GrB_Scalar_extractElement(&normval, normaccum);
  GrB_free(&normaccum);

  return normval;
}

// c_float OSQPVectorf_norm_1(const OSQPVectorf *v){

//   c_int   i;
//   c_int length  = v->length;
//   c_float*  vv  = v->values;
//   c_float normval = 0.0;

//   for (i = 0; i < length; i++) {
//     normval += c_absval(vv[i]);
//   }
//   return normval;
// }

c_float OSQPVectorf_scaled_norm_inf(const OSQPVectorf *S,
                                    const OSQPVectorf *v){

  c_int   i;
  c_int length  = v->length;
  c_float*  vv  = v->values;
  c_float*  Sv  = S->values;
  c_float absval;
  c_float normval = 0.0;

  for (i = 0; i < length; i++) {
    absval = c_absval(Sv[i] * vv[i]);
    if (absval > normval) normval = absval;
  }
  return normval;
}

// c_float OSQPVectorf_scaled_norm_1(const OSQPVectorf *S, const OSQPVectorf *v){

//   c_int   i;
//   c_int length  = v->length;
//   c_float*  vv  = v->values;
//   c_float*  Sv  = S->values;
//   c_float normval = 0.0;

//   for (i = 0; i < length; i++) {
//     normval += c_absval(Sv[i] * vv[i]);
//   }
//   return normval;
// }

c_float OSQPVectorf_norm_inf_diff(const OSQPVectorf *a,
                                  const OSQPVectorf *b){
  c_int   i;
  c_int   length = a->length;
  c_float*  av   = a->values;
  c_float*  bv   = b->values;
  c_float absval;
  c_float normDiff = 0.0;

  for (i = 0; i < length; i++) {
    absval = c_absval(av[i] - bv[i]);
    if (absval > normDiff) normDiff = absval;
  }
  return normDiff;
}

// c_float OSQPVectorf_norm_1_diff(const OSQPVectorf *a,
//                                 const OSQPVectorf *b){

//   c_int   i;
//   c_int   length = a->length;
//   c_float*  av   = a->values;
//   c_float*  bv   = b->values;
//   c_float normDiff = 0.0;

//   for (i = 0; i < length; i++) {
//     normDiff += c_absval(av[i] - bv[i]);
//   }
//   return normDiff;
// }

// c_float OSQPVectorf_sum(const OSQPVectorf *a){

//   c_int   i;
//   c_int   length = a->length;
//   c_float*  av   = a->values;
//   c_float val = 0.0;

//   for (i = 0; i < length; i++) {
//     val += av[i];
//   }

//   return val;
// }

c_float OSQPVectorf_dot_prod(const OSQPVectorf *a,
                             const OSQPVectorf *b){

  c_int   i;
  c_int   length = a->length;
  c_float*  av   = a->values;
  c_float*  bv   = b->values;
  c_float dotprod = 0.0;

  for (i = 0; i < length; i++) {
    dotprod += av[i] * bv[i];
  }
  return dotprod;
}

c_float OSQPVectorf_dot_prod_signed(const OSQPVectorf *a,
                                    const OSQPVectorf *b,
                                    c_int              sign){

  c_int   i;
  c_int   length = a->length;
  c_float*  av   = a->values;
  c_float*  bv   = b->values;
  c_float dotprod = 0.0;

  if (sign == 1) {  /* dot with positive part of b */
    for (i = 0; i < length; i++) {
      dotprod += av[i] * c_max(bv[i], 0.);
    }
  }
  else if (sign == -1){  /* dot with negative part of b */
    for (i = 0; i < length; i++) {
      dotprod += av[i] * c_min(bv[i],0.);
    }
  }
  else{
    /* return the conventional dot product */
    dotprod = OSQPVectorf_dot_prod(a, b);
  }
  return dotprod;
}

void OSQPVectorf_ew_prod(OSQPVectorf       *c,
                         const OSQPVectorf *a,
                         const OSQPVectorf *b){

  c_int i;
  c_int   length = a->length;
  c_float*  av   = a->values;
  c_float*  bv   = b->values;
  c_float*  cv   = c->values;


  if (c == a) {
    for (i = 0; i < length; i++) {
      cv[i] *= bv[i];
    }
  }
  else {
    for (i = 0; i < length; i++) {
      cv[i] = av[i] * bv[i];
    }
  }
}

c_int OSQPVectorf_all_leq(const OSQPVectorf *l,
                          const OSQPVectorf *u){

  c_int i;
  c_int length = l->length;
  c_float*  lv = l->values;
  c_float*  uv = u->values;

  for (i = 0; i < length; i++) {
    if (lv[i] > uv[i]) return 0;
  }
  return 1;
}

void OSQPVectorf_ew_bound_vec(OSQPVectorf       *x,
                              const OSQPVectorf *z,
                              const OSQPVectorf *l,
                              const OSQPVectorf *u){

  c_int i;
  c_int length = x->length;
  c_float*  xv = x->values;
  c_float*  zv = z->values;
  c_float*  lv = l->values;
  c_float*  uv = u->values;

  for (i = 0; i < length; i++) {
    xv[i] = c_min(c_max(zv[i], lv[i]), uv[i]);
  }
}

void OSQPVectorf_project_polar_reccone(OSQPVectorf       *y,
                                       const OSQPVectorf *l,
                                       const OSQPVectorf *u,
                                       c_float            infval){

  c_int i; // Index for loops
  c_int    length = y->length;
  c_float* yv = y->values;
  c_float* lv = l->values;
  c_float* uv = u->values;

  for (i = 0; i < length; i++) {
    if (uv[i]   > +infval) {       // Infinite upper bound
      if (lv[i] < -infval) {       // Infinite lower bound
        // Both bounds infinite
        yv[i] = 0.0;
      } else {
        // Only upper bound infinite
        yv[i] = c_min(yv[i], 0.0);
      }
    } else if (lv[i] < -infval) {  // Infinite lower bound
      // Only lower bound infinite
      yv[i] = c_max(yv[i], 0.0);
    }
  }
}

c_int OSQPVectorf_in_reccone(const OSQPVectorf *y,
                             const OSQPVectorf *l,
                             const OSQPVectorf *u,
                             c_float            infval,
                             c_float            tol){

  c_int i; // Index for loops

  c_int    length = y->length;
  c_float* yv     = y->values;
  c_float* lv     = l->values;
  c_float* uv     = u->values;

  for (i = 0; i < length; i++) {
    if (((uv[i] < +infval) &&
         (yv[i] > +tol)) ||
        ((lv[i] > -infval) &&
         (yv[i] < -tol))) {
      // At least one condition not satisfied -> not dual infeasible
      return 0;
    }
  }
  return 1;
}


// void OSQPVectorf_permute(OSQPVectorf *x, const OSQPVectorf *b, const OSQPVectori *p){

//   c_int j;
//   c_int length = x->length;
//   c_float*  xv = x->values;
//   c_float*  bv = b->values;
//   c_int*    pv = p->values;

//   for (j = 0; j < length; j++) {
//     xv[j] = bv[pv[j]];
//   }
// }

// void OSQPVectori_permute(OSQPVectori *x, const OSQPVectori *b, const OSQPVectori *p){

//   c_int j;
//   c_int length = x->length;
//   c_int*    xv = x->values;
//   c_int*    bv = b->values;
//   c_int*    pv = p->values;

//   for (j = 0; j < length; j++) {
//     xv[j] = bv[pv[j]];
//   }
// }

// void OSQPVectorf_ipermute(OSQPVectorf *x, const OSQPVectorf *b, const OSQPVectori *p){

//   c_int j;
//   c_int length = x->length;
//   c_float*  xv = x->values;
//   c_float*  bv = b->values;
//   c_int*    pv = p->values;

//   for (j = 0; j < length; j++) {
//     xv[pv[j]] = bv[j];
//   }
// }

// void OSQPVectori_ipermute(OSQPVectori *x, const OSQPVectori *b, const OSQPVectori *p){

//   c_int j;
//   c_int length = x->length;
//   c_int*    xv = x->values;
//   c_int*    bv = b->values;
//   c_int*    pv = p->values;

//   for (j = 0; j < length; j++) {
//     xv[pv[j]] = bv[j];
//   }
// }



#if EMBEDDED != 1

c_float OSQPVectorf_mean(const OSQPVectorf *a){

  c_int i;
  c_int length = a->length;
  c_float *av  = a->values;
  c_float val = 0.0;

  if (length) {
    for (i = 0; i < length; i++) {
      val += av[i];
    }
    return val / length;
  }
  else return val;
}

void OSQPVectorf_ew_reciprocal(OSQPVectorf      *b,
                              const OSQPVectorf *a){

  c_int i;
  c_int length = a->length;
  c_float*  av = a->values;
  c_float*  bv = b->values;

  for (i = 0; i < length; i++) {
    bv[i] = (c_float)1.0 / av[i];
  }
}

void OSQPVectorf_ew_sqrt(OSQPVectorf *a){

  c_int i;
  c_int length = a->length;
  c_float*  av = a->values;

  for (i = 0; i < length; i++) {
    av[i] = c_sqrt(av[i]);
  }
}

// void OSQPVectorf_ew_max(OSQPVectorf       *c,
//                         const OSQPVectorf *a,
//                         c_float            max_val){

//   c_int i;
//   c_int length = c->length;
//   c_float*  av = a->values;
//   c_float*  cv = c->values;

//   for (i = 0; i < length; i++) {
//     cv[i] = c_max(av[i], max_val);
//   }
// }

// void OSQPVectorf_ew_min(OSQPVectorf *c, const OSQPVectorf *a, c_float min_val){

//   c_int i;
//   c_int length = a->length;
//   c_float*  av = a->values;
//   c_float*  cv = c->values;

//   for (i = 0; i < length; i++) {
//     cv[i] = c_min(av[i], min_val);
//   }
// }

void OSQPVectorf_ew_max_vec(OSQPVectorf       *c,
                            const OSQPVectorf *a,
                            const OSQPVectorf *b){
  c_int i;
  c_int length = a->length;
  c_float*  av = a->values;
  c_float*  bv = b->values;
  c_float*  cv = c->values;

  for (i = 0; i < length; i++) {
    cv[i] = c_max(av[i], bv[i]);
  }
}

// void OSQPVectorf_ew_min_vec(OSQPVectorf       *c,
//                             const OSQPVectorf *a,
//                             const OSQPVectorf *b){
//   c_int i;
//   c_int length = a->length;
//   c_float*  av = a->values;
//   c_float*  bv = b->values;
//   c_float*  cv = c->values;

//   for (i = 0; i < length; i++) {
//     cv[i] = c_min(av[i], bv[i]);
//   }
// }

c_int OSQPVectorf_ew_bounds_type(OSQPVectori      *iseq,
                                const OSQPVectorf *l,
                                const OSQPVectorf *u,
                                c_float            tol,
                                c_float            infval){

  c_int i;
  c_int length   = iseq->length;
  c_int*   iseqv = iseq->values;
  c_float* lv    = l->values;
  c_float* uv    = u->values;
  c_int old_value, has_changed;

  has_changed = 0;

  for (i = 0; i < length; i++) {

    old_value = iseqv[i];

    if ((lv[i] < -infval) && (uv[i] > infval)) {
      // Loose bounds
      iseqv[i] = -1;
    } else if (uv[i] - lv[i] < tol) {
      // Equality constraints
      iseqv[i] = 1;
    } else {
      // Inequality constraints
      iseqv[i] = 0;
    }

    //has anything changed?
    has_changed = has_changed || (iseqv[i] != old_value);
  }

  return has_changed;

}

void OSQPVectorf_set_scalar_if_lt(OSQPVectorf       *x,
                                  const OSQPVectorf *z,
                                  c_float            testval,
                                  c_float            newval){
  c_int i;
  c_int length = x->length;
  c_float*  xv = x->values;
  c_float*  zv = z->values;

  for (i = 0; i < length; i++) {
    xv[i] = zv[i] < testval ? newval : zv[i];
  }
}

void OSQPVectorf_set_scalar_if_gt(OSQPVectorf       *x,
                                  const OSQPVectorf *z,
                                  c_float            testval,
                                  c_float            newval){
  c_int i;
  c_int length = x->length;
  c_float*  xv = x->values;
  c_float*  zv = z->values;

  for (i = 0; i < length; i++) {
    xv[i] = zv[i] > testval ? newval : zv[i];
  }
}



#endif /* EMBEDDED != 1 */
