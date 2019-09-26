#include "osqp.h"
#include "algebra_vector.h"
#include "algebra_types.h"

#include "cuda_handler.h"
#include "cuda_malloc.h"
#include "cuda_wrapper.h"
#include "helper_cuda.h"    /* --> checkCudaErrors */

/* VECTOR FUNCTIONS ----------------------------------------------------------*/

#ifndef EMBEDDED

OSQPVectorf* OSQPVectorf_new(const c_float *a, c_int length){

  OSQPVectorf* out = OSQPVectorf_malloc(length);
  if(!out) return OSQP_NULL;

  if (length > 0) {
    OSQPVectorf_from_raw(out, a);
  }
  return out;
}

OSQPVectori* OSQPVectori_new(const c_int *a, c_int length){

  OSQPVectori* out = OSQPVectori_malloc(length);
  if(!out) return OSQP_NULL;

  if (length > 0) {
    OSQPVectori_from_raw(out, a);
  }
  return out;
}

OSQPVectorf* OSQPVectorf_malloc(c_int length){

  OSQPVectorf *b;
  b = c_malloc(sizeof(OSQPVectorf));
  if (b) {
    b->length = length;
    if (length) {
      b->values = c_malloc(length * sizeof(c_float));
      if (!(b->values)) {
        c_free(b);
        b = OSQP_NULL;
      }
    }
    else {
      b->values = OSQP_NULL;
    }
  }
  return b;
}

OSQPVectori* OSQPVectori_malloc(c_int length){

  OSQPVectori *b;

  b = c_malloc(sizeof(OSQPVectori));
  if (b) {
    b->length = length;
    if (length) {
      b->values = c_malloc(length * sizeof(c_int));
      if (!(b->values)) {
        c_free(b);
        b = OSQP_NULL;
      }
    }
    else {
      b->values = OSQP_NULL;
    }
  }
  return b;
}

OSQPVectorf* OSQPVectorf_calloc(c_int length){

  OSQPVectorf *b;
  b = c_malloc(sizeof(OSQPVectorf));
  if (b) {
    b->length = length;
    if (length) {
      b->values = c_calloc(length, sizeof(c_float));
      if (!(b->values)) {
        c_free(b);
        b = OSQP_NULL;
      }
    }
    else {
      b->values = OSQP_NULL;
    }
  }
  return b;
}

OSQPVectori* OSQPVectori_calloc(c_int length){

  OSQPVectori *b;
  b = c_malloc(sizeof(OSQPVectori));
  if (b) {
    b->length = length;
    if (length) {
      b->values = c_calloc(length, sizeof(c_int));
      if (!(b->values)) {
        c_free(b);
        b = OSQP_NULL;
      }
    }
    else {
      b->values = OSQP_NULL;
    }
  }
  return b;
}

OSQPVectorf* OSQPVectorf_copy_new(const OSQPVectorf *a){

  OSQPVectorf* b = OSQPVectorf_malloc(a->length);
  if(b) OSQPVectorf_copy(b,a);
  return b;

}

OSQPVectori* OSQPVectori_copy_new(const OSQPVectori *a){

  OSQPVectori* b = OSQPVectori_malloc(a->length);
  if(b) OSQPVectori_copy(b,a);
  return b;
}

void OSQPVectorf_free(OSQPVectorf *a){
  if (a) c_free(a->values);
  c_free(a);
}

void OSQPVectori_free(OSQPVectori *a){
  if (a) c_free(a->values);
  c_free(a);
}

OSQPVectorf* OSQPVectorf_view(const OSQPVectorf *a, c_int head, c_int length){

  OSQPVectorf* view = c_malloc(sizeof(OSQPVectorf));
  if(view){
    view->length = length;
    view->values   = a->values + head;
  }
  return view;
}

void OSQPVectorf_view_free(OSQPVectorf *a){
  c_free(a);
}


#endif /* ifndef EMBEDDED */

c_int OSQPVectorf_length(const OSQPVectorf *a){return a->length;}
c_int OSQPVectori_length(const OSQPVectori *a){return a->length;}

/* Pointer to vector data (floats) */
c_float* OSQPVectorf_data(const OSQPVectorf *a){return a->values;}
c_int*   OSQPVectori_data(const OSQPVectori *a){return a->values;}

void OSQPVectorf_copy(OSQPVectorf *b, const OSQPVectorf *a){
  OSQPVectorf_from_raw(b, a->values);
}

void OSQPVectori_copy(OSQPVectori *b, const OSQPVectori *a){
  OSQPVectori_from_raw(b, a->values);
}

void OSQPVectorf_from_raw(OSQPVectorf *b, const c_float *av){
  c_int i;
  c_int length = b->length;
  c_float* bv  = b->values;

  for (i = 0; i < length; i++) {
    bv[i] = av[i];
  }
}

void OSQPVectori_from_raw(OSQPVectori *b, const c_int *av){
  c_int i;
  c_int length = b->length;
  c_int* bv = b->values;

  for (i = 0; i < length; i++) {
    bv[i] = av[i];
  }
}

void OSQPVectorf_to_raw(c_float *bv, const OSQPVectorf *a){
  c_int i;
  c_int length = a->length;
  c_float* av = a->values;

  for (i = 0; i < length; i++) {
    bv[i] = av[i];
  }
}

void OSQPVectori_to_raw(c_int *bv, const OSQPVectori *a){
  c_int i;
  c_int length = a->length;
  c_int* av = a->values;

  for (i = 0; i < length; i++) {
    bv[i] = av[i];
  }
}

void OSQPVectorf_set_scalar(OSQPVectorf *a, c_float sc){
  c_int i;
  c_int length = a->length;
  c_float* av  = a->values;

  for (i = 0; i < length; i++) {
    av[i] = sc;
  }
}


void OSQPVectorf_set_scalar_conditional(OSQPVectorf *a,
                                        OSQPVectori *test,
                                        c_float sc_if_neg,
                                        c_float sc_if_zero,
                                        c_float sc_if_pos){
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


void OSQPVectorf_mult_scalar(OSQPVectorf *a, c_float sc){
  c_int i;
  c_int length = a->length;
  c_float*  av = a->values;

  for (i = 0; i < length; i++) {
    av[i] *= sc;
  }
}

void OSQPVectorf_plus(OSQPVectorf      *x,
                     const OSQPVectorf *a,
                     const OSQPVectorf *b){
  c_int i;
  c_int length = a->length;
  c_float*  av = a->values;
  c_float*  bv = b->values;
  c_float*  xv = x->values;

  if (x == a){
    for (i = 0; i < length; i++) {
      xv[i] += bv[i];
    }
  }
  else {
    for (i = 0; i < length; i++) {
      xv[i] = av[i] + bv[i];
    }
  }
}

void OSQPVectorf_minus(OSQPVectorf       *x,
                       const OSQPVectorf *a,
                       const OSQPVectorf *b){
  c_int i;
  c_int length = a->length;
  c_float*  av = a->values;
  c_float*  bv = b->values;
  c_float*  xv = x->values;

  if (x == a) {
    for (i = 0; i < length; i++) {
      xv[i] -= bv[i];
    }
  }
  else {
    for (i = 0; i < length; i++) {
      xv[i] = av[i] - bv[i];
    }
  }
}


void OSQPVectorf_add_scaled(OSQPVectorf       *x,
                            c_float           sca,
                            const OSQPVectorf *a,
                            c_float           scb,
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

void OSQPVectorf_add_scaled3(OSQPVectorf       *x,
                             c_float           sca,
                             const OSQPVectorf *a,
                             c_float           scb,
                             const OSQPVectorf *b,
                             c_float           scc,
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

  c_int   i;
  c_int length  = v->length;
  c_float*  vv  = v->values;
  c_float normval = 0.0;
  c_float absval;

  for (i = 0; i < length; i++) {
    absval = c_absval(vv[i]);
    if (absval > normval) normval = absval;
  }
  return normval;
}

c_float OSQPVectorf_norm_1(const OSQPVectorf *v){

  c_int   i;
  c_int length  = v->length;
  c_float*  vv  = v->values;
  c_float normval = 0.0;

  for (i = 0; i < length; i++) {
    normval += c_absval(vv[i]);
  }
  return normval;
}

c_float OSQPVectorf_scaled_norm_inf(const OSQPVectorf *S, const OSQPVectorf *v){

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

c_float OSQPVectorf_scaled_norm_1(const OSQPVectorf *S, const OSQPVectorf *v){

  c_int   i;
  c_int length  = v->length;
  c_float*  vv  = v->values;
  c_float*  Sv  = S->values;
  c_float normval = 0.0;

  for (i = 0; i < length; i++) {
    normval += c_absval(Sv[i] * vv[i]);
  }
  return normval;
}

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

c_float OSQPVectorf_norm_1_diff(const OSQPVectorf *a,
                                const OSQPVectorf *b){

  c_int   i;
  c_int   length = a->length;
  c_float*  av   = a->values;
  c_float*  bv   = b->values;
  c_float normDiff = 0.0;

  for (i = 0; i < length; i++) {
    normDiff += c_absval(av[i] - bv[i]);
  }
  return normDiff;
}

c_float OSQPVectorf_sum(const OSQPVectorf *a){

  c_int   i;
  c_int   length = a->length;
  c_float*  av   = a->values;
  c_float val = 0.0;

  for (i = 0; i < length; i++) {
    val += av[i];
  }

  return val;
}

c_float OSQPVectorf_dot_prod(const OSQPVectorf *a, const OSQPVectorf *b){

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

c_float OSQPVectorf_dot_prod_signed(const OSQPVectorf *a, const OSQPVectorf *b, c_int sign){

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

c_int OSQPVectorf_all_leq(OSQPVectorf *l, OSQPVectorf* u){

  c_int i;
  c_int length = l->length;
  c_float*  lv = l->values;
  c_float*  uv = u->values;

  for (i = 0; i < length; i++) {
    if (lv[i] > uv[i]) return 0;
  }
  return 1;
}

void OSQPVectorf_ew_bound_vec(OSQPVectorf *x,
                              OSQPVectorf *z,
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
                                       c_float        infval){

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
                             c_float           infval,
                             c_float           tol){

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


void OSQPVectorf_permute(OSQPVectorf *x, const OSQPVectorf *b, const OSQPVectori *p){

  c_int j;
  c_int length = x->length;
  c_float*  xv = x->values;
  c_float*  bv = b->values;
  c_int*    pv = p->values;

  for (j = 0; j < length; j++) {
    xv[j] = bv[pv[j]];
  }
}

void OSQPVectori_permute(OSQPVectori *x, const OSQPVectori *b, const OSQPVectori *p){

  c_int j;
  c_int length = x->length;
  c_int*    xv = x->values;
  c_int*    bv = b->values;
  c_int*    pv = p->values;

  for (j = 0; j < length; j++) {
    xv[j] = bv[pv[j]];
  }
}

void OSQPVectorf_ipermute(OSQPVectorf *x, const OSQPVectorf *b, const OSQPVectori *p){

  c_int j;
  c_int length = x->length;
  c_float*  xv = x->values;
  c_float*  bv = b->values;
  c_int*    pv = p->values;

  for (j = 0; j < length; j++) {
    xv[pv[j]] = bv[j];
  }
}

void OSQPVectori_ipermute(OSQPVectori *x, const OSQPVectori *b, const OSQPVectori *p){

  c_int j;
  c_int length = x->length;
  c_int*    xv = x->values;
  c_int*    bv = b->values;
  c_int*    pv = p->values;

  for (j = 0; j < length; j++) {
    xv[pv[j]] = bv[j];
  }
}



#if EMBEDDED != 1

c_float OSQPVectorf_mean(const OSQPVectorf *a){

  if (a->length) {
    return OSQPVectorf_sum(a) / (a->length);
  }
  else {
    return 0;
  }
}

void OSQPVectorf_ew_reciprocal(OSQPVectorf *b, const OSQPVectorf *a){

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

void OSQPVectorf_ew_max(OSQPVectorf *c, const OSQPVectorf *a, c_float max_val){

  c_int i;
  c_int length = c->length;
  c_float*  av = a->values;
  c_float*  cv = c->values;

  for (i = 0; i < length; i++) {
    cv[i] = c_max(av[i], max_val);
  }
}

void OSQPVectorf_ew_min(OSQPVectorf *c, const OSQPVectorf *a, c_float min_val){

  c_int i;
  c_int length = a->length;
  c_float*  av = a->values;
  c_float*  cv = c->values;

  for (i = 0; i < length; i++) {
    cv[i] = c_min(av[i], min_val);
  }
}

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

void OSQPVectorf_ew_min_vec(OSQPVectorf       *c,
                            const OSQPVectorf *a,
                            const OSQPVectorf *b){
  c_int i;
  c_int length = a->length;
  c_float*  av = a->values;
  c_float*  bv = b->values;
  c_float*  cv = c->values;

  for (i = 0; i < length; i++) {
    cv[i] = c_min(av[i], bv[i]);
  }
}

c_int OSQPVectorf_ew_bounds_type(OSQPVectori* iseq,
                                const OSQPVectorf* l,
                                const OSQPVectorf* u,
                                c_float tol,
                                c_float infval){

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

void OSQPVectorf_set_scalar_if_lt(OSQPVectorf *x,
                                  const OSQPVectorf *z,
                                  c_float testval,
                                  c_float newval){
  c_int i;
  c_int length = x->length;
  c_float*  xv = x->values;
  c_float*  zv = z->values;

  for (i = 0; i < length; i++) {
    xv[i] = zv[i] < testval ? newval : zv[i];
  }
}

void OSQPVectorf_set_scalar_if_gt(OSQPVectorf *x,
                                  const OSQPVectorf *z,
                                  c_float testval,
                                  c_float newval){
  c_int i;
  c_int length = x->length;
  c_float*  xv = x->values;
  c_float*  zv = z->values;


  for (i = 0; i < length; i++) {
    xv[i] = zv[i] > testval ? newval : zv[i];
  }
}



#endif /* EMBEDDED != 1 */
