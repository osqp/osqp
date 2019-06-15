#include "lin_alg.h"
#include <assert.h>

/* VECTOR FUNCTIONS (old) ----------------------------------------------------------*/


void vec_add_scaled(c_float       *c,
                    const c_float *a,
                    const c_float *b,
                    c_int          n,
                    c_float        sc) {
  c_int i;

  for (i = 0; i < n; i++) {
    c[i] =  a[i] + sc * b[i];
  }
}

c_float vec_scaled_norm_inf(const c_float *S, const c_float *v, c_int l) {
  c_int   i;
  c_float abs_Sv_i;
  c_float max = 0.0;

  for (i = 0; i < l; i++) {
    abs_Sv_i = c_absval(S[i] * v[i]);

    if (abs_Sv_i > max) max = abs_Sv_i;
  }
  return max;
}

c_float vec_norm_inf(const c_float *v, c_int l) {
  c_int   i;
  c_float abs_v_i;
  c_float max = 0.0;

  for (i = 0; i < l; i++) {
    abs_v_i = c_absval(v[i]);

    if (abs_v_i > max) max = abs_v_i;
  }
  return max;
}

c_float vec_norm_inf_diff(const c_float *a, const c_float *b, c_int l) {
  c_float nmDiff = 0.0, tmp;
  c_int   i;

  for (i = 0; i < l; i++) {
    tmp = c_absval(a[i] - b[i]);

    if (tmp > nmDiff) nmDiff = tmp;
  }
  return nmDiff;
}

c_float vec_mean(const c_float *a, c_int n) {
  c_float mean = 0.0;
  c_int   i;

  for (i = 0; i < n; i++) {
    mean += a[i];
  }
  mean /= (c_float)n;

  return mean;
}

void int_vec_set_scalar(c_int *a, c_int sc, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    a[i] = sc;
  }
}

void vec_set_scalar(c_float *a, c_float sc, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    a[i] = sc;
  }
}

void vec_add_scalar(c_float *a, c_float sc, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    a[i] += sc;
  }
}

void vec_mult_scalar(c_float *a, c_float sc, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    a[i] *= sc;
  }
}

#ifndef EMBEDDED
c_float* vec_copy(c_float *a, c_int n) {
  c_float *b;
  c_int    i;

  b = c_malloc(n * sizeof(c_float));
  if (!b) return OSQP_NULL;

  for (i = 0; i < n; i++) {
    b[i] = a[i];
  }

  return b;
}

#endif // end EMBEDDED


void prea_int_vec_copy(const c_int *a, c_int *b, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    b[i] = a[i];
  }
}

void prea_vec_copy(const c_float *a, c_float *b, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    b[i] = a[i];
  }
}

void vec_ew_recipr(const c_float *a, c_float *b, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    b[i] = (c_float)1.0 / a[i];
  }
}

c_float vec_prod(const c_float *a, const c_float *b, c_int n) {
  c_float prod = 0.0;
  c_int   i; // Index

  for (i = 0; i < n; i++) {
    prod += a[i] * b[i];
  }

  return prod;
}

void vec_ew_prod(const c_float *a, const c_float *b, c_float *c, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    c[i] = b[i] * a[i];
  }
}

#if EMBEDDED != 1

void vec_ew_sqrt(c_float *a, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    a[i] = c_sqrt(a[i]);
  }
}

void vec_ew_max(c_float *a, c_int n, c_float max_val) {
  c_int i;

  for (i = 0; i < n; i++) {
    a[i] = c_max(a[i], max_val);
  }
}

void vec_ew_min(c_float *a, c_int n, c_float min_val) {
  c_int i;

  for (i = 0; i < n; i++) {
    a[i] = c_min(a[i], min_val);
  }
}

void vec_ew_max_vec(const c_float *a, const c_float *b, c_float *c, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    c[i] = c_max(a[i], b[i]);
  }
}

void vec_ew_min_vec(const c_float *a, const c_float *b, c_float *c, c_int n) {
  c_int i;

  for (i = 0; i < n; i++) {
    c[i] = c_min(a[i], b[i]);
  }
}

#endif // EMBEDDED != 1

/* VECTOR FUNCTIONS (new)----------------------------------------------------------*/


#ifndef EMBEDDED

OSQPVectorf* OSQPVectorf_new(const c_float *a, c_int length){
  OSQPVectorf* out = OSQPVectorf_malloc(length);
  OSQPVectorf_copy_raw(out,a);
  return out;
}

OSQPVectori* OSQPVectori_new(const c_int *a, c_int length){
  OSQPVectori* out = OSQPVectori_malloc(length);
  OSQPVectori_copy_raw(out,a);
  return out;
}

OSQPVectorf* OSQPVectorf_malloc(c_int len){

  OSQPVectorf *b;
  b         = c_malloc(sizeof(OSQPVectorf));
  b->length = len;
  b->values = c_malloc(len * sizeof(c_float));
  return b;
}

OSQPVectorf* OSQPVectorf_calloc(c_int len){

  OSQPVectorf *b;
  b         = c_malloc(sizeof(OSQPVectorf));
  b->length = len;
  b->values = c_calloc(len,sizeof(c_float));
  return b;
}

OSQPVectori* OSQPVectori_malloc(c_int len){

  OSQPVectori *b;
  b         = c_malloc(sizeof(OSQPVectori));
  b->length = len;
  b->values = c_malloc(len * sizeof(c_int));
  return b;
}

OSQPVectori* OSQPVectori_calloc(c_int len){

  OSQPVectori *b;
  b         = c_malloc(sizeof(OSQPVectori));
  b->length = len;
  b->values = c_calloc(len,sizeof(c_int));
  return b;
}

OSQPVectorf* OSQPVectorf_copy_new(const OSQPVectorf *a){

  OSQPVectorf* b = OSQPVectorf_malloc(a->length);
  OSQPVectorf_copy(b,a);
  return b;

}

OSQPVectori* OSQPVectori_copy_new(const OSQPVectori *a){

  OSQPVectori* b = OSQPVectori_malloc(a->length);
  OSQPVectori_copy(b,a);
  return b;
}

/* Free a float vector*/
OSQPVectorf* OSQPVectorf_free(OSQPVectorf *a){
  if(a) c_free(a->values);
  c_free(a);
  return OSQP_NULL;
}

/* Free an int vector*/
OSQPVectori* OSQPVectori_free(OSQPVectori *a){
  if(a) c_free(a->values);
  c_free(a);
  return OSQP_NULL;
}

OSQPVectorf* OSQPVectorf_view(const OSQPVectorf *a, c_int head, c_int len){
  OSQPVectorf* view = c_malloc(sizeof(OSQPVectori));;
  view->length = len;
  view->values   = a->values + head;
  return view;
}

OSQPVectorf* OSQPVectorf_view_free(OSQPVectorf *a){
  c_free(a);
  return OSQP_NULL;
}


#endif // end EMBEDDED

c_int OSQPVectorf_length(const OSQPVectorf *a){return a->length;}
c_int OSQPVectori_length(const OSQPVectori *a){return a->length;}

/* Pointer to vector data (floats) */
c_float* OSQPVectorf_data(const OSQPVectorf *a){return a->values;}
c_int*   OSQPVectori_data(const OSQPVectori *a){return a->values;}

void OSQPVectorf_copy(OSQPVectorf *b, const OSQPVectorf *a){
  OSQPVectorf_copy_raw(b,a->values);
}

void OSQPVectori_copy(OSQPVectori *b, const OSQPVectori *a){
  OSQPVectori_copy_raw(b,a->values);
}

void OSQPVectorf_copy_raw(OSQPVectorf *b, const c_float *av){
  c_int i;
  c_float* bv;

  for (i = 0; i < b->length; i++) {
    bv[i] = av[i];
  }
}

void OSQPVectori_copy_raw(OSQPVectori *b, const c_int *av){
  c_int i;
  c_int* bv;

  for (i = 0; i < b->length; i++) {
    bv[i] = av[i];
  }
}

void OSQPVectorf_set_scalar(OSQPVectorf *a, c_float sc){
  c_int i;
  for (i = 0; i < a->length; i++) {
    a->values[i] = sc;
  }
  return;
}

void OSQPVectori_set_scalar(OSQPVectori *a, c_int sc){
  c_int i;
  for (i = 0; i < a->length; i++) {
    a->values[i] = sc;
  }
}

void OSQPVectorf_set_scalar_conditional(OSQPVectorf *a,
                                        OSQPVectori *test,
                                        c_float sctrue,
                                        c_float scfalse){
    c_int i;
    for (i = 0; i < a->length; i++) {
      a->values[i] = test->values[i] ? sctrue : scfalse;
    }
}

void OSQPVectorf_add_scalar(OSQPVectorf *a, c_float sc){
  c_int i;
  for (i = 0; i < a->length; i++) {
    a->values[i] += sc;
  }
}

void OSQPVectori_add_scalar(OSQPVectori *a, c_int sc){
  c_int i;
  for (i = 0; i < a->length; i++) {
    a->values[i] += sc;
  }
}

void OSQPVectorf_mult_scalar(OSQPVectorf *a, c_float sc){
  c_int i;
  for (i = 0; i < a->length; i++) {
    a->values[i] *= sc;
  }
}

void OSQPVectorf_negate(OSQPVectorf *a){
  c_int i;
  for (i = 0; i < a->length; i++) {
    a->values[i] = -a->values[i];
  }
}

void OSQPVectorf_plus(OSQPVectorf      *x,
                     const OSQPVectorf *a,
                     const OSQPVectorf *b){
  c_int i;
  assert(a->length == b->length);

  if(x == a){
    for (i = 0; i < a->length; i++) {
      x->values[i] += b->values[i];
    }
  }
  else{
    for (i = 0; i < a->length; i++) {
      x->values[i] = a->values[i] + b->values[i];
    }
  }
}

void OSQPVectorf_minus(OSQPVectorf     *x,
                     const OSQPVectorf *a,
                     const OSQPVectorf *b){
  c_int i;
  assert(a->length == b->length);
  assert(a->length == x->length);

  if(x == a){
    for (i = 0; i < a->length; i++) {
      x->values[i] -= b->values[i];
    }
  }
  else{
    for (i = 0; i < a->length; i++) {
      x->values[i] = a->values[i] - b->values[i];
    }
  }
}


void OSQPVectorf_add_scaled(OSQPVectorf       *x,
                             c_float           sca,
                             const OSQPVectorf *a,
                             c_float           scb,
                             const OSQPVectorf *b){
  c_int i;
  assert(a->length == b->length);
  assert(a->length == x->length);

  //shorter version when incrementing
  if(x == a && sca == 1.){
    for (i = 0; i < x->length; i++) {
      x->values[i] += scb * b->values[i];
    }
  }
  else{
    for (i = 0; i < x->length; i++) {
      x->values[i] = sca * a->values[i] + scb * b->values[i];
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
  assert(a->length == b->length);
  assert(b->length == c->length);
  assert(a->length == x->length);

  //shorter version when incrementing
  if(x == a && sca == 1.){
    for (i = 0; i < x->length; i++) {
      x->values[i] += scb * b->values[i] + scc * c->values[i];
    }
  }
  else{
    for (i = 0; i < x->length; i++) {
      c->values[i] =  sca * a->values[i] + scb * b->values[i] + scc * c->values[i];
    }
  }
}


c_float OSQPVectorf_norm_inf(const OSQPVectorf *v){

  c_int   i;
  c_float normval = 0.0;
  c_float absval;

  for (i = 0; i < v->length; i++) {
    absval = c_absval(v->values[i]);
    if (absval > normval) normval = absval;
  }
  return normval;
}

c_float OSQPVectorf_norm_1(const OSQPVectorf *v){

  c_int   i;
  c_float normval = 0.0;

  for (i = 0; i < v->length; i++) {
    normval += c_absval(v->values[i]);
  }
  return normval;
}

c_float OSQPVectorf_scaled_norm_inf(const OSQPVectorf *S, const OSQPVectorf *v){

  c_int   i;
  c_float absval;
  c_float normval = 0.0;

  assert(S->length == v->length);

  for (i = 0; i < v->length; i++) {
    absval = c_absval(S->values[i] * v->values[i]);
    if (absval > normval) normval = absval;
  }
  return normval;
}

c_float OSQPVectorf_scaled_norm_1(const OSQPVectorf *S, const OSQPVectorf *v){

  c_int   i;
  c_float normval = 0.0;

  assert(S->length == v->length);

  for (i = 0; i < v->length; i++) {
    normval += c_absval(S->values[i] * v->values[i]);
  }
  return normval;
}

c_float OSQPVectorf_norm_inf_diff(const OSQPVectorf *a,
                                  const OSQPVectorf *b){
  c_int   i;
  c_float absval;
  c_float normDiff = 0.0;

  assert(a->length == b->length);

  for (i = 0; i < a->length; i++) {
    absval = c_absval(a->values[i] - b->values[i]);
    if (absval > normDiff) normDiff = absval;
  }
  return normDiff;
}

c_float OSQPVectorf_norm_1_diff(const OSQPVectorf *a,
                                const OSQPVectorf *b){

  c_int   i;
  c_float normDiff = 0.0;
  assert(a->length == b->length);

  for (i = 0; i < a->length; i++) {
    normDiff += c_absval(a->values[i] - b->values[i]);
  }
  return normDiff;
}

c_float OSQPVectorf_sum(const OSQPVectorf *a){

  c_int   i;
  c_float val = 0.0;

  for (i = 0; i < a->length; i++) {
    val += a->values[i];
  }

  return val;
}

c_float OSQPVectorf_dot_prod(const OSQPVectorf *a, const OSQPVectorf *b){

  c_int   i; // Index
  c_float dotprod = 0.0;
  assert(a->length == b->length);

  for (i = 0; i < a->length; i++) {
    dotprod += a->values[i] * b->values[i];
  }
  return dotprod;
}

c_float OSQPVectorf_dot_prod_signed(const OSQPVectorf *a, const OSQPVectorf *b, c_int sign){

  c_int   i; // Index
  c_float dotprod = 0.0;
  assert(a->length == b->length);

  if(sign == 1){  /* dot with positive part of b */
    for (i = 0; i < a->length; i++) {
      dotprod += a->values[i] * c_max(b->values[i],0.);
    }
  }
  else if(sign == -1){  /* dot with negative part of b */
    for (i = 0; i < a->length; i++) {
      dotprod += a->values[i] * c_min(b->values[i],0.);
    }
  }
  else{
    /* return the conventional dot product */
    dotprod = OSQPVectorf_dot_prod(a,b);
  }
  return dotprod;
}

void OSQPVectorf_ew_prod(OSQPVectorf       *c,
                         const OSQPVectorf *a,
                         const OSQPVectorf *b){

    c_int i;
    assert(a->length == b->length);
    assert(a->length == c->length);

    if(c==a) {
      for (i = 0; i < a->length; i++) {
        c->values[i] *= b->values[i];
      }
    }
    else{
      for (i = 0; i < a->length; i++) {
        c->values[i] = a->values[i] * b->values[i];
      }
    }
}

void OSQPVectorf_permute(OSQPVectorf *x, const OSQPVectorf *b, const OSQPVectori *p){

  assert(x->length == b->length);
  assert(x->length == p->length);

  for (c_int j = 0; j < x->length ; j++) x->values[j] = b->values[p->values[j]];
}

void OSQPVectori_permute(OSQPVectori *x, const OSQPVectori *b, const OSQPVectori *p){

  assert(x->length == b->length);
  assert(x->length == p->length);

  for (c_int j = 0; j < x->length ; j++) x->values[j] = b->values[p->values[j]];
}

void OSQPVectorf_ipermute(OSQPVectorf *x, const OSQPVectorf *b, const OSQPVectori *p){

  assert(x->length == b->length);
  assert(x->length == p->length);

  for (c_int j = 0; j < x->length ; j++) x->values[p->values[j]] = b->values[j];
}

void OSQPVectori_ipermute(OSQPVectori *x, const OSQPVectori *b, const OSQPVectori *p){

  assert(x->length == b->length);
  assert(x->length == p->length);

  for (c_int j = 0; j < x->length ; j++) x->values[p->values[j]] = b->values[j];
}



#if EMBEDDED != 1

c_float OSQPVectorf_mean(const OSQPVectorf *a){

  if(a->length){
    return OSQPVectorf_sum(a)/(a->length);
  }
  else{
    return 0;
  }
}

void OSQPVectorf_ew_reciprocal(OSQPVectorf *b, const OSQPVectorf *a){

  c_int i;

  assert(a->length == b->length);

  for (i = 0; i < a->length; i++) {
    b->values[i] = (c_float)1.0 / a->values[i];
  }
}

void OSQPVectorf_ew_sqrt(OSQPVectorf *a){

  c_int i;

  for (i = 0; i < a->length; i++) {
    a->values[i] = c_sqrt(a->values[i]);
  }
}

void OSQPVectorf_ew_max(OSQPVectorf *c, const OSQPVectorf *a, c_float max_val){

  c_int i;

  assert(a->length == c->length);

  for (i = 0; i < c->length; i++) {
    c->values[i] = c_max(a->values[i], max_val);
  }
}

void OSQPVectorf_ew_min(OSQPVectorf *c, const OSQPVectorf *a, c_float min_val){

  c_int i;

  assert(a->length == c->length);

  for (i = 0; i < a->length; i++) {
    c->values[i] = c_min(a->values[i], min_val);
  }
}

void OSQPVectorf_ew_max_vec(OSQPVectorf       *c,
                            const OSQPVectorf *a,
                            const OSQPVectorf *b){
  c_int i;

  assert(a->length == c->length);
  assert(b->length == c->length);

  for (i = 0; i < a->length; i++) {
    c->values[i] = c_max(a->values[i], b->values[i]);
  }
}

void OSQPVectorf_ew_min_vec(OSQPVectorf       *c,
                            const OSQPVectorf *a,
                            const OSQPVectorf *b){
  c_int i;

  assert(a->length == c->length);
  assert(b->length == c->length);

  for (i = 0; i < a->length; i++) {
    c->values[i] = c_min(a->values[i], b->values[i]);
  }
}

c_int OSQPVectorf_ew_lt(OSQPVectorf *l, OSQPVectorf* u){

  assert(l->length == u->length);

  c_int i;
  for (i = 0; i < l->length; i++) {
    if(l->values[i] > u->values[i]){return 0;}
  }
  return 1;
}

void OSQPVectorf_ew_bound_vec(OSQPVectorf *x,
                              OSQPVectorf *z,
                              const OSQPVectorf *l,
                              const OSQPVectorf *u){

  c_int i;

  assert(x->length == z->length);
  assert(x->length == l->length);
  assert(x->length == u->length);

  for(i=0; i < x->length; i++){
      x->values[i] = c_min(c_max(z->values[i],l->values[i]),u->values[i]);
  }
}

void OSQPVectorf_set_scalar_if_lt(OSQPVectorf *x,
                                  const OSQPVectorf *z,
                                  c_float testval,
                                  c_float newval){
  c_int i;
  assert(x->length == z->length);
  for(i=0; i < x->length; i++){
      x->values[i] = z->values[i] < testval ? newval : z->values[i];
  }
}

 /* Elementwise replacement based on gt comparison.
    x[i] = z[i] > testval ? newval : z[i];
  */
void OSQPVectorf_set_scalar_if_gt(OSQPVectorf *x,
                                  const OSQPVectorf *z,
                                  c_float testval,
                                  c_float newval){
  c_int i;
  assert(x->length == z->length);
  for(i=0; i < x->length; i++){
      x->values[i] = z->values[i] > testval ? newval : z->values[i];
  }
}



#endif // EMBEDDED != 1
