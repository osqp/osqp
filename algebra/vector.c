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

OSQPVectorf* OSQPVectorf_copy_new(OSQPVectorf *a){

  OSQPVectorf *b;
  b         = c_malloc(sizeof(OSQPVectorf));
  b->length = a->length;
  b->values = c_malloc(b->length * sizeof(c_float));

  OSQPVectorf_copy(a,b);
  return b;
}

OSQPVectori* OSQPVectori_copy_new(OSQPVectori *a){

  OSQPVectori *b;
  b         = c_malloc(sizeof(OSQPVectori));
  b->length = a->length;
  b->values = c_malloc(b->length * sizeof(c_int));

  OSQPVectori_copy(a,b);
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

#endif // end EMBEDDED

c_int OSQPVectorf_length(OSQPVectorf *a){return a->length;}
c_int OSQPVectori_length(OSQPVectori *a){return a->length;}

/* Pointer to vector data (floats) */
c_float* OSQPVectorf_data(OSQPVectorf *a){return a->values;}
c_int*   OSQPVectori_data(OSQPVectori *a){return a->values;}

void OSQPVectorf_copy(OSQPVectorf *a,OSQPVectorf *b){

  c_int i;

  assert(a->length == b->length);

  for (i = 0; i < b->length; i++) {
    b->values[i] = a->values[i];
  }
}

void OSQPVectori_copy(OSQPVectori *a,OSQPVectori *b){

  c_int i;

  assert(a->length == b->length);

  for (i = 0; i < b->length; i++) {
    b->values[i] = a->values[i];
  }
}

void OSQPVectorf_set_scalar(OSQPVectorf *a, c_float sc){
  c_int i;
  for (i = 0; i < (int)a->length; i++) {
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

void OSQPVectorf_add_scaled(OSQPVectorf       *c,
                             const OSQPVectorf *a,
                             const OSQPVectorf *b,
                             c_float           sc){
  c_int i;

  assert(a->length == b->length);
  assert(a->length == c->length);

  for (i = 0; i < a->length; i++) {
    c->values[i] =  a->values[i] + sc * b->values[i];
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

void OSQPVectorf_ew_prod(const OSQPVectorf *a,
                         const OSQPVectorf *b,
                               OSQPVectorf *c){

    c_int i;

    assert(a->length == b->length);
    assert(a->length == c->length);

    for (i = 0; i < a->length; i++) {
      c->values[i] = a->values[i] * b->values[i];
    }
}

void OSQPVectorf_permute(OSQPVectorf *x, OSQPVectorf *b, OSQPVectori *p){
  for (c_int j = 0; j < x->length ; j++) x->values[j] = b->values[p->values[j]];
}

void OSQPVectori_permute(OSQPVectori *x, OSQPVectori *b, OSQPVectori *p){
  for (c_int j = 0; j < x->length ; j++) x->values[j] = b->values[p->values[j]];
}

void OSQPVectorf_ipermute(OSQPVectorf *x, OSQPVectorf *b, OSQPVectori *p){
  for (c_int j = 0; j < x->length ; j++) x->values[p->values[j]] = b->values[j];
}

void OSQPVectori_ipermute(OSQPVectori *x, OSQPVectori *b, OSQPVectori *p){
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

void OSQPVectorf_ew_reciprocal(const OSQPVectorf *a, OSQPVectorf *b){

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

void OSQPVectorf_ew_max(OSQPVectorf *a, c_float max_val){

  c_int i;

  for (i = 0; i < a->length; i++) {
    a->values[i] = c_max(a->values[i], max_val);
  }
}

void OSQPVectorf_ew_min(OSQPVectorf *a, c_float min_val){

  c_int i;

  for (i = 0; i < a->length; i++) {
    a->values[i] = c_min(a->values[i], min_val);
  }
}

void OSQPVectorf_ew_max_vec(const OSQPVectorf *a,
                            const OSQPVectorf *b,
                            OSQPVectorf       *c){
  c_int i;

  assert(a->length == b->length);
  assert(a->length == c->length);

  for (i = 0; i < a->length; i++) {
    c->values[i] = c_max(a->values[i], b->values[i]);
  }
}

void OSQPVectorf_ew_min_vec(const OSQPVectorf *a,
                            const OSQPVectorf *b,
                            OSQPVectorf       *c){
  c_int i;

  assert(a->length == b->length);
  assert(a->length == c->length);

  for (i = 0; i < a->length; i++) {
    c->values[i] = c_min(a->values[i], b->values[i]);
  }
}

void OSQPVectorf_ew_bound_vec(const OSQPVectorf *l,
                              const OSQPVectorf *u,
                                    OSQPVectorf *z,
                                    OSQPVectorf *x){

  c_int i;

  assert(l->length == u->length);
  assert(u->length == z->length);
  assert(z->length == x->length);

  c_int m = x->length;
  c_float* xval = x->values;
  c_float* zval = z->values;

  for(i=0; i < m; i++){
      xval[i] = c_min(c_max(zval[i],l->values[i]),u->values[i]);
  }
}



#endif // EMBEDDED != 1
