/**
 *  Copyright (c) 2019-2021 ETH Zurich, Automatic Control Lab,
 *  Michel Schubiger, Goran Banjac.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "algebra_types.h"
#include "algebra_vector.h"

#include "cuda_lin_alg.h"
#include "cuda_malloc.h"


/*******************************************************************************
 *                           API Functions                                     *
 *******************************************************************************/

OSQPInt OSQPVectorf_is_eq(const OSQPVectorf* a,
                          const OSQPVectorf* b,
                                OSQPFloat    tol) {

  OSQPInt res = 0;

  if (a->length != b->length)
    return 0;

  cuda_vec_eq(a->d_val, b->d_val, tol, a->length, &res);

  return res;
}

OSQPVectorf* OSQPVectorf_new(const OSQPFloat* a,
                                   OSQPInt    length) {

  OSQPVectorf* out = OSQPVectorf_malloc(length);
  if (!out) return OSQP_NULL;

  if (length > 0) OSQPVectorf_from_raw(out, a);
  
  return out;
}

OSQPVectorf* OSQPVectorf_malloc(OSQPInt length) {

  OSQPVectorf* b = (OSQPVectorf*) c_malloc(sizeof(OSQPVectorf));
  if (!b) return OSQP_NULL;

  b->length = length;
  if (length) {
    cuda_malloc((void **) &b->d_val, length * sizeof(OSQPFloat));
    if (!(b->d_val)) {
      c_free(b);
      b = OSQP_NULL;
    }
    cuda_vec_create(&b->vec, b->d_val, length);
  }
  else {
    b->d_val = OSQP_NULL;
    b->vec = NULL;
  }
  return b;
}

OSQPVectorf* OSQPVectorf_calloc(OSQPInt length) {

  OSQPVectorf* b = (OSQPVectorf*) c_malloc(sizeof(OSQPVectorf));
  if (!b) return OSQP_NULL;

  b->length = length;
  if (length) {
    cuda_calloc((void **) &b->d_val, length * sizeof(OSQPFloat));
    if (!(b->d_val)) {
      c_free(b);
      b = OSQP_NULL;
    }
    cuda_vec_create(&b->vec, b->d_val, length);
  }
  else {
    b->d_val = OSQP_NULL;
    b->vec = NULL;
  }
  return b;
}

OSQPVectori* OSQPVectori_new(const OSQPInt* a,
                                   OSQPInt  length) {

  OSQPVectori* out = OSQPVectori_malloc(length);
  if (!out) return OSQP_NULL;

  if (length > 0) OSQPVectori_from_raw(out, a);

  return out;
}

OSQPVectori* OSQPVectori_malloc(OSQPInt length) {

  OSQPVectori* b = (OSQPVectori*) c_malloc(sizeof(OSQPVectori));
  if (!b) return OSQP_NULL;

  b->length = length;
  if (length) {
    cuda_malloc((void **) &b->d_val, length * sizeof(OSQPInt));
    if (!(b->d_val)) {
      c_free(b);
      b = OSQP_NULL;
    }
  }
  else {
    b->d_val = OSQP_NULL;
  }
  return b;
}

OSQPVectori* OSQPVectori_calloc(OSQPInt length) {

  OSQPVectori* b = (OSQPVectori*) c_malloc(sizeof(OSQPVectori));
  if (!b) return OSQP_NULL;
  
  b->length = length;
  if (length) {
    cuda_calloc((void **) &b->d_val, length * sizeof(OSQPInt));
    if (!(b->d_val)) {
      c_free(b);
      b = OSQP_NULL;
    }
  }
  else {
    b->d_val = OSQP_NULL;
  }
  return b;
}

OSQPVectorf* OSQPVectorf_copy_new(const OSQPVectorf* a) {

  OSQPVectorf* b = OSQPVectorf_malloc(a->length);

  if (b) cuda_vec_copy_d2d(b->d_val, a->d_val, a->length);

  return b;
}

void OSQPVectorf_free(OSQPVectorf* a) {

  if (a) {
    cuda_vec_destroy(a->vec);
    cuda_free((void **) &a->d_val);
  }
  c_free(a);
}

void OSQPVectori_free(OSQPVectori* a) {

  if (a) cuda_free((void **) &a->d_val);
  c_free(a);
}

OSQPVectorf* OSQPVectorf_view(const OSQPVectorf* a,
                                    OSQPInt      head,
                                    OSQPInt      length) {

  OSQPVectorf* view = (OSQPVectorf*) c_malloc(sizeof(OSQPVectorf));
  if (view) {
    view->length = length;
    view->d_val  = a->d_val  + head;
    cuda_vec_create(&view->vec, view->d_val, length);
  }
  return view;
}

void OSQPVectorf_view_free(OSQPVectorf* a) {
  c_free(a);
}

OSQPInt OSQPVectorf_length(const OSQPVectorf* a) {return a->length;}
OSQPInt OSQPVectori_length(const OSQPVectori* a) {return a->length;}

void OSQPVectorf_copy(OSQPVectorf*       b,
                      const OSQPVectorf* a) {

  if (a) cuda_vec_copy_d2d(b->d_val, a->d_val, a->length);
}

void OSQPVectorf_from_raw(OSQPVectorf*     b,
                          const OSQPFloat* av) {

  if (av) cuda_vec_copy_h2d(b->d_val, av, b->length);
}

void OSQPVectori_from_raw(OSQPVectori*   b,
                          const OSQPInt* av) {

  cuda_vec_int_copy_h2d(b->d_val, av, b->length);
}

void OSQPVectorf_to_raw(OSQPFloat*         bv,
                        const OSQPVectorf* a) {

  cuda_vec_copy_d2h(bv, a->d_val, a->length);
}

void OSQPVectori_to_raw(OSQPInt*           bv,
                        const OSQPVectori* a) {

  cuda_vec_int_copy_d2h(bv, a->d_val, a->length);
}

void OSQPVectorf_set_scalar(OSQPVectorf* a,
                            OSQPFloat    sc) {

  cuda_vec_set_sc(a->d_val, sc, a->length);
}


void OSQPVectorf_set_scalar_conditional(OSQPVectorf*       a,
                                        const OSQPVectori* test,
                                        OSQPFloat          sc_if_neg,
                                        OSQPFloat          sc_if_zero,
                                        OSQPFloat          sc_if_pos) {

  cuda_vec_set_sc_cond(a->d_val, test->d_val, sc_if_neg, sc_if_zero, sc_if_pos, a->length);
}


void OSQPVectorf_mult_scalar(OSQPVectorf* a,
                             OSQPFloat    sc) {

  if (sc == 1.0 || !a->d_val) return;
  cuda_vec_mult_sc(a->d_val, sc, a->length);
}

void OSQPVectorf_plus(OSQPVectorf*      x,
                     const OSQPVectorf* a,
                     const OSQPVectorf* b) {

  cuda_vec_add_scaled(x->d_val, a->d_val, b->d_val, 1.0, 1.0, a->length);
}

void OSQPVectorf_minus(OSQPVectorf*       x,
                       const OSQPVectorf* a,
                       const OSQPVectorf* b) {

  cuda_vec_add_scaled(x->d_val, a->d_val, b->d_val, 1.0, -1.0, a->length);
}


void OSQPVectorf_add_scaled(OSQPVectorf*       x,
                            OSQPFloat          sca,
                            const OSQPVectorf* a,
                            OSQPFloat          scb,
                            const OSQPVectorf* b) {

  cuda_vec_add_scaled(x->d_val, a->d_val, b->d_val, sca, scb, x->length);
}

void OSQPVectorf_add_scaled3(OSQPVectorf*       x,
                             OSQPFloat          sca,
                             const OSQPVectorf* a,
                             OSQPFloat          scb,
                             const OSQPVectorf* b,
                             OSQPFloat          scc,
                             const OSQPVectorf* c) {

  cuda_vec_add_scaled3(x->d_val, a->d_val, b->d_val, c->d_val, sca, scb, scc, x->length);
}


OSQPFloat OSQPVectorf_norm_inf(const OSQPVectorf* v) {

  OSQPFloat normval;

  if (v->length) cuda_vec_norm_inf(v->d_val, v->length, &normval);
  else           normval = 0.0;

  return normval;
}

OSQPFloat OSQPVectorf_scaled_norm_inf(const OSQPVectorf* S,
                                      const OSQPVectorf* v) {

  OSQPFloat normval;

  if (v->length) cuda_vec_scaled_norm_inf(S->d_val, v->d_val, v->length, &normval);
  else           normval = 0.0;

  return normval;
}

OSQPFloat OSQPVectorf_norm_inf_diff(const OSQPVectorf* a,
                                    const OSQPVectorf* b) {

  OSQPFloat normDiff;

  if (a->length) cuda_vec_diff_norm_inf(a->d_val, b->d_val, a->length, &normDiff);
  else           normDiff = 0.0;

  return normDiff;
}

OSQPFloat OSQPVectorf_norm_1(const OSQPVectorf* a) {

  OSQPFloat val = 0.0;

  if (a->length) cuda_vec_norm_1(a->d_val, a->length, &val);

  return val;
}

OSQPFloat OSQPVectorf_dot_prod(const OSQPVectorf* a,
                               const OSQPVectorf* b) {

  OSQPFloat dotprod;

  if (a->length) cuda_vec_prod(a->d_val, b->d_val, a->length, &dotprod);
  else           dotprod = 0.0;

  return dotprod;
}

OSQPFloat OSQPVectorf_dot_prod_signed(const OSQPVectorf* a,
                                      const OSQPVectorf* b,
                                            OSQPInt      sign) {

  OSQPFloat dotprod;

  if (a->length) cuda_vec_prod_signed(a->d_val, b->d_val, sign, a->length, &dotprod);
  else           dotprod = 0.0;

  return dotprod;
}

void OSQPVectorf_ew_prod(OSQPVectorf*       c,
                         const OSQPVectorf* a,
                         const OSQPVectorf* b) {

  if (c->length) cuda_vec_ew_prod(c->d_val, a->d_val, b->d_val, c->length);
}

OSQPInt OSQPVectorf_all_leq(const OSQPVectorf* l,
                            const OSQPVectorf* u) {

  OSQPInt res;

  cuda_vec_leq(l->d_val, u->d_val, l->length, &res);

  return res;
}

void OSQPVectorf_ew_bound_vec(OSQPVectorf*       x,
                              const OSQPVectorf* z,
                              const OSQPVectorf* l,
                              const OSQPVectorf* u) {

  cuda_vec_bound(x->d_val, z->d_val, l->d_val, u->d_val, x->length);
}

void OSQPVectorf_project_polar_reccone(OSQPVectorf*       y,
                                       const OSQPVectorf* l,
                                       const OSQPVectorf* u,
                                       OSQPFloat          infval) {

  cuda_vec_project_polar_reccone(y->d_val, l->d_val, u->d_val, infval, y->length);
}

OSQPInt OSQPVectorf_in_reccone(const OSQPVectorf* y,
                               const OSQPVectorf* l,
                               const OSQPVectorf* u,
                                     OSQPFloat    infval,
                                     OSQPFloat    tol) {

  OSQPInt res;

  cuda_vec_in_reccone(y->d_val, l->d_val, u->d_val, infval, tol, y->length, &res);

  return res;
}

void OSQPVectorf_ew_reciprocal(OSQPVectorf*       b,
                               const OSQPVectorf* a) {

  if (b->length) cuda_vec_reciprocal(b->d_val, a->d_val, b->length);
}

void OSQPVectorf_ew_sqrt(OSQPVectorf* a) {

  if (a->length) cuda_vec_sqrt(a->d_val, a->length);
}

void OSQPVectorf_ew_max_vec(OSQPVectorf*       c,
                            const OSQPVectorf* a,
                            const OSQPVectorf* b) {

  if (c->length) cuda_vec_max(c->d_val, a->d_val, b->d_val, c->length);
}

void OSQPVectorf_ew_min_vec(OSQPVectorf*       c,
                            const OSQPVectorf* a,
                            const OSQPVectorf* b) {

  if (c->length) cuda_vec_min(c->d_val, a->d_val, b->d_val, c->length);
}

OSQPInt OSQPVectorf_ew_bounds_type(OSQPVectori*       iseq,
                                   const OSQPVectorf* l,
                                   const OSQPVectorf* u,
                                   OSQPFloat          tol,
                                   OSQPFloat          infval) {

  OSQPInt has_changed;

  cuda_vec_bounds_type(iseq->d_val, l->d_val, u->d_val, infval, tol, iseq->length, &has_changed);

  return has_changed;
}

void OSQPVectorf_set_scalar_if_lt(OSQPVectorf*       x,
                                  const OSQPVectorf* z,
                                  OSQPFloat          testval,
                                  OSQPFloat          newval) {

  cuda_vec_set_sc_if_lt(x->d_val, z->d_val, testval, newval, x->length);
}

void OSQPVectorf_set_scalar_if_gt(OSQPVectorf*       x,
                                  const OSQPVectorf* z,
                                  OSQPFloat          testval,
                                  OSQPFloat          newval) {

  cuda_vec_set_sc_if_gt(x->d_val, z->d_val, testval, newval, x->length);
}

