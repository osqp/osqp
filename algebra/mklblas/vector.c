#include "osqp.h"
#include "algebra_vector.h"
#include "algebra_impl.h"
#include "stdio.h"
#include "time.h"

#define USEMKLBLAS
#ifdef USEMKLBLAS
  #include "mkl_blas.h"
  #ifdef DFLOAT
    #define blas_copy scopy
    #define blas_dot sdot
    #define blas_scale sscal
    #define blas_swap sswap
    #define blas_axpy saxpy
    #define blas_2norm snrm2
    #else
      #define blas_copy dcopy
      #define blas_dot ddot
      #define blas_scale dscal
      #define blas_swap dswap
      #define blas_axpy daxpy
      #define blas_2norm dnrm2
  #endif //dfloat endif
#endif //Usemkl endif

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
    OSQPVectorf_view_update(view,a,head,length);
  }
  return view;
}

void OSQPVectorf_view_update(OSQPVectorf *a, const OSQPVectorf *b, c_int head, c_int length){
    a->length = length;
    a->values   = b->values + head;
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

// Which of these functions are actually called elsewhere?

#ifdef USEMKLBLAS

void OSQPVectorf_copy(OSQPVectorf *b, const OSQPVectorf *a){
  const MKL_INT length = a->length;
  const MKL_INT INCX = 1; //How long should the spacing be (?)
  const MKL_INT INCY = 1;
  blas_copy(&length, a->values, &INCX, b->values, &INCY);

}

void OSQPVectori_copy(OSQPVectori *b, const OSQPVectori *a){
  const MKL_INT length = a->length;
  const MKL_INT INCX = 1; //How long should the spacing be (?)
  const MKL_INT INCY = 1;
  //blas_copy(&length, a->values, &INCX, b->values, &INCY);
}

#else

void OSQPVectorf_copy(OSQPVectorf *b, const OSQPVectorf *a){
  OSQPVectorf_from_raw(b, a->values);
}

void OSQPVectori_copy(OSQPVectori *b, const OSQPVectori *a){
  OSQPVectori_from_raw(b, a->values);
}

#endif // MKL for vector copy

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

// For now I wont deal with the raw functions, I hope they dont feel left out :(


void OSQPVectorf_set_scalar(OSQPVectorf *a, c_float sc){
  c_int i;
  c_int length = a->length;
  c_float* av  = a->values;

  for (i = 0; i < length; i++) {
    av[i] = sc;
  }
}

// I seem to have the same problem here, if I try to exploit the copy function, I will still need to compare the values and populate an initial vector

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

// Scaling a vector by a constant
#ifdef USEMKLBLAS

void OSQPVectorf_mult_scalar(OSQPVectorf *a, c_float sc){
  const MKL_INT length = a->length;
  const MKL_INT INCX = 1; //How long should the spacing be (?)
  blas_scale(&length, &sc, a->values, &INCX);
}
#else
void OSQPVectorf_mult_scalar(OSQPVectorf *a, c_float sc){
  c_int i;
  c_int length = a->length;
  c_float*  av = a->values;

  for (i = 0; i < length; i++) {
    av[i] *= sc;
  }
}
#endif // for mult scalar MKL

#ifdef USEMKLBLAS

void OSQPVectorf_plus(OSQPVectorf      *x,
                     const OSQPVectorf *a,
                     const OSQPVectorf *b){
  c_int length = a->length;
  c_float*  av = a->values;
  c_float*  bv = b->values;
  c_float*  xv = x->values;



  if (x == a){
    const MKL_INT lengthmkl = a->length;
    const MKL_INT INCX = 1;
    const MKL_INT INCY = 1;

    const c_float scalar = 1; // The number b is scaled by
    blas_axpy(&lengthmkl, &scalar, b->values, &INCX, x->values, &INCY);
  }
  else {
   /*
    for (i = 0; i < length; i++) {
      puts("hi");
      xv[i] = av[i] + bv[i];
    }
    */
    // need some help to get the test pointers working to be able to debug this
    // Little experiment here
    //printf("..\n");
    const MKL_INT lengthmkl = a->length;
    const MKL_INT INCX = 1; // The sapcing must be at least 1 here, not sure why
    const MKL_INT INCY = 1;
    const c_float scalar = 1; // The number b is scaled by
    blas_copy(&lengthmkl, a->values, &INCX, x->values, &INCY); // I copy av into xv
    blas_axpy(&lengthmkl, &scalar, b->values, &INCX, x->values, &INCY); //final addition

    // I am not sure if the extra function calls justifies the blas implementation, I can only find out when I get the benchamrk code working
    //I think I got the code to work properly now
  }
}

#else

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

#endif // MKL blas for the y = ax + y problem (i was only able to substitute one case)


#ifdef USEMKLBLAS

void OSQPVectorf_minus(OSQPVectorf       *x,
                       const OSQPVectorf *a,
                       const OSQPVectorf *b){

  const MKL_INT lengthmkl = a->length;
  const MKL_INT INCX = 1;
  const MKL_INT INCY = 1;
  c_float scalar;

  if (x == a){
    scalar = -1;
    blas_axpy(&lengthmkl, &scalar, b->values, &INCX, x->values, &INCY);
  }
  else if (x == b){
    scalar = 1.0;
    OSQPVectorf_mult_scalar(x,-1.);
    blas_axpy(&lengthmkl, &scalar, a->values, &INCX, x->values, &INCY);
  }
  else {
    scalar = -1.0;
    blas_copy(&lengthmkl, a->values, &INCX, x->values, &INCY);
    blas_axpy(&lengthmkl, &scalar, b->values, &INCX, x->values, &INCY);
  }
}

#else

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

#endif // axpy problem with minus sign

#ifdef USEMKLBLAS

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
    const MKL_INT lengthmkl = x->length;
    const MKL_INT INCX = 1; // The spacing must be at least 1 here, not sure why
    const MKL_INT INCY = 1;
    blas_axpy(&lengthmkl, &scb, b->values, &INCX, x->values, &INCY);
  }
  else {
    for (i = 0; i < length; i++) {
      xv[i] = sca * av[i] + scb * bv[i];
    }
  }

}

#else

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

#endif // axpy using scaling

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



#ifdef USEMKLBLAS

c_float OSQPVectorf_dot_prod(const OSQPVectorf *a, const OSQPVectorf *b){
  const MKL_INT length = a->length;
  const MKL_INT INCX = 1; //How long should the spacing be (?)
  const MKL_INT INCY = 1;

  return blas_dot(&length, a->values, &INCX, b->values, &INCY); // blas_dot is called the preprocesor
}

#else

//original dot product function

c_float OSQPVectorf_dot_prod(const OSQPVectorf *a, const OSQPVectorf *b) {

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
#endif

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

c_int OSQPVectorf_in_polar_reccone(const OSQPVectorf *y,
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
