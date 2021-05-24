#include <stdio.h>
#include "osqp.h"
#include "lin_alg.h"
#include "minunit.h"
#include "lin_alg/data.h"

#ifndef CUDA_SUPPORT

#include "csc_utils.h"


static const char* test_constr_sparse_mat() {

  c_float *Adns; // Conversion to dense matrix

  OSQPVectorf *v1, *v2;
  c_int mn;

  lin_alg_sols_data *data = generate_problem_lin_alg_sols_data();

  mn = data->test_sp_matrix_A->m * data->test_sp_matrix_A->n;

  // Convert sparse to dense
  Adns = csc_to_dns(data->test_sp_matrix_A);

  //copy data into OSQPVectors
  v1 = OSQPVectorf_new(Adns, mn);
  v2 = OSQPVectorf_new(data->test_sp_matrix_Adns, mn);

  // Compute norm of the elementwise difference with
  mu_assert("Linear algebra tests: error in constructing sparse/dense matrix!",
            OSQPVectorf_norm_inf_diff(v1, v2) < TESTS_TOL);

  // Free memory
  c_free(Adns); // because of vars from file matrices.h
  OSQPVectorf_free(v1);
  OSQPVectorf_free(v2);
  clean_problem_lin_alg_sols_data(data);

  return 0;
}

#endif /* ifndef CUDA_SUPPORT */

static const char* test_vec_operations() {

  c_float  scresult, scref;
  OSQPVectorf *v1, *v2, *ref, *result;
  lin_alg_sols_data *data = generate_problem_lin_alg_sols_data();

  //allocate space for two vectors, results and reference results
  //for each vector operations
  v1     = OSQPVectorf_malloc(data->test_vec_ops_n);
  v2     = OSQPVectorf_malloc(data->test_vec_ops_n);
  ref    = OSQPVectorf_malloc(data->test_vec_ops_n);
  result = OSQPVectorf_malloc(data->test_vec_ops_n);

  //copy data vectors into OSQPVectors
  OSQPVectorf_from_raw(v1, data->test_vec_ops_v1);
  OSQPVectorf_from_raw(v2, data->test_vec_ops_v2);


  // scaled additions
  //------------------
  OSQPVectorf_add_scaled(result, data->test_vec_ops_sc1,v1,data->test_vec_ops_sc2,v2);
  OSQPVectorf_from_raw(ref, data->test_vec_ops_add_scaled);

  mu_assert(
    "Linear algebra tests: error in vector operation, adding scaled vector",
    OSQPVectorf_norm_inf_diff(ref, result) < TESTS_TOL);

  // Norm_inf of the difference
  mu_assert(
    "Linear algebra tests: error in vector operation, norm_inf of difference",
    c_absval(OSQPVectorf_norm_inf_diff(v1,v2) - data->test_vec_ops_norm_inf_diff) < TESTS_TOL);

  // norm_inf
  scresult = OSQPVectorf_norm_inf(v1);
  scref    = data->test_vec_ops_norm_inf;
  mu_assert("Linear algebra tests: error in vector operation, norm_inf",
            c_absval(scresult - scref) < TESTS_TOL);


  // Elementwise reciprocal
  //-----------------------

  OSQPVectorf_ew_reciprocal(result, v1);
  OSQPVectorf_from_raw(ref, data->test_vec_ops_ew_reciprocal);

  mu_assert(
    "Linear algebra tests: error in vector operation, elementwise reciprocal",
    OSQPVectorf_norm_inf_diff(ref, result) < TESTS_TOL);


  // dot product reciprocal
  //-----------------------
  scresult = OSQPVectorf_dot_prod(v1,v2);
  scref    = data->test_vec_ops_vec_prod;
  mu_assert("Linear algebra tests: error in vector operation, vector product",
            c_absval(scresult - scref) < TESTS_TOL);

  // Elementwise maximum
  //-----------------------
  OSQPVectorf_ew_max_vec(result, v1, v2);
  OSQPVectorf_from_raw(ref, data->test_vec_ops_ew_max_vec);

  mu_assert(
    "Linear algebra tests: error in vector operation, elementwise maximum between vectors",
    OSQPVectorf_norm_inf_diff(result, ref) < TESTS_TOL);

  // Elementwise maximum
  //-----------------------
  OSQPVectorf_ew_min_vec(result, v1, v2);
  OSQPVectorf_from_raw(ref, data->test_vec_ops_ew_min_vec);

  mu_assert(
    "Linear algebra tests: error in vector operation, elementwise minimum between vectors",
    OSQPVectorf_norm_inf_diff(result, ref) < TESTS_TOL);

  // cleanup
  OSQPVectorf_free(v1);
  OSQPVectorf_free(v2);
  OSQPVectorf_free(ref);
  OSQPVectorf_free(result);
  clean_problem_lin_alg_sols_data(data);

  return 0;
}

static const char* test_mat_operations() {

  OSQPMatrix *A, *Ad, *dA; // Matrices used for tests
  OSQPMatrix *refM;
  OSQPVectorf *d, *refv, *resultv;

  lin_alg_sols_data *data = generate_problem_lin_alg_sols_data();


  // Import matrices (3 copies) and vector data
  A  = OSQPMatrix_new_from_csc(data->test_mat_ops_A,0); //asymmetric
  Ad = OSQPMatrix_new_from_csc(data->test_mat_ops_A,0); //asymmetric
  dA = OSQPMatrix_new_from_csc(data->test_mat_ops_A,0); //asymmetric
  d  = OSQPVectorf_new(data->test_mat_ops_d, data->test_mat_ops_n);

#ifndef CUDA_SUPPORT

  // Premultiply matrix A
  OSQPMatrix_lmult_diag(dA, d);
  refM = OSQPMatrix_new_from_csc(data->test_mat_ops_prem_diag, 0); //asymmetric
  mu_assert(
    "Linear algebra tests: error in matrix operation, premultiply diagonal",
    OSQPMatrix_is_eq(dA, refM, TESTS_TOL));
  OSQPMatrix_free(refM);


  // Postmultiply matrix A
  OSQPMatrix_rmult_diag(Ad, d);
  refM = OSQPMatrix_new_from_csc(data->test_mat_ops_postm_diag, 0); //asymmetric
  mu_assert(
    "Linear algebra tests: error in matrix operation, postmultiply diagonal",
    OSQPMatrix_is_eq(Ad, refM, TESTS_TOL));
  OSQPMatrix_free(refM);

#endif /* ifndef CUDA_SUPPORT */

  // Maximum norm over columns
  resultv = OSQPVectorf_malloc(data->test_mat_ops_n);
  OSQPMatrix_col_norm_inf(A, resultv);
  refv    = OSQPVectorf_new(data->test_mat_ops_inf_norm_cols, data->test_mat_ops_n);
  mu_assert(
    "Linear algebra tests: error in matrix operation, max norm over columns",
    OSQPVectorf_norm_inf_diff(refv, resultv) < TESTS_TOL);
  OSQPVectorf_free(resultv);
  OSQPVectorf_free(refv);

  // Maximum norm over rows
  resultv = OSQPVectorf_malloc(data->test_mat_ops_n);
  OSQPMatrix_row_norm_inf(A, resultv);
  refv    = OSQPVectorf_new(data->test_mat_ops_inf_norm_rows, data->test_mat_ops_n);
  mu_assert(
    "Linear algebra tests: error in matrix operation, max norm over rows",
    OSQPVectorf_norm_inf_diff(refv, resultv) < TESTS_TOL);
  OSQPVectorf_free(resultv);
  OSQPVectorf_free(refv);


  // cleanup
  OSQPVectorf_free(d);
  OSQPMatrix_free(A);
  OSQPMatrix_free(Ad);
  OSQPMatrix_free(dA);
  clean_problem_lin_alg_sols_data(data);

  return 0;
}

static const char* test_mat_vec_multiplication() {

  OSQPVectorf *x, *y;
  OSQPVectorf *ref, *result;
  OSQPMatrix  *Pu, *A;

  lin_alg_sols_data *data = generate_problem_lin_alg_sols_data();


  //import data
  A  = OSQPMatrix_new_from_csc(data->test_mat_vec_A, 0); //asymmetric
  Pu = OSQPMatrix_new_from_csc(data->test_mat_vec_Pu, 1); //symmetric
  x  = OSQPVectorf_new(data->test_mat_vec_x, data->test_mat_vec_n);
  y  = OSQPVectorf_new(data->test_mat_vec_y, data->test_mat_vec_m);

  // Matrix-vector multiplication:  y = Ax
  result = OSQPVectorf_malloc(data->test_mat_vec_m);
  OSQPMatrix_Axpy(A, x, result, 1.0, 0.0);
  ref = OSQPVectorf_new(data->test_mat_vec_Ax, data->test_mat_vec_m);
  mu_assert(
    "Linear algebra tests: error in matrix-vector operation, matrix-vector multiplication",
    OSQPVectorf_norm_inf_diff(result, ref) < TESTS_TOL);
  OSQPVectorf_free(ref);
  OSQPVectorf_free(result);

  // Cumulative matrix-vector multiplication:  y += Ax
  result = OSQPVectorf_new(data->test_mat_vec_y, data->test_mat_vec_m);
  ref    = OSQPVectorf_new(data->test_mat_vec_Ax_cum, data->test_mat_vec_m);
  OSQPMatrix_Axpy(A, x, result, 1.0, 1.0);
  mu_assert(
    "Linear algebra tests: error in matrix-vector operation, cumulative matrix-vector multiplication",
    OSQPVectorf_norm_inf_diff(result, ref) < TESTS_TOL);
  OSQPVectorf_free(result);
  OSQPVectorf_free(ref);

  // Matrix-transpose-vector multiplication:  x = A'*y
  result = OSQPVectorf_malloc(data->test_mat_vec_n);
  OSQPMatrix_Atxpy(A, y, result, 1.0, 0.0);
  ref = OSQPVectorf_new(data->test_mat_vec_ATy,data->test_mat_vec_n);
  mu_assert(
    "Linear algebra tests: error in matrix-vector operation, matrix-transpose-vector multiplication",
    OSQPVectorf_norm_inf_diff(result, ref) < TESTS_TOL);
  OSQPVectorf_free(ref);
  OSQPVectorf_free(result);

  // Cumulative matrix-transpose-vector multiplication:  x += A'*y
  result = OSQPVectorf_new(data->test_mat_vec_x, data->test_mat_vec_n);
  ref    = OSQPVectorf_new(data->test_mat_vec_ATy_cum, data->test_mat_vec_n);
  OSQPMatrix_Atxpy(A, y, result, 1.0, 1.0);
  mu_assert(
    "Linear algebra tests: error in matrix-vector operation, cumulative matrix-transpose-vector multiplication",
    OSQPVectorf_norm_inf_diff(result,ref) < TESTS_TOL);
  OSQPVectorf_free(ref);
  OSQPVectorf_free(result);

  // Symmetric-matrix-vector multiplication (only upper part is stored)
  result = OSQPVectorf_malloc(data->test_mat_vec_n);
  OSQPMatrix_Axpy(Pu, x, result, 1.0, 0.0);
  ref = OSQPVectorf_new(data->test_mat_vec_Px,data->test_mat_vec_n);
  mu_assert(
    "Linear algebra tests: error in matrix-vector operation, symmetric matrix-vector multiplication",
  OSQPVectorf_norm_inf_diff(result, ref) < TESTS_TOL);
  OSQPVectorf_free(ref);
  OSQPVectorf_free(result);

  // Cumulative symmetric-matrix-vector multiplication x += Px
  result = OSQPVectorf_new(data->test_mat_vec_x, data->test_mat_vec_n);
  ref    = OSQPVectorf_new(data->test_mat_vec_Px_cum, data->test_mat_vec_n);
  OSQPMatrix_Axpy(Pu, x, result, 1.0,1.0);

  mu_assert(
    "Linear algebra tests: error in matrix-vector operation, cumulative symmetric matrix-vector multiplication",
  OSQPVectorf_norm_inf_diff(result, ref) < TESTS_TOL);
  OSQPVectorf_free(ref);
  OSQPVectorf_free(result);


  // cleanup
  OSQPVectorf_free(x);
  OSQPVectorf_free(y);
  OSQPMatrix_free(A);
  OSQPMatrix_free(Pu);

  clean_problem_lin_alg_sols_data(data);

  return 0;
}

static const char* test_quad_form_upper_triang() {

  c_float val;
  lin_alg_sols_data *data = generate_problem_lin_alg_sols_data();
  OSQPMatrix* P  = OSQPMatrix_new_from_csc(data->test_qpform_Pu, 1); //triu;
  OSQPVectorf* x = OSQPVectorf_new(data->test_qpform_x, data->test_mat_vec_n);

  // Compute quadratic form
  val = OSQPMatrix_quad_form(P, x);

  mu_assert(
    "Linear algebra tests: error in computing quadratic form using upper triangular matrix!",
    (c_absval(val - data->test_qpform_value) < TESTS_TOL));

  // cleanup
  OSQPMatrix_free(P);
  OSQPVectorf_free(x);
  clean_problem_lin_alg_sols_data(data);

  return 0;
}

static const char* test_lin_alg()
{
  // initialize algebra libraries
  osqp_algebra_init_libs(0);

#ifndef CUDA_SUPPORT
  mu_run_test(test_constr_sparse_mat);
#endif

  mu_run_test(test_vec_operations);
  mu_run_test(test_mat_operations);
  mu_run_test(test_mat_vec_multiplication);
  mu_run_test(test_quad_form_upper_triang);

  // free algebra libraries
  osqp_algebra_free_libs();

  return 0;
}
