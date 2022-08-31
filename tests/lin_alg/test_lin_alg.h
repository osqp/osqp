#include <stdio.h>
#include "osqp.h"
#include "lin_alg.h"
#include "osqp_tester.h"
#include "lin_alg/data.h"

#ifndef OSQP_ALGEBRA_CUDA

#ifdef __cplusplus
extern "C" {
#endif
  #include "csc_utils.h"
#ifdef __cplusplus
}
#endif

void test_constr_sparse_mat() {

  OSQPFloat* Adns; // Conversion to dense matrix

  OSQPVectorf_ptr v1{nullptr};
  OSQPVectorf_ptr v2{nullptr};
  OSQPInt mn;

  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  mn = data->test_sp_matrix_A->m * data->test_sp_matrix_A->n;

  // Convert sparse to dense
  Adns = csc_to_dns(data->test_sp_matrix_A);

  //copy data into OSQPVectors
  v1.reset(OSQPVectorf_new(Adns, mn));
  v2.reset(OSQPVectorf_new(data->test_sp_matrix_Adns, mn));

  // Compute norm of the elementwise difference with
  mu_assert("Linear algebra tests: error in constructing sparse/dense matrix!",
            OSQPVectorf_norm_inf_diff(v1.get(), v2.get()) < TESTS_TOL);

  // Free memory
  c_free(Adns); // because of vars from file matrices.h
}

#endif /* ifndef OSQP_ALGEBRA_CUDA */

void test_vec_operations() {

  OSQPFloat scresult;
  OSQPFloat scref;

  OSQPVectorf_ptr v1{nullptr};
  OSQPVectorf_ptr v2{nullptr};
  OSQPVectorf_ptr ref{nullptr};
  OSQPVectorf_ptr result{nullptr};

  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  //allocate space for two vectors, results and reference results
  //for each vector operations
  v1.reset(OSQPVectorf_malloc(data->test_vec_ops_n));
  v2.reset(OSQPVectorf_malloc(data->test_vec_ops_n));
  ref.reset(OSQPVectorf_malloc(data->test_vec_ops_n));
  result.reset(OSQPVectorf_malloc(data->test_vec_ops_n));

  //copy data vectors into OSQPVectors
  OSQPVectorf_from_raw(v1.get(), data->test_vec_ops_v1);
  OSQPVectorf_from_raw(v2.get(), data->test_vec_ops_v2);


  // scaled additions
  //------------------
  OSQPVectorf_add_scaled(result.get(), data->test_vec_ops_sc1, v1.get(), data->test_vec_ops_sc2, v2.get());
  OSQPVectorf_from_raw(ref.get(), data->test_vec_ops_add_scaled);

  mu_assert(
    "Linear algebra tests: error in vector operation, adding scaled vector",
    OSQPVectorf_norm_inf_diff(ref.get(), result.get()) < TESTS_TOL);

  // Norm_inf of the difference
  mu_assert(
    "Linear algebra tests: error in vector operation, norm_inf of difference",
    c_absval(OSQPVectorf_norm_inf_diff(v1.get(), v2.get()) - data->test_vec_ops_norm_inf_diff) < TESTS_TOL);

  // norm_inf
  scresult = OSQPVectorf_norm_inf(v1.get());
  scref    = data->test_vec_ops_norm_inf;
  mu_assert("Linear algebra tests: error in vector operation, norm_inf",
            c_absval(scresult - scref) < TESTS_TOL);


  // Elementwise reciprocal
  //-----------------------

  OSQPVectorf_ew_reciprocal(result.get(), v1.get());
  OSQPVectorf_from_raw(ref.get(), data->test_vec_ops_ew_reciprocal);

  mu_assert(
    "Linear algebra tests: error in vector operation, elementwise reciprocal",
    OSQPVectorf_norm_inf_diff(ref.get(), result.get()) < TESTS_TOL);


  // dot product reciprocal
  //-----------------------
  scresult = OSQPVectorf_dot_prod(v1.get(), v2.get());
  scref    = data->test_vec_ops_vec_prod;
  mu_assert("Linear algebra tests: error in vector operation, vector product",
            c_absval(scresult - scref) < TESTS_TOL);

  // Elementwise maximum
  //-----------------------
  OSQPVectorf_ew_max_vec(result.get(), v1.get(), v2.get());
  OSQPVectorf_from_raw(ref.get(), data->test_vec_ops_ew_max_vec);

  mu_assert(
    "Linear algebra tests: error in vector operation, elementwise maximum between vectors",
    OSQPVectorf_norm_inf_diff(result.get(), ref.get()) < TESTS_TOL);

  // // Elementwise minimum
  // //-----------------------
  // OSQPVectorf_ew_min_vec(result, v1, v2);
  // OSQPVectorf_from_raw(ref, data->test_vec_ops_ew_min_vec);

  // mu_assert(
  //   "Linear algebra tests: error in vector operation, elementwise minimum between vectors",
  //   OSQPVectorf_norm_inf_diff(result, ref) < TESTS_TOL);
}

void test_mat_operations() {
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  // Import data matrices (4 copies) and vector data
  // Matrices used for tests
  OSQPMatrix_ptr A{OSQPMatrix_new_from_csc(data->test_mat_ops_A, 0)}; //asymmetric
  OSQPMatrix_ptr Ad{OSQPMatrix_new_from_csc(data->test_mat_ops_A,0)}; //asymmetric
  OSQPMatrix_ptr dA{OSQPMatrix_new_from_csc(data->test_mat_ops_A,0)}; //asymmetric
  OSQPMatrix_ptr sA{OSQPMatrix_new_from_csc(data->test_mat_ops_A,0)}; //asymmetric

  OSQPVectorf_ptr d{OSQPVectorf_new(data->test_mat_ops_d, data->test_mat_ops_n)};

  // Result vectors
  OSQPMatrix_ptr refM{nullptr};
  OSQPVectorf_ptr refv{nullptr};
  OSQPVectorf_ptr resultv{nullptr};

#ifndef OSQP_ALGEBRA_CUDA

  // Scalar multiply every element in A
  refM.reset(OSQPMatrix_new_from_csc(data->test_mat_ops_scaled, 0)); //asymmetric

  OSQPMatrix_mult_scalar(sA.get(), 2.0);
  mu_assert(
    "Linear algebra tests: error in matrix operation, scalar multiply",
    OSQPMatrix_is_eq(sA.get(), refM.get(), TESTS_TOL));

  // Premultiply matrix A
  refM.reset(OSQPMatrix_new_from_csc(data->test_mat_ops_prem_diag, 0)); //asymmetric

  OSQPMatrix_lmult_diag(dA.get(), d.get());
  mu_assert(
    "Linear algebra tests: error in matrix operation, premultiply diagonal",
    OSQPMatrix_is_eq(dA.get(), refM.get(), TESTS_TOL));

  // Postmultiply matrix A
  refM.reset(OSQPMatrix_new_from_csc(data->test_mat_ops_postm_diag, 0)); //asymmetric

  OSQPMatrix_rmult_diag(Ad.get(), d.get());
  mu_assert(
    "Linear algebra tests: error in matrix operation, postmultiply diagonal",
    OSQPMatrix_is_eq(Ad.get(), refM.get(), TESTS_TOL));

#endif /* ifndef OSQP_ALGEBRA_CUDA */

  // Maximum norm over columns
  refv.reset(OSQPVectorf_new(data->test_mat_ops_inf_norm_cols, data->test_mat_ops_n));
  resultv.reset(OSQPVectorf_malloc(data->test_mat_ops_n));

  OSQPMatrix_col_norm_inf(A.get(), resultv.get());
  mu_assert(
    "Linear algebra tests: error in matrix operation, max norm over columns",
    OSQPVectorf_norm_inf_diff(refv.get(), resultv.get()) < TESTS_TOL);

  // Maximum norm over rows
  refv.reset(OSQPVectorf_new(data->test_mat_ops_inf_norm_rows, data->test_mat_ops_n));
  resultv.reset(OSQPVectorf_malloc(data->test_mat_ops_n));

  OSQPMatrix_row_norm_inf(A.get(), resultv.get());
  mu_assert(
    "Linear algebra tests: error in matrix operation, max norm over rows",
    OSQPVectorf_norm_inf_diff(refv.get(), resultv.get()) < TESTS_TOL);
}

void test_mat_vec_multiplication() {
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  // Import data
  OSQPMatrix_ptr  A{OSQPMatrix_new_from_csc(data->test_mat_vec_A, 0)};     //asymmetric
  OSQPMatrix_ptr  Pu{OSQPMatrix_new_from_csc(data->test_mat_vec_Pu, 1)};   //symmetric
  OSQPVectorf_ptr x{OSQPVectorf_new(data->test_mat_vec_x, data->test_mat_vec_n)};
  OSQPVectorf_ptr y{OSQPVectorf_new(data->test_mat_vec_y, data->test_mat_vec_m)};

  OSQPVectorf_ptr ref{nullptr};
  OSQPVectorf_ptr result{nullptr};

  // Matrix-vector multiplication:  y = Ax
  ref.reset(OSQPVectorf_new(data->test_mat_vec_Ax, data->test_mat_vec_m));
  result.reset(OSQPVectorf_malloc(data->test_mat_vec_m));

  OSQPMatrix_Axpy(A.get(), x.get(), result.get(), 1.0, 0.0);
  mu_assert(
    "Linear algebra tests: error in matrix-vector operation, matrix-vector multiplication",
    OSQPVectorf_norm_inf_diff(result.get(), ref.get()) < TESTS_TOL);

  // Cumulative matrix-vector multiplication:  y += Ax
  ref.reset(OSQPVectorf_new(data->test_mat_vec_Ax_cum, data->test_mat_vec_m));
  result.reset(OSQPVectorf_new(data->test_mat_vec_y, data->test_mat_vec_m));

  OSQPMatrix_Axpy(A.get(), x.get(), result.get(), 1.0, 1.0);
  mu_assert(
    "Linear algebra tests: error in matrix-vector operation, cumulative matrix-vector multiplication",
    OSQPVectorf_norm_inf_diff(result.get(), ref.get()) < TESTS_TOL);

  // Matrix-transpose-vector multiplication:  x = A'*y
  ref.reset(OSQPVectorf_new(data->test_mat_vec_ATy,data->test_mat_vec_n));
  result.reset(OSQPVectorf_malloc(data->test_mat_vec_n));

  OSQPMatrix_Atxpy(A.get(), y.get(), result.get(), 1.0, 0.0);
  mu_assert(
    "Linear algebra tests: error in matrix-vector operation, matrix-transpose-vector multiplication",
    OSQPVectorf_norm_inf_diff(result.get(), ref.get()) < TESTS_TOL);

  // Cumulative matrix-transpose-vector multiplication:  x += A'*y
  ref.reset(OSQPVectorf_new(data->test_mat_vec_ATy_cum, data->test_mat_vec_n));
  result.reset(OSQPVectorf_new(data->test_mat_vec_x, data->test_mat_vec_n));

  OSQPMatrix_Atxpy(A.get(), y.get(), result.get(), 1.0, 1.0);
  mu_assert(
    "Linear algebra tests: error in matrix-vector operation, cumulative matrix-transpose-vector multiplication",
    OSQPVectorf_norm_inf_diff(result.get(), ref.get()) < TESTS_TOL);

  // Symmetric-matrix-vector multiplication (only upper part is stored)
  ref.reset(OSQPVectorf_new(data->test_mat_vec_Px,data->test_mat_vec_n));
  result.reset(OSQPVectorf_malloc(data->test_mat_vec_n));

  OSQPMatrix_Axpy(Pu.get(), x.get(), result.get(), 1.0, 0.0);
  mu_assert(
    "Linear algebra tests: error in matrix-vector operation, symmetric matrix-vector multiplication",
  OSQPVectorf_norm_inf_diff(result.get(), ref.get()) < TESTS_TOL);

  // Cumulative symmetric-matrix-vector multiplication x += Px
  ref.reset(OSQPVectorf_new(data->test_mat_vec_Px_cum, data->test_mat_vec_n));
  result.reset(OSQPVectorf_new(data->test_mat_vec_x, data->test_mat_vec_n));
  OSQPMatrix_Axpy(Pu.get(), x.get(), result.get(), 1.0,1.0);

  mu_assert(
    "Linear algebra tests: error in matrix-vector operation, cumulative symmetric matrix-vector multiplication",
  OSQPVectorf_norm_inf_diff(result.get(), ref.get()) < TESTS_TOL);
}

void test_empty_mat_vec() {
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  // Import data
  OSQPMatrix_ptr  Aempty{OSQPMatrix_new_from_csc(data->test_mat_no_entries, 0)};     //asymmetric
  OSQPMatrix_ptr  Anorows{OSQPMatrix_new_from_csc(data->test_mat_no_rows, 0)};   //asymmetric
  OSQPMatrix_ptr  Anocols{OSQPMatrix_new_from_csc(data->test_mat_no_cols, 0)};   //asymmetric

  OSQPVectorf_ptr z{OSQPVectorf_new(data->test_vec_zeros, 2)};
  OSQPVectorf_ptr ee{OSQPVectorf_new(data->test_vec_empty, 0)};
  OSQPVectorf_ptr e2{OSQPVectorf_new(data->test_vec_mat_empty, 2)};

  // Ensure all the matrices/vectors could be created properly
  mu_assert(
    "Linear algebra tests: error with empty matrix, empty matrix creation",
    Aempty.get() != OSQP_NULL);

  mu_assert(
    "Linear algebra tests: error with empty matrix, matrix with no rows creation",
    Anorows.get() != OSQP_NULL);

  mu_assert(
    "Linear algebra tests: error with empty matrix, matrix with no columns creation",
    Anocols.get() != OSQP_NULL);

  mu_assert(
    "Linear algebra tests: error with empty matrix, empty vector creation",
    ee.get() != OSQP_NULL);

  OSQPVectorf_ptr result{nullptr};

  // Matrix-vector multiplication of empty matrix with vector:  y = Ax
  result.reset(OSQPVectorf_new(data->test_vec_zeros, 2));

  OSQPMatrix_Axpy(Aempty.get(), e2.get(), result.get(), 1.0, 0.0);
  mu_assert(
    "Linear algebra tests: error with empty matrix, matrix-vector multiplication",
    OSQPVectorf_norm_inf_diff(result.get(), z.get()) < TESTS_TOL);

  // Matrix-transpose-vector multiplication of empty matrix with vector:  x = A'*y
  result.reset(OSQPVectorf_new(data->test_vec_zeros, 2));

  OSQPMatrix_Atxpy(Aempty.get(), e2.get(), result.get(), 1.0, 0.0);
  mu_assert(
    "Linear algebra tests: error with empty matrix, matrix-transpose-vector multiplication",
    OSQPVectorf_norm_inf_diff(result.get(), z.get()) < TESTS_TOL);

  // Matrix-vector multiplication of matrix with no rows (2 columns) with vector:  y = Ax
  result.reset(OSQPVectorf_malloc(0));

  OSQPMatrix_Axpy(Anorows.get(), e2.get(), result.get(), 1.0, 0.0);
  mu_assert(
    "Linear algebra tests: error with no row matrix, matrix-vector multiplication",
    OSQPVectorf_norm_inf_diff(result.get(), ee.get()) < TESTS_TOL);

  // Matrix-transpose-vector multiplication of matrix with no rows (2 columns) with vector:  x = A'*y
  result.reset(OSQPVectorf_new(data->test_vec_zeros, 2));

  OSQPMatrix_Atxpy(Anorows.get(), ee.get(), result.get(), 1.0, 0.0);
  mu_assert(
    "Linear algebra tests: error with no row matrix, matrix-transpose-vector multiplication",
    OSQPVectorf_norm_inf_diff(result.get(), z.get()) < TESTS_TOL);

  // Matrix-vector multiplication of matrix with no columns (2 rows) with vector:  y = Ax
  result.reset(OSQPVectorf_new(data->test_vec_zeros, 2));

  OSQPMatrix_Axpy(Anocols.get(), ee.get(), result.get(), 1.0, 0.0);
  mu_assert(
    "Linear algebra tests: error with no column matrix, matrix-vector multiplication",
    OSQPVectorf_norm_inf_diff(result.get(), z.get()) < TESTS_TOL);

  // Matrix-transpose-vector multiplication of matrix with no columns (2 rows) with vector:  x = A'*y
  result.reset(OSQPVectorf_malloc(0));

  OSQPMatrix_Atxpy(Anocols.get(), e2.get(), result.get(), 1.0, 0.0);
  mu_assert(
    "Linear algebra tests: error with no column matrix, matrix-transpose-vector multiplication",
    OSQPVectorf_norm_inf_diff(result.get(), ee.get()) < TESTS_TOL);
}

// void test_quad_form_upper_triang() {

//   OSQPFloat val;
//   lin_alg_sols_data *data = generate_problem_lin_alg_sols_data();
//   OSQPMatrix* P  = OSQPMatrix_new_from_csc(data->test_qpform_Pu, 1); //triu;
//   OSQPVectorf* x = OSQPVectorf_new(data->test_qpform_x, data->test_mat_vec_n);

//   // Compute quadratic form
//   val = OSQPMatrix_quad_form(P, x);

//   mu_assert(
//     "Linear algebra tests: error in computing quadratic form using upper triangular matrix!",
//     (c_absval(val - data->test_qpform_value) < TESTS_TOL));

//   // cleanup
//   OSQPMatrix_free(P);
//   OSQPVectorf_free(x);
//   clean_problem_lin_alg_sols_data(data);
// }
