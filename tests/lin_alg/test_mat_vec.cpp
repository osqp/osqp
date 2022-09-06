#include "test_lin_alg.h"
#include "lin_alg_data.h"

TEST_CASE("Matrix-vector: multiplication", "[mat-vec][operation]") {
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

TEST_CASE("Matrix-vector: Empty matrix multiplication", "[mat-vec][operation]") {
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
