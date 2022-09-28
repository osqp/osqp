#include "test_lin_alg.h"
#include "lin_alg_data.h"

#ifndef OSQP_ALGEBRA_CUDA

TEST_CASE("Matrix: Sparse matrix construction", "[matrix][construction]") {

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


TEST_CASE("Matrix: Submatrix creation", "[matrix][creation") {
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  OSQPMatrix_ptr lA{OSQPMatrix_new_from_csc(data->test_mat_vec_A,0)}; //asymmetric

  SECTION("submatrix: 4 rows") {
    OSQPInt        numInd = data->test_submat_A4_num;
    OSQPInt*       ind    = data->test_submat_A4_ind;
    OSQPCscMatrix* refCsc = data->test_submat_A4;

    // Reference matrix
    OSQPMatrix_ptr  refM{OSQPMatrix_new_from_csc(refCsc, 0)};

    // Extract submatrix
    OSQPVectori_ptr rows{OSQPVectori_new(ind, data->test_mat_vec_m)};
    OSQPMatrix_ptr  subM{OSQPMatrix_submatrix_byrows(lA.get(), rows.get())};

    mu_assert("Linear algebra tests: error in matrix operation, submatrix_by_rows has wrong column count",
              OSQPMatrix_get_n(subM.get()) == OSQPMatrix_get_n(lA.get()));

    mu_assert("Linear algebra tests: error in matrix operation, submatrix_by_rows has wrong row count",
              OSQPMatrix_get_m(subM.get()) == numInd);

    mu_assert("Linear algebra tests: error in matrix operation,submatrix_by_rows has wrong data",
              OSQPMatrix_is_eq(subM.get(), refM.get(), TESTS_TOL));
  }

  SECTION("submatrix: All (5) rows") {
    OSQPInt        numInd = data->test_submat_A5_num;
    OSQPInt*       ind    = data->test_submat_A5_ind;
    OSQPCscMatrix* refCsc = data->test_submat_A5;

    // Reference matrix
    OSQPMatrix_ptr  refM{OSQPMatrix_new_from_csc(refCsc, 0)};

    // Extract submatrix
    OSQPVectori_ptr rows{OSQPVectori_new(ind, data->test_mat_vec_m)};
    OSQPMatrix_ptr  subM{OSQPMatrix_submatrix_byrows(lA.get(), rows.get())};

    mu_assert("Linear algebra tests: error in matrix operation, submatrix_by_rows has wrong column count",
              OSQPMatrix_get_n(subM.get()) == OSQPMatrix_get_n(lA.get()));

    mu_assert("Linear algebra tests: error in matrix operation, submatrix_by_rows has wrong row count",
              OSQPMatrix_get_m(subM.get()) == numInd);

    mu_assert("Linear algebra tests: error in matrix operation,submatrix_by_rows has wrong data",
              OSQPMatrix_is_eq(subM.get(), refM.get(), TESTS_TOL));
  }

  SECTION("submatrix: Half (3) rows") {
    OSQPInt        numInd = data->test_submat_A3_num;
    OSQPInt*       ind    = data->test_submat_A3_ind;
    OSQPCscMatrix* refCsc = data->test_submat_A3;

    // Reference matrix
    OSQPMatrix_ptr  refM{OSQPMatrix_new_from_csc(refCsc, 0)};

    // Extract submatrix
    OSQPVectori_ptr rows{OSQPVectori_new(ind, data->test_mat_vec_m)};
    OSQPMatrix_ptr  subM{OSQPMatrix_submatrix_byrows(lA.get(), rows.get())};

    mu_assert("Linear algebra tests: error in matrix operation, submatrix_by_rows has wrong column count",
              OSQPMatrix_get_n(subM.get()) == OSQPMatrix_get_n(lA.get()));

    mu_assert("Linear algebra tests: error in matrix operation, submatrix_by_rows has wrong row count",
              OSQPMatrix_get_m(subM.get()) == numInd);

    mu_assert("Linear algebra tests: error in matrix operation,submatrix_by_rows has wrong data",
              OSQPMatrix_is_eq(subM.get(), refM.get(), TESTS_TOL));
  }

  SECTION("submatrix: No rows") {
    OSQPInt        numInd = data->test_submat_A0_num;
    OSQPInt*       ind    = data->test_submat_A0_ind;
    OSQPCscMatrix* refCsc = data->test_submat_A0;

    // Reference matrix
    OSQPMatrix_ptr  refM{OSQPMatrix_new_from_csc(refCsc, 0)};

    // Extract submatrix
    OSQPVectori_ptr rows{OSQPVectori_new(ind, data->test_mat_vec_m)};
    OSQPMatrix_ptr  subM{OSQPMatrix_submatrix_byrows(lA.get(), rows.get())};

    mu_assert("Linear algebra tests: error in matrix operation, submatrix_by_rows has wrong column count",
              OSQPMatrix_get_n(subM.get()) == OSQPMatrix_get_n(lA.get()));

    mu_assert("Linear algebra tests: error in matrix operation, submatrix_by_rows has wrong row count",
              OSQPMatrix_get_m(subM.get()) == numInd);

    mu_assert("Linear algebra tests: error in matrix operation,submatrix_by_rows has wrong data",
              OSQPMatrix_is_eq(subM.get(), refM.get(), TESTS_TOL));
  }
}

#ifndef OSQP_ALGEBRA_CUDA
TEST_CASE("Matrix: Diagonal extraction", "[matrix]") {
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  SECTION("Square non-symmetric") {
    OSQPMatrix_ptr A{OSQPMatrix_new_from_csc(data->test_mat_ops_diag_A, 0)};
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_mat_ops_diag_dA, data->test_mat_ops_diag_n)};

    OSQPVectorf_ptr res{OSQPVectorf_malloc(data->test_mat_ops_diag_n)};

    OSQPMatrix_extract_diag(A.get(), res.get());

    mu_assert("Error in extracted diagonal",
              OSQPVectorf_is_eq(ref.get(), res.get(), TESTS_TOL));
  }

  SECTION("Square symmetric, triangular") {
    OSQPMatrix_ptr P{OSQPMatrix_new_from_csc(data->test_mat_ops_diag_Pu, 1)};
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_mat_ops_diag_dP, data->test_mat_ops_diag_n)};

    OSQPVectorf_ptr res{OSQPVectorf_malloc(data->test_mat_ops_diag_n)};

    OSQPMatrix_extract_diag(P.get(), res.get());

    mu_assert("Error in extracted diagonal",
              OSQPVectorf_is_eq(ref.get(), res.get(), TESTS_TOL));
  }
}
#endif

#ifndef OSQP_ALGEBRA_CUDA
TEST_CASE("Matrix: AtDA Diagonal extraction", "[matrix]") {
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  OSQPInt m = data->test_mat_ops_diag_m;
  OSQPInt n = data->test_mat_ops_diag_n;

  // Matrix to use
  OSQPMatrix_ptr A{OSQPMatrix_new_from_csc(data->test_mat_ops_diag_Ar, 0)};

  // Result vector
  OSQPVectorf_ptr res{OSQPVectorf_malloc(n)};

  SECTION("Unity diagonal element") {
    OSQPVectorf_ptr D{OSQPVectorf_new(data->test_vec_ops_ones, m)};         // Diagonal
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_mat_ops_diag_AtA, n)};   // Reference

    OSQPMatrix_AtDA_extract_diag(A.get(), D.get(), res.get());

    mu_assert("Error in extracted diagonal",
              OSQPVectorf_is_eq(ref.get(), res.get(), TESTS_TOL));
  }

  SECTION("Simple diagonal scaling") {
    OSQPVectorf_ptr D{OSQPVectorf_new(data->test_vec_ops_vn, m)};           // Diagonal
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_mat_ops_diag_AtRnA, n)}; // Reference

    OSQPMatrix_AtDA_extract_diag(A.get(), D.get(), res.get());

    mu_assert("Error in extracted diagonal",
              OSQPVectorf_is_eq(ref.get(), res.get(), TESTS_TOL));
  }

  SECTION("Complicated diagonal scaling") {
    OSQPVectorf_ptr D{OSQPVectorf_new(data->test_vec_ops_v1, m)};          // Diagonal
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_mat_ops_diag_AtRA, n)}; // Reference

    OSQPMatrix_AtDA_extract_diag(A.get(), D.get(), res.get());

    mu_assert("Error in extracted diagonal",
              OSQPVectorf_is_eq(ref.get(), res.get(), TESTS_TOL));
  }
}
#endif

TEST_CASE("Matrix: Equality test", "[matrix]") {
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  // Import data matrices (4 copies) and vector data
  // Matrices used for tests
  OSQPMatrix_ptr A{OSQPMatrix_new_from_csc(data->test_mat_ops_A, 0)}; //asymmetric

  OSQPMatrix_ptr vA{OSQPMatrix_new_from_csc(data->test_mat_vec_A,0)}; //asymmetric
  OSQPMatrix_ptr A4{OSQPMatrix_new_from_csc(data->test_submat_A4,0)}; //asymmetric
  OSQPMatrix_ptr A5{OSQPMatrix_new_from_csc(data->test_submat_A5,0)}; //asymmetric

  SECTION("Matrix equality") {
    mu_assert("Linear algebra tests: error in matrix equality - same matrix not equal",
              OSQPMatrix_is_eq(vA.get(), vA.get(), TESTS_TOL));

    mu_assert("Linear algebra tests: error in matrix equality - similar matrices not equal",
              OSQPMatrix_is_eq(vA.get(), A5.get(), TESTS_TOL));
  }

  SECTION("Row/column count differents") {
    mu_assert("Linear algebra tests: error in matrix equality - different row count",
              OSQPMatrix_is_eq(vA.get(), A4.get(), TESTS_TOL) == 0);

    mu_assert("Linear algebra tests: error in matrix equality - different column count",
              OSQPMatrix_is_eq(vA.get(), A.get(), TESTS_TOL) == 0);
  }

  SECTION("All values different") {
    OSQPMatrix_mult_scalar(vA.get(), 3.0);

    mu_assert("Linear algebra tests: error in matrix equality - values changed",
              OSQPMatrix_is_eq(vA.get(), A5.get(), TESTS_TOL) == 0);
  }

  SECTION("Initial values different") {
    std::unique_ptr<OSQPFloat[]> mulval(new OSQPFloat[5]);
    mulval[0] = 4.0;
    mulval[1] = 1.0;
    mulval[2] = 1.0;
    mulval[3] = 1.0;
    mulval[4] = 1.0;

    OSQPVectorf_ptr mulvec{OSQPVectorf_new(mulval.get(), 5)};

    OSQPMatrix_lmult_diag(vA.get(), mulvec.get());

    mu_assert("Linear algebra tests: error in matrix equality - initial values changed",
              OSQPMatrix_is_eq(vA.get(), A5.get(), TESTS_TOL) == 0);
  }

  SECTION("Final values different") {
    std::unique_ptr<OSQPFloat[]> mulval(new OSQPFloat[5]);
    mulval[0] = 1.0;
    mulval[1] = 1.0;
    mulval[2] = 1.0;
    mulval[3] = 1.0;
    mulval[4] = 4.0;

    OSQPVectorf_ptr mulvec{OSQPVectorf_new(mulval.get(), 5)};

    OSQPMatrix_lmult_diag(vA.get(), mulvec.get());

    mu_assert("Linear algebra tests: error in matrix equality - final values changed",
              OSQPMatrix_is_eq(vA.get(), A5.get(), TESTS_TOL) == 0);
  }

}

TEST_CASE("Matrix: Operations", "[matrix][operation]")  {
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  // Import data matrices (4 copies) and vector data
  // Matrices used for tests
  OSQPMatrix_ptr A{OSQPMatrix_new_from_csc(data->test_mat_ops_A, 0)}; //asymmetric
  OSQPMatrix_ptr Ad{OSQPMatrix_new_from_csc(data->test_mat_ops_A,0)}; //asymmetric
  OSQPMatrix_ptr dA{OSQPMatrix_new_from_csc(data->test_mat_ops_A,0)}; //asymmetric
  OSQPMatrix_ptr sA{OSQPMatrix_new_from_csc(data->test_mat_ops_A,0)}; //asymmetric

  OSQPVectorf_ptr d{OSQPVectorf_new(data->test_mat_ops_d, data->test_mat_ops_n)};

  // Result vectors
  OSQPMatrix_ptr  refM{nullptr};
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
