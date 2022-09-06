#include "test_lin_alg.h"
#include "lin_alg_data.h"

TEST_CASE("Vector: Operations", "[vector][operation]") {

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
