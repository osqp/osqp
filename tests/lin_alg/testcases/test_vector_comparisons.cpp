#include "test_lin_alg.h"
#include "lin_alg_data.h"

TEST_CASE("Vector: Equality", "[vector],[operation]")
{
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  // Create the vectors
  OSQPVectorf_ptr v1{OSQPVectorf_malloc(data->test_vec_ops_n)};
  OSQPVectorf_ptr v2{OSQPVectorf_malloc(data->test_vec_ops_n)};
  OSQPVectorf_ptr v3{OSQPVectorf_malloc(data->test_vec_ops_n-2)};
  OSQPVectorf_ptr v4{OSQPVectorf_malloc(data->test_vec_ops_n-2)};

  //copy data vectors into OSQPVectors
  OSQPVectorf_from_raw(v1.get(), data->test_vec_ops_v1);
  OSQPVectorf_from_raw(v2.get(), data->test_vec_ops_v2);
  OSQPVectorf_from_raw(v3.get(), data->test_vec_ops_v2);
  OSQPVectorf_from_raw(v4.get(), data->test_vec_ops_v2);

  mu_assert("Vectors should be equal",
            OSQPVectorf_is_eq(v1.get(), v1.get(), TESTS_TOL));

  mu_assert("Vectors should be equal",
            OSQPVectorf_is_eq(v3.get(), v4.get(), TESTS_TOL));

  mu_assert("Vectors should not be equal",
            OSQPVectorf_is_eq(v1.get(), v2.get(), TESTS_TOL) == 0);

  mu_assert("Vectors should not be equal",
            OSQPVectorf_is_eq(v2.get(), v3.get(), TESTS_TOL) == 0);
}
