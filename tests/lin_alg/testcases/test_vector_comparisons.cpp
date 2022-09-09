#include "test_lin_alg.h"
#include "lin_alg_data.h"

TEST_CASE("Vector: Equality", "[vector],[operation]")
{
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  // Create the vectors
  OSQPVectorf_ptr v1{OSQPVectorf_new(data->test_vec_ops_v1, data->test_vec_ops_n)};
  OSQPVectorf_ptr v2{OSQPVectorf_new(data->test_vec_ops_v2, data->test_vec_ops_n)};
  OSQPVectorf_ptr v3{OSQPVectorf_new(data->test_vec_ops_v2, data->test_vec_ops_n-2)};
  OSQPVectorf_ptr v4{OSQPVectorf_new(data->test_vec_ops_v2, data->test_vec_ops_n-2)};

  mu_assert("Vectors should be equal",
            OSQPVectorf_is_eq(v1.get(), v1.get(), TESTS_TOL));

  mu_assert("Vectors should be equal",
            OSQPVectorf_is_eq(v3.get(), v4.get(), TESTS_TOL));

  mu_assert("Vectors should not be equal",
            OSQPVectorf_is_eq(v1.get(), v2.get(), TESTS_TOL) == 0);

  mu_assert("Vectors should not be equal",
            OSQPVectorf_is_eq(v2.get(), v3.get(), TESTS_TOL) == 0);
}

TEST_CASE("Vector: All less than", "[vector],[operation]")
{
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  // Create the vectors
  OSQPVectorf_ptr v1{OSQPVectorf_new(data->test_vec_ops_v1, data->test_vec_ops_n)};
  OSQPVectorf_ptr v2{OSQPVectorf_new(data->test_vec_ops_v2, data->test_vec_ops_n)};

  SECTION("Negated vector")
  {
    OSQPVectorf_ptr nv1{OSQPVectorf_new(data->test_vec_ops_neg_v1, data->test_vec_ops_n)};

    mu_assert("Incorrect comparison",
            OSQPVectorf_all_leq(nv1.get(), v1.get()) == 0);
  }

  SECTION("Constant vector")
  {
    OSQPVectorf_ptr z{OSQPVectorf_new(data->test_vec_ops_zero, data->test_vec_ops_n)};
    OSQPVectorf_ptr s{OSQPVectorf_new(data->test_vec_ops_same, data->test_vec_ops_n)};

    // This depends on the sign of the constant, since s contains the constant in every element
    mu_assert("Incorrect comparison",
            OSQPVectorf_all_leq(z.get(), s.get()) == (data->test_vec_ops_sc1 >= 0));

    mu_assert("Incorrect comparison",
            OSQPVectorf_all_leq(s.get(), z.get()) == (data->test_vec_ops_sc1 < 0));
  }

  SECTION("Shifted vector")
  {
    OSQPVectorf_ptr sv1{OSQPVectorf_new(data->test_vec_ops_shift_v1, data->test_vec_ops_n)};

    mu_assert("Incorrect comparison",
            OSQPVectorf_all_leq(sv1.get(), v1.get()) == 0);

    mu_assert("Incorrect comparison",
            OSQPVectorf_all_leq(v1.get(), sv1.get()) == 1);
  }

  SECTION("Offset vectors")
  {
    OSQPVectorf_ptr sv1{OSQPVectorf_new(data->test_vec_ops_shift_v1, data->test_vec_ops_n)};
    OSQPVectorf_ptr sv2{OSQPVectorf_new(data->test_vec_ops_shift_v2, data->test_vec_ops_n)};

    mu_assert("Incorrect comparison",
            OSQPVectorf_all_leq(sv1.get(), sv2.get()) == 0);

    mu_assert("Incorrect comparison",
            OSQPVectorf_all_leq(sv2.get(), sv1.get()) == 1);
  }
}
