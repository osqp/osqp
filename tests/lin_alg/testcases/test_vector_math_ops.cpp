#include "test_lin_alg.h"
#include "lin_alg_data.h"

TEST_CASE("Vector: Addition/Subtraction", "[vector],[operation]")
{
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  // Create all the vectors we will use
  OSQPVectorf_ptr v1{OSQPVectorf_malloc(data->test_vec_ops_n)};
  OSQPVectorf_ptr v2{OSQPVectorf_malloc(data->test_vec_ops_n)};
  OSQPVectorf_ptr ref{OSQPVectorf_malloc(data->test_vec_ops_n)};
  OSQPVectorf_ptr result{OSQPVectorf_malloc(data->test_vec_ops_n)};

  // Get the data vectors
  OSQPVectorf_from_raw(v1.get(), data->test_vec_ops_v1);
  OSQPVectorf_from_raw(v2.get(), data->test_vec_ops_v2);

  SECTION("Addition: Output vector")
  {
    OSQPVectorf_from_raw(ref.get(), data->test_vec_ops_add);

    OSQPVectorf_plus(result.get(), v1.get(), v2.get());

    // Vector should match the reference
    mu_assert("Error adding vector",
              OSQPVectorf_norm_inf_diff(ref.get(), result.get()) < TESTS_TOL);
  }

  SECTION("Addition: Accumulating to vector")
  {
    OSQPVectorf_from_raw(ref.get(), data->test_vec_ops_add);

    OSQPVectorf_plus(v1.get(), v1.get(), v2.get());

    // Vector should match the reference
    mu_assert("Error adding vector",
              OSQPVectorf_norm_inf_diff(ref.get(), v1.get()) < TESTS_TOL);
  }

  SECTION("Subtraction: Output vector")
  {
    OSQPVectorf_from_raw(ref.get(), data->test_vec_ops_sub);

    OSQPVectorf_minus(result.get(), v1.get(), v2.get());

    // Vector should match the reference
    mu_assert("Error subtracting vector",
              OSQPVectorf_norm_inf_diff(ref.get(), result.get()) < TESTS_TOL);
  }

  SECTION("Subtraction: Accumulating to vector")
  {
    OSQPVectorf_from_raw(ref.get(), data->test_vec_ops_sub);

    OSQPVectorf_minus(v1.get(), v1.get(), v2.get());

    // Vector should match the reference
    mu_assert("Error subtracting vector",
              OSQPVectorf_norm_inf_diff(ref.get(), v1.get()) < TESTS_TOL);
  }
}

TEST_CASE("Vector: Scaled Addition", "[vector],[operation]")
{
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  // Create all the vectors we will use
  OSQPVectorf_ptr v1{OSQPVectorf_new(data->test_vec_ops_v1, data->test_vec_ops_n)};
  OSQPVectorf_ptr v2{OSQPVectorf_new(data->test_vec_ops_v2, data->test_vec_ops_n)};
  OSQPVectorf_ptr v3{OSQPVectorf_new(data->test_vec_ops_v3, data->test_vec_ops_n)};
  OSQPVectorf_ptr ref{OSQPVectorf_malloc(data->test_vec_ops_n)};
  OSQPVectorf_ptr result{OSQPVectorf_malloc(data->test_vec_ops_n)};

  SECTION("Scaled addition: Result vector")
  {
    // Setup the reference
    OSQPVectorf_from_raw(ref.get(), data->test_vec_ops_add_scaled);

    OSQPVectorf_add_scaled(result.get(), data->test_vec_ops_sc1, v1.get(), data->test_vec_ops_sc2, v2.get());

    // Vector should match the reference
    mu_assert("Error adding scaled vector",
              OSQPVectorf_norm_inf_diff(ref.get(), result.get()) < TESTS_TOL);
  }

  SECTION("Scaled addition: Accumulate vector")
  {
    // Setup the reference
    OSQPVectorf_from_raw(ref.get(), data->test_vec_ops_add_scaled_inc);

    OSQPVectorf_add_scaled(v1.get(), 1.0, v1.get(), data->test_vec_ops_sc2, v2.get());

    // Vector should match the reference
    mu_assert("Error adding scaled vector",
              OSQPVectorf_norm_inf_diff(ref.get(), v1.get()) < TESTS_TOL);
  }

  SECTION("Scaled addition: Scaled accumulate vector")
  {
    // Setup the reference
    OSQPVectorf_from_raw(ref.get(), data->test_vec_ops_add_scaled);

    OSQPVectorf_add_scaled(v1.get(), data->test_vec_ops_sc1, v1.get(), data->test_vec_ops_sc2, v2.get());

    // Vector should match the reference
    mu_assert("Error adding scaled vector",
              OSQPVectorf_norm_inf_diff(ref.get(), v1.get()) < TESTS_TOL);
  }

  SECTION("3-term scaled addition: Result vector")
  {
    // Setup the reference
    OSQPVectorf_from_raw(ref.get(), data->test_vec_ops_add_scaled3);

    OSQPVectorf_add_scaled3(result.get(), data->test_vec_ops_sc1, v1.get(),
                                          data->test_vec_ops_sc2, v2.get(),
                                          data->test_vec_ops_sc3, v3.get());

    // Vector should match the reference
    mu_assert("Error adding 3-term scaled vector",
              OSQPVectorf_norm_inf_diff(ref.get(), result.get()) < TESTS_TOL);
  }

  SECTION("3-term scaled addition: Accumulate vector")
  {
    // Setup the reference
    OSQPVectorf_from_raw(ref.get(), data->test_vec_ops_add_scaled3_inc);

    OSQPVectorf_add_scaled3(v1.get(), 1.0, v1.get(),
                                      data->test_vec_ops_sc2, v2.get(),
                                      data->test_vec_ops_sc3, v3.get());

    // Vector should match the reference
    mu_assert("Error adding 3-term scaled vector",
              OSQPVectorf_norm_inf_diff(ref.get(), v1.get()) < TESTS_TOL);
  }

  SECTION("3-term scaled addition: Scaled accumulate vector")
  {
    // Setup the reference
    OSQPVectorf_from_raw(ref.get(), data->test_vec_ops_add_scaled3);

    OSQPVectorf_add_scaled3(v1.get(), data->test_vec_ops_sc1, v1.get(),
                                      data->test_vec_ops_sc2, v2.get(),
                                      data->test_vec_ops_sc3, v3.get());

    // Vector should match the reference
    mu_assert("Error adding 3-term scaled vector",
              OSQPVectorf_norm_inf_diff(ref.get(), v1.get()) < TESTS_TOL);
  }
}

TEST_CASE("Vector: Elementwise multiplication", "[vector],[operation]")
{
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  // Create all the vectors we will use
  OSQPVectorf_ptr v1{OSQPVectorf_new(data->test_vec_ops_v1, data->test_vec_ops_n)};
  OSQPVectorf_ptr v2{OSQPVectorf_new(data->test_vec_ops_v2, data->test_vec_ops_n)};
  OSQPVectorf_ptr v3{OSQPVectorf_new(data->test_vec_ops_v3, data->test_vec_ops_n)};
  OSQPVectorf_ptr result{OSQPVectorf_malloc(data->test_vec_ops_n)};

  SECTION("Scalar: Multiply by 1")
  {
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_vec_ops_v1, data->test_vec_ops_n)};

    OSQPVectorf_mult_scalar(v1.get(), 1.0);

    mu_assert("Error multiplying vector by scalar",
              OSQPVectorf_is_eq(ref.get(), v1.get(), TESTS_TOL));
  }

  SECTION("Scalar: Multiply by scalar")
  {
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_vec_ops_sca_prod, data->test_vec_ops_n)};

    OSQPVectorf_mult_scalar(v1.get(), data->test_vec_ops_sc1);

    mu_assert("Error multiplying vector by scalar",
              OSQPVectorf_is_eq(ref.get(), v1.get(), TESTS_TOL));
  }

  SECTION("Elementwise multiply with vector")
  {
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_vec_ops_ew_prod, data->test_vec_ops_n)};

    OSQPVectorf_ew_prod(result.get(), v1.get(), v2.get());

    mu_assert("Error multiplying vector elementwise",
              OSQPVectorf_is_eq(ref.get(), result.get(), TESTS_TOL));
  }

  SECTION("Elementwise multiply with vector in place")
  {
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_vec_ops_ew_prod, data->test_vec_ops_n)};

    OSQPVectorf_ew_prod(v1.get(), v1.get(), v2.get());

    mu_assert("Error multiplying vector elementwise",
              OSQPVectorf_is_eq(ref.get(), v1.get(), TESTS_TOL));
  }
}

TEST_CASE("Vector: Elementwise squareroot", "[vector],[operation]")
{
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  SECTION("General vector")
  {
    OSQPVectorf_ptr v1{OSQPVectorf_new(data->test_vec_ops_shift_v1, data->test_vec_ops_n)};
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_vec_ops_ew_sqrt, data->test_vec_ops_n)};

    OSQPVectorf_ew_sqrt(v1.get());

    mu_assert("Squareroot not working",
              OSQPVectorf_is_eq(v1.get(), ref.get(), TESTS_TOL));
  }

  SECTION("Squareroot of 1")
  {
    OSQPVectorf_ptr ones{OSQPVectorf_new(data->test_vec_ops_ones, data->test_vec_ops_n)};
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_vec_ops_ones, data->test_vec_ops_n)};

    OSQPVectorf_ew_sqrt(ones.get());

    mu_assert("Squareroot not preserving 1",
              OSQPVectorf_is_eq(ones.get(), ref.get(), TESTS_TOL));
  }
}

TEST_CASE("Vector: Elementwise reciprocal", "[vector],[operation]")
{
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  SECTION("General vector")
  {
    OSQPVectorf_ptr v1{OSQPVectorf_new(data->test_vec_ops_v1, data->test_vec_ops_n)};
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_vec_ops_ew_reciprocal, data->test_vec_ops_n)};
    OSQPVectorf_ptr res{OSQPVectorf_malloc(data->test_vec_ops_n)};

    OSQPVectorf_ew_reciprocal(res.get(), v1.get());

    mu_assert("Reciprocal not working",
              OSQPVectorf_is_eq(res.get(), ref.get(), TESTS_TOL));
  }

  SECTION("Reciprocal of 1")
  {
    OSQPVectorf_ptr ones{OSQPVectorf_new(data->test_vec_ops_ones, data->test_vec_ops_n)};
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_vec_ops_ones, data->test_vec_ops_n)};
    OSQPVectorf_ptr res{OSQPVectorf_malloc(data->test_vec_ops_n)};

    OSQPVectorf_ew_reciprocal(res.get(), ones.get());

    mu_assert("Reciprocal not preserving 1",
              OSQPVectorf_is_eq(res.get(), ref.get(), TESTS_TOL));
  }
}

TEST_CASE("Vector: Elementwise maximum", "[vector],[operation]")
{
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  SECTION("One vector is larger")
  {
    // All elements of sv1 are larger than those of sv2
    OSQPVectorf_ptr sv1{OSQPVectorf_new(data->test_vec_ops_shift_v1, data->test_vec_ops_n)};
    OSQPVectorf_ptr sv2{OSQPVectorf_new(data->test_vec_ops_shift_v2, data->test_vec_ops_n)};

    OSQPVectorf_ptr res{OSQPVectorf_malloc(data->test_vec_ops_n)};

    OSQPVectorf_ew_max_vec(res.get(), sv1.get(), sv2.get());

    mu_assert("Maximum not computed properly",
              OSQPVectorf_is_eq(res.get(), sv1.get(), TESTS_TOL));
  }

  SECTION("General vectors")
  {
    OSQPVectorf_ptr v1{OSQPVectorf_new(data->test_vec_ops_v1, data->test_vec_ops_n)};
    OSQPVectorf_ptr v2{OSQPVectorf_new(data->test_vec_ops_v2, data->test_vec_ops_n)};
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_vec_ops_ew_max_vec, data->test_vec_ops_n)};

    OSQPVectorf_ptr res{OSQPVectorf_malloc(data->test_vec_ops_n)};

    OSQPVectorf_ew_max_vec(res.get(), v1.get(), v2.get());

    mu_assert("Maximum not computed properly",
              OSQPVectorf_is_eq(res.get(), ref.get(), TESTS_TOL));
  }

  SECTION("Max of all 1")
  {

    OSQPVectorf_ptr ones{OSQPVectorf_new(data->test_vec_ops_ones, data->test_vec_ops_n)};
    OSQPVectorf_ptr res{OSQPVectorf_malloc(data->test_vec_ops_n)};

    OSQPVectorf_ew_max_vec(res.get(), ones.get(), ones.get());

    mu_assert("Max not computed properly",
              OSQPVectorf_is_eq(ones.get(), res.get(), TESTS_TOL));
  }
}

TEST_CASE("Vector: Elementwise minimum", "[vector],[operation]")
{
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  SECTION("One vector is larger")
  {
    // All elements of sv1 are larger than those of sv2
    OSQPVectorf_ptr sv1{OSQPVectorf_new(data->test_vec_ops_shift_v1, data->test_vec_ops_n)};
    OSQPVectorf_ptr sv2{OSQPVectorf_new(data->test_vec_ops_shift_v2, data->test_vec_ops_n)};

    OSQPVectorf_ptr res{OSQPVectorf_malloc(data->test_vec_ops_n)};

    OSQPVectorf_ew_min_vec(res.get(), sv1.get(), sv2.get());

    mu_assert("Minimum not computed properly",
              OSQPVectorf_is_eq(res.get(), sv2.get(), TESTS_TOL));
  }

  SECTION("General vectors")
  {
    OSQPVectorf_ptr v1{OSQPVectorf_new(data->test_vec_ops_v1, data->test_vec_ops_n)};
    OSQPVectorf_ptr v2{OSQPVectorf_new(data->test_vec_ops_v2, data->test_vec_ops_n)};
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_vec_ops_ew_min_vec, data->test_vec_ops_n)};

    OSQPVectorf_ptr res{OSQPVectorf_malloc(data->test_vec_ops_n)};

    OSQPVectorf_ew_min_vec(res.get(), v1.get(), v2.get());

    mu_assert("Minimum not computed properly",
              OSQPVectorf_is_eq(res.get(), ref.get(), TESTS_TOL));
  }

  SECTION("Min of all 1")
  {

    OSQPVectorf_ptr ones{OSQPVectorf_new(data->test_vec_ops_ones, data->test_vec_ops_n)};
    OSQPVectorf_ptr res{OSQPVectorf_malloc(data->test_vec_ops_n)};

    OSQPVectorf_ew_min_vec(res.get(), ones.get(), ones.get());

    mu_assert("Minimum not computed properly",
              OSQPVectorf_is_eq(ones.get(), res.get(), TESTS_TOL));
  }
}

TEST_CASE("Vector: Elementwise bound vector", "[vector],[operation]")
{
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  SECTION("No change")
  {
    OSQPVectorf_ptr v1{OSQPVectorf_new(data->test_vec_ops_v1, data->test_vec_ops_n)};
    OSQPVectorf_ptr lb{OSQPVectorf_new(data->test_vec_ops_vn_neg, data->test_vec_ops_n)};
    OSQPVectorf_ptr ub{OSQPVectorf_new(data->test_vec_ops_vn, data->test_vec_ops_n)};

    OSQPVectorf_ptr res{OSQPVectorf_malloc(data->test_vec_ops_n)};

    OSQPVectorf_ew_bound_vec(res.get(), v1.get(), lb.get(), ub.get());

    mu_assert("Bounds not computed properly",
              OSQPVectorf_is_eq(res.get(), v1.get(), TESTS_TOL));
  }

  SECTION("All above upper bound")
  {
    OSQPVectorf_ptr v1{OSQPVectorf_new(data->test_vec_ops_vn, data->test_vec_ops_n)};
    OSQPVectorf_ptr lb{OSQPVectorf_new(data->test_vec_ops_vn_neg, data->test_vec_ops_n)};
    OSQPVectorf_ptr ub{OSQPVectorf_new(data->test_vec_ops_zero, data->test_vec_ops_n)};

    OSQPVectorf_ptr res{OSQPVectorf_malloc(data->test_vec_ops_n)};

    OSQPVectorf_ew_bound_vec(res.get(), v1.get(), lb.get(), ub.get());

    mu_assert("Bounds not computed properly",
              OSQPVectorf_is_eq(res.get(), ub.get(), TESTS_TOL));
  }

  SECTION("All below lower bound")
  {
    OSQPVectorf_ptr v1{OSQPVectorf_new(data->test_vec_ops_vn_neg, data->test_vec_ops_n)};
    OSQPVectorf_ptr lb{OSQPVectorf_new(data->test_vec_ops_zero, data->test_vec_ops_n)};
    OSQPVectorf_ptr ub{OSQPVectorf_new(data->test_vec_ops_vn, data->test_vec_ops_n)};

    OSQPVectorf_ptr res{OSQPVectorf_malloc(data->test_vec_ops_n)};

    OSQPVectorf_ew_bound_vec(res.get(), v1.get(), lb.get(), ub.get());

    mu_assert("Bounds not computed properly",
              OSQPVectorf_is_eq(res.get(), lb.get(), TESTS_TOL));
  }

  SECTION("General vector")
  {
    OSQPVectorf_ptr v1{OSQPVectorf_new(data->test_vec_ops_v1, data->test_vec_ops_n)};
    OSQPVectorf_ptr lb{OSQPVectorf_new(data->test_vec_ops_v3, data->test_vec_ops_n)};
    OSQPVectorf_ptr ub{OSQPVectorf_new(data->test_vec_ops_v2, data->test_vec_ops_n)};
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_vec_ops_ew_bound_vec, data->test_vec_ops_n)};

    OSQPVectorf_ptr res{OSQPVectorf_malloc(data->test_vec_ops_n)};

    OSQPVectorf_ew_bound_vec(res.get(), v1.get(), lb.get(), ub.get());

    mu_assert("Bounds not computed properly",
              OSQPVectorf_is_eq(res.get(), ref.get(), TESTS_TOL));
  }

  SECTION("Both bounds the same")
  {
    OSQPVectorf_ptr v1{OSQPVectorf_new(data->test_vec_ops_v1, data->test_vec_ops_n)};
    OSQPVectorf_ptr lb{OSQPVectorf_new(data->test_vec_ops_zero, data->test_vec_ops_n)};
    OSQPVectorf_ptr ub{OSQPVectorf_new(data->test_vec_ops_zero, data->test_vec_ops_n)};

    OSQPVectorf_ptr res{OSQPVectorf_malloc(data->test_vec_ops_n)};

    OSQPVectorf_ew_bound_vec(res.get(), v1.get(), lb.get(), ub.get());

    mu_assert("Bounds not computed properly",
              OSQPVectorf_is_eq(res.get(), lb.get(), TESTS_TOL));

        mu_assert("Bounds not computed properly",
              OSQPVectorf_is_eq(res.get(), ub.get(), TESTS_TOL));
  }
}

TEST_CASE("Vector: Norms")
{
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  // Create all the vectors we will use
  OSQPVectorf_ptr vn{OSQPVectorf_new(data->test_vec_ops_vn, data->test_vec_ops_n)};
  OSQPVectorf_ptr v1{OSQPVectorf_new(data->test_vec_ops_v1, data->test_vec_ops_n)};
  OSQPVectorf_ptr v2{OSQPVectorf_new(data->test_vec_ops_v2, data->test_vec_ops_n)};
  OSQPVectorf_ptr v3{OSQPVectorf_new(data->test_vec_ops_v3, data->test_vec_ops_n)};
  OSQPVectorf_ptr pv1{OSQPVectorf_new(data->test_vec_ops_pos_v1, data->test_vec_ops_n)};
  OSQPVectorf_ptr nv1{OSQPVectorf_new(data->test_vec_ops_neg_v1, data->test_vec_ops_n)};
  OSQPVectorf_ptr nv2{OSQPVectorf_new(data->test_vec_ops_neg_v2, data->test_vec_ops_n)};
  OSQPVectorf_ptr nv3{OSQPVectorf_new(data->test_vec_ops_neg_v3, data->test_vec_ops_n)};
  OSQPVectorf_ptr result{OSQPVectorf_malloc(data->test_vec_ops_n)};

  SECTION("1-norm: Easy computation")
  {
    OSQPFloat asum = OSQPVectorf_norm_1(vn.get());
    OSQPFloat ref  = data->test_vec_ops_n * data->test_vec_ops_n;

    mu_assert("1-norm computation failed",
              c_absval(asum - ref) < TESTS_TOL);
  }

  SECTION("1-norm: Full negative vector")
  {
    OSQPFloat asum = OSQPVectorf_norm_1(nv1.get());
    OSQPFloat ref  = data->test_vec_ops_neg_norm_1;

    mu_assert("1-norm computation failed",
              c_absval(asum - ref) < TESTS_TOL);
  }

  SECTION("1-norm: Full positive vector")
  {
    OSQPFloat asum = OSQPVectorf_norm_1(pv1.get());
    OSQPFloat ref  = data->test_vec_ops_pos_norm_1;

    mu_assert("1-norm computation failed",
              c_absval(asum - ref) < TESTS_TOL);
  }

  SECTION("1-norm: Single element")
  {
    OSQPVectorf_ptr sv1{OSQPVectorf_new(data->test_vec_ops_pos_v1, 1)};

    OSQPFloat asum = OSQPVectorf_norm_1(sv1.get());
    OSQPFloat ref  = data->test_vec_ops_pos_v1[0];

    mu_assert("1-norm computation failed",
              c_absval(asum - ref) < TESTS_TOL);
  }

  SECTION("1-norm: No data")
  {
    OSQPVectorf_ptr ev1{OSQPVectorf_new(data->test_vec_ops_pos_v1, 0)};

    OSQPFloat asum = OSQPVectorf_norm_1(ev1.get());
    OSQPFloat ref  = 0.0;

    mu_assert("1-norm computation failed",
              c_absval(asum - ref) < TESTS_TOL );
  }

#ifdef OSQP_ALGEBRA_BUILTIN
  SECTION("2-norm")
  {
    OSQPFloat res = OSQPVectorf_norm_2(v1.get());

    mu_assert("Error in computation",
              c_absval(res - data->test_vec_ops_norm_2) < TESTS_TOL);
  }
#endif

  SECTION("Infinity norm")
  {
    OSQPFloat res = OSQPVectorf_norm_inf(v1.get());

    mu_assert("Error in computation",
              c_absval(res - data->test_vec_ops_norm_inf) < TESTS_TOL);
  }

  SECTION("Infinity norm of negated vector")
  {
    OSQPFloat res = OSQPVectorf_norm_inf(nv1.get());

    mu_assert("Error in computation",
              c_absval(res - data->test_vec_ops_norm_inf) < TESTS_TOL);
  }

  SECTION("Scaled infinity norm")
  {
    OSQPFloat res = OSQPVectorf_scaled_norm_inf(v1.get(), v2.get());

    mu_assert("Error in computation",
              c_absval(res - data->test_vec_ops_norm_inf_scaled) < TESTS_TOL);
  }

  SECTION("Scaled infinity norm of negated vector")
  {
    OSQPFloat res = OSQPVectorf_scaled_norm_inf(v1.get(), nv2.get());

    mu_assert("Error in computation",
              c_absval(res - data->test_vec_ops_norm_inf_scaled) < TESTS_TOL);
  }

  SECTION("Difference infinity norm")
  {
    OSQPFloat res = OSQPVectorf_norm_inf_diff(v1.get(), v2.get());

    mu_assert("Error in computation",
              c_absval(res - data->test_vec_ops_norm_inf_diff) < TESTS_TOL);
  }

  SECTION("Difference infinity norm of negated vector")
  {
    OSQPFloat res = OSQPVectorf_norm_inf_diff(nv1.get(), nv2.get());

    mu_assert("Error in computation",
              c_absval(res - data->test_vec_ops_norm_inf_diff) < TESTS_TOL);
  }
}

TEST_CASE("Vector: Dot product", "[vector],[operation]")
{
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  // Create all the vectors we will use
  OSQPVectorf_ptr v1{OSQPVectorf_new(data->test_vec_ops_v1, data->test_vec_ops_n)};
  OSQPVectorf_ptr v2{OSQPVectorf_new(data->test_vec_ops_v2, data->test_vec_ops_n)};
  OSQPVectorf_ptr v3{OSQPVectorf_new(data->test_vec_ops_v3, data->test_vec_ops_n)};
  OSQPVectorf_ptr nv1{OSQPVectorf_new(data->test_vec_ops_neg_v1, data->test_vec_ops_n)};
  OSQPVectorf_ptr nv2{OSQPVectorf_new(data->test_vec_ops_neg_v2, data->test_vec_ops_n)};
  OSQPVectorf_ptr nv3{OSQPVectorf_new(data->test_vec_ops_neg_v3, data->test_vec_ops_n)};

  SECTION("Same vector")
  {
    OSQPFloat dot = OSQPVectorf_dot_prod(v1.get(), v1.get());

    mu_assert("Incorrect dot product",
              c_absval(dot - data->test_vec_ops_vec_dot_v1) < TESTS_TOL);
  }

  SECTION("2-Norm comparison")
  {
    OSQPFloat dot = OSQPVectorf_dot_prod(v1.get(), v1.get());

#ifdef OSQP_ALGEBRA_BUILTIN
    OSQPFloat nrm = OSQPVectorf_norm_2(v1.get());
    nrm = nrm * nrm;

    mu_assert("Dot product and norm don't match",
              c_absval(dot - nrm) < TESTS_TOL);
#endif

    OSQPFloat pynorm = data->test_vec_ops_norm_2 * data->test_vec_ops_norm_2;

    mu_assert("Dot product and Python norm don't match",
              c_absval(dot - pynorm) < TESTS_TOL);
  }

  SECTION("Different vectors")
  {
    OSQPFloat dot = OSQPVectorf_dot_prod(v1.get(), v2.get());

    mu_assert("Incorrect dot product",
              c_absval(dot - data->test_vec_ops_vec_dot) < TESTS_TOL);
  }
}

TEST_CASE("Vector: Scaled dot product", "[vector],[operation]")
{
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  // Create all the vectors we will use
  OSQPVectorf_ptr v1{OSQPVectorf_new(data->test_vec_ops_v1, data->test_vec_ops_n)};
  OSQPVectorf_ptr v2{OSQPVectorf_new(data->test_vec_ops_v2, data->test_vec_ops_n)};
  OSQPVectorf_ptr v3{OSQPVectorf_new(data->test_vec_ops_v1, data->test_vec_ops_n)};

  // Use only positive elements of second argument
  SECTION("Positive elements")
  {
    SECTION("Same vector")
    {
      OSQPFloat dot = OSQPVectorf_dot_prod_signed(v1.get(), v3.get(), 1);

      mu_assert("Incorrect signed dot product",
                c_absval(dot - data->test_vec_ops_vec_dot_pos_v1) < TESTS_TOL);

      // v3 has the same data as v1, so should have same signed dot product here
      dot = OSQPVectorf_dot_prod_signed(v3.get(), v1.get(), 1);

      mu_assert("Incorrect signed dot product",
                c_absval(dot - data->test_vec_ops_vec_dot_pos_v1) < TESTS_TOL);
    }

    SECTION("Different vectors")
    {
      OSQPFloat dot = OSQPVectorf_dot_prod_signed(v1.get(), v2.get(), 1);

      mu_assert("Incorrect signed dot product",
                c_absval(dot - data->test_vec_ops_vec_dot_pos) < TESTS_TOL);

      // Uses only positive from v1 now instead of v2, so dot product is different
      dot = OSQPVectorf_dot_prod_signed(v2.get(), v1.get(), 1);

      mu_assert("Incorrect signed dot product",
                c_absval(dot - data->test_vec_ops_vec_dot_pos_flip) < TESTS_TOL);
    }

    SECTION("No possible elements")
    {
      // All elements of this vector are negative
      OSQPVectorf_ptr sv2{OSQPVectorf_new(data->test_vec_ops_shift_v2, data->test_vec_ops_n)};

      OSQPFloat dot = OSQPVectorf_dot_prod_signed(v1.get(), sv2.get(), 1);

      mu_assert("Incorrect signed dot product",
                c_absval(dot - 0) < TESTS_TOL);
    }
  }

  // Use only negative elements of second argument
  SECTION("Negative elements")
  {
    SECTION("Same vector")
    {
      OSQPFloat dot = OSQPVectorf_dot_prod_signed(v1.get(), v3.get(), -1);

      mu_assert("Incorrect signed dot product",
                c_absval(dot - data->test_vec_ops_vec_dot_neg_v1) < TESTS_TOL);

      // v3 has the same data as v1, so should have same signed dot product here
      dot = OSQPVectorf_dot_prod_signed(v3.get(), v1.get(), -1);

      mu_assert("Incorrect signed dot product",
                c_absval(dot - data->test_vec_ops_vec_dot_neg_v1) < TESTS_TOL);
    }

    SECTION("Different vectors")
    {
      OSQPFloat dot = OSQPVectorf_dot_prod_signed(v1.get(), v2.get(), -1);

      mu_assert("Incorrect signed dot product",
                c_absval(dot - data->test_vec_ops_vec_dot_neg) < TESTS_TOL);

      // Uses only positive from v1 now instead of v2, so dot product is different
      dot = OSQPVectorf_dot_prod_signed(v2.get(), v1.get(), -1);

      mu_assert("Incorrect signed dot product",
                c_absval(dot - data->test_vec_ops_vec_dot_neg_flip) < TESTS_TOL);
    }

    SECTION("No possible elements")
    {
      // All elements of this vector are positive
      OSQPVectorf_ptr sv1{OSQPVectorf_new(data->test_vec_ops_shift_v1, data->test_vec_ops_n)};

      OSQPFloat dot = OSQPVectorf_dot_prod_signed(v1.get(), sv1.get(), -1);

      mu_assert("Incorrect signed dot product",
                c_absval(dot - 0) < TESTS_TOL);
    }
  }

  // Falls back to the normal dot product for non -1/1 signs
  SECTION("Fallback with all elements")
  {
    SECTION("Same vector")
    {
      OSQPFloat dot = OSQPVectorf_dot_prod_signed(v1.get(), v1.get(), -4);

      mu_assert("Incorrect signed dot product",
                c_absval(dot - data->test_vec_ops_vec_dot_v1) < TESTS_TOL);
    }

    SECTION("2-Norm comparison")
    {
      OSQPFloat dot = OSQPVectorf_dot_prod_signed(v1.get(), v1.get(), 6);

#ifdef OSQP_ALGEBRA_BUILTIN
      OSQPFloat nrm = OSQPVectorf_norm_2(v1.get());
      nrm = nrm * nrm;

      mu_assert("Signed Dot product and norm don't match",
                c_absval(dot - nrm) < TESTS_TOL);
#endif

      OSQPFloat pynorm = data->test_vec_ops_norm_2 * data->test_vec_ops_norm_2;

      mu_assert("Signed dot product and Python norm don't match",
                c_absval(dot - pynorm) < TESTS_TOL);
    }

    SECTION("Different vectors")
    {
      OSQPFloat dot = OSQPVectorf_dot_prod_signed(v1.get(), v2.get(), 0);

      mu_assert("Incorrect signed dot product",
                c_absval(dot - data->test_vec_ops_vec_dot) < TESTS_TOL);
    }
  }
}
