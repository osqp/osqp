#include "test_lin_alg.h"
#include "lin_alg_data.h"

TEST_CASE("Vector: Addition/Subtraction", "[vector],[operation]")
{
  OSQPFloat scresult;
  OSQPFloat scref;

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
  OSQPFloat scresult;
  OSQPFloat scref;

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
