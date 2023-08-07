#include "test_lin_alg.h"
#include "lin_alg_data.h"

#include "cuda_memory.h"
#include "cuda_lin_alg.h"
#include "algebra_types.h"

TEST_CASE("Vector: Float creation from device vector", "[vector],[creation]")
{
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  OSQPFloat* dev_v2;

  cuda_malloc((void**) &dev_v2, data->test_vec_ops_n * sizeof(OSQPFloat));

  cuda_vec_copy_h2d(dev_v2, data->test_vec_ops_v2, data->test_vec_ops_n);

  mu_assert("Vector not on device",
            cuda_isdeviceptr(dev_v2));

  SECTION("Assignment")
  {
    OSQPVectorf_ptr v2{OSQPVectorf_malloc(data->test_vec_ops_n)};

    OSQPVectorf_from_raw(v2.get(), dev_v2);

    mu_assert("Vector not allocated",
              v2.get());

    mu_assert("Vector not correct length",
              OSQPVectorf_length(v2.get()) == data->test_vec_ops_n);

    OSQPInt res;
    cuda_vec_eq(dev_v2, v2->d_val, TESTS_TOL, data->test_vec_ops_n, &res);

    mu_assert("Vector not assigned properly",
              res == 1);
  }

  SECTION("New from data")
  {
    OSQPVectorf_ptr v2{OSQPVectorf_new(dev_v2, data->test_vec_ops_n)};

    mu_assert("Vector not allocated",
              v2.get());

    mu_assert("Vector not correct length",
              OSQPVectorf_length(v2.get()) == data->test_vec_ops_n);

    OSQPInt res;
    cuda_vec_eq(dev_v2, v2->d_val, TESTS_TOL, data->test_vec_ops_n, &res);

    mu_assert("Vector not assigned properly",
              res == 1);
  }

  cuda_free((void**) &dev_v2);
}

TEST_CASE("Vector: Int creation from device vector", "[vector],[creation]")
{
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  OSQPInt* dev_v2;

  cuda_malloc((void**) &dev_v2, data->test_vec_ops_n * sizeof(OSQPFloat));

  cuda_vec_int_copy_h2d(dev_v2, data->test_vec_subvec_ind5, data->test_vec_ops_n);

  mu_assert("Vector not on device",
            cuda_isdeviceptr(dev_v2));

  SECTION("Assignment")
  {
    OSQPVectori_ptr v2{OSQPVectori_malloc(data->test_vec_ops_n)};

    OSQPVectori_from_raw(v2.get(), dev_v2);

    mu_assert("Vector not allocated",
              v2.get());

    mu_assert("Vector not correct length",
              OSQPVectori_length(v2.get()) == data->test_vec_ops_n);

    OSQPInt res;
    cuda_vec_int_eq(dev_v2, v2->d_val, data->test_vec_ops_n, &res);

    mu_assert("Vector not assigned properly",
              res == 1);
  }

  SECTION("New from data")
  {
    OSQPVectori_ptr v2{OSQPVectori_new(dev_v2, data->test_vec_ops_n)};

    mu_assert("Vector not allocated",
              v2.get());

    mu_assert("Vector not correct length",
              OSQPVectori_length(v2.get()) == data->test_vec_ops_n);

    OSQPInt res;
    cuda_vec_int_eq(dev_v2, v2->d_val, data->test_vec_ops_n, &res);

    mu_assert("Vector not assigned properly",
              res == 1);
  }

  cuda_free((void**) &dev_v2);
}
