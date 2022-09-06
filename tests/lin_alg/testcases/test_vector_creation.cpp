#include "test_lin_alg.h"
#include "lin_alg_data.h"

TEST_CASE("Vector: Creation", "[vector],[creation]")
{
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  SECTION("Malloc")
  {
    OSQPVectorf_ptr v1{OSQPVectorf_malloc(0)};
    OSQPVectorf_ptr v2{OSQPVectorf_malloc(data->test_vec_ops_n)};

    mu_assert("Vector not allocated",
              v1.get());

    mu_assert("Vector not allocated",
              v2.get());

    mu_assert("Vector not correct length",
              OSQPVectorf_length(v1.get()) == 0);

    mu_assert("Vector not correct length",
              OSQPVectorf_length(v2.get()) == data->test_vec_ops_n);
  }

  SECTION("Calloc")
  {
    OSQPVectorf_ptr v1{OSQPVectorf_calloc(0)};
    OSQPVectorf_ptr v2{OSQPVectorf_calloc(data->test_vec_ops_n)};
    OSQPVectorf_ptr z{OSQPVectorf_new(data->test_vec_ops_zero, data->test_vec_ops_n)};

    mu_assert("Vector not allocated",
              v1.get());

    mu_assert("Vector not allocated",
              v2.get());

    mu_assert("Vector not correct length",
              OSQPVectorf_length(v1.get()) == 0);

    mu_assert("Vector not correct length",
              OSQPVectorf_length(v2.get()) == data->test_vec_ops_n);

    mu_assert("Vector not zero-initialized",
              OSQPVectorf_is_eq(v2.get(), z.get(), TESTS_TOL));
  }

  SECTION("Assignment")
  {
    OSQPVectorf_ptr v1{OSQPVectorf_malloc(0)};
    OSQPVectorf_ptr v2{OSQPVectorf_malloc(data->test_vec_ops_n)};

    OSQPVectorf_from_raw(v1.get(), data->test_vec_ops_v1);
    OSQPVectorf_from_raw(v2.get(), data->test_vec_ops_v2);

    mu_assert("Vector not allocated",
              v1.get());

    mu_assert("Vector not allocated",
              v2.get());

    mu_assert("Vector not correct length",
              OSQPVectorf_length(v1.get()) == 0);

    mu_assert("Vector not correct length",
              OSQPVectorf_length(v2.get()) == data->test_vec_ops_n);
  }

  SECTION("New from data")
  {
    OSQPVectorf_ptr v1{OSQPVectorf_new(OSQP_NULL, 0)};
    OSQPVectorf_ptr v2{OSQPVectorf_new(data->test_vec_ops_v1, data->test_vec_ops_n)};

    mu_assert("Vector not allocated",
              v1.get());

    mu_assert("Vector not allocated",
              v2.get());

    mu_assert("Vector not correct length",
              OSQPVectorf_length(v1.get()) == 0);

    mu_assert("Vector not correct length",
              OSQPVectorf_length(v2.get()) == data->test_vec_ops_n);
  }

  SECTION("Copy from existing")
  {
    OSQPVectorf_ptr v1{OSQPVectorf_new(data->test_vec_ops_v1, data->test_vec_ops_n)};

    OSQPVectorf_ptr res{OSQPVectorf_copy_new(v1.get())};

    mu_assert("Vector not correct length",
              OSQPVectorf_length(res.get()) == OSQPVectorf_length(v1.get()));

    mu_assert("Vector data not correct",
              OSQPVectorf_is_eq(v1.get(), res.get(), TESTS_TOL));
  }
}

/* This is used by the derivatives right now */
#ifdef OSQP_ALGEBRA_BUILTIN
TEST_CASE("Vector: Subvector", "[vector],[creation]")
{
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  OSQPVectorf_ptr v{OSQPVectorf_new(data->test_vec_ops_v1, data->test_vec_ops_n)};

  SECTION("0 rows")
  {
    OSQPVectori_ptr ind{OSQPVectori_new(data->test_vec_subvec_ind0, data->test_vec_ops_n)};
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_vec_subvec_0, 0)};

    OSQPVectorf_ptr res{OSQPVectorf_subvector_byrows(v.get(), ind.get())};

    mu_assert("Subvector not correct length",
              OSQPVectorf_length(res.get()) == OSQPVectorf_length(ref.get()));
  }


  SECTION("5 rows")
  {
    OSQPVectori_ptr ind{OSQPVectori_new(data->test_vec_subvec_ind5, data->test_vec_ops_n)};
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_vec_subvec_5, 5)};

    OSQPVectorf_ptr res{OSQPVectorf_subvector_byrows(v.get(), ind.get())};

    mu_assert("Subvector not correct length",
              OSQPVectorf_length(res.get()) == OSQPVectorf_length(ref.get()));

    mu_assert("Subvector data incorrect",
              OSQPVectorf_is_eq(res.get(), ref.get(), TESTS_TOL));
  }


  SECTION("All rows")
  {
    OSQPVectori_ptr ind{OSQPVectori_new(data->test_vec_subvec_ind10, data->test_vec_ops_n)};

    OSQPVectorf_ptr res{OSQPVectorf_subvector_byrows(v.get(), ind.get())};

    mu_assert("Subvector not correct length",
              OSQPVectorf_length(res.get()) == OSQPVectorf_length(v.get()));

    mu_assert("Subvector data incorrect",
              OSQPVectorf_is_eq(res.get(), v.get(), TESTS_TOL));
  }
}
#endif
