#include "test_lin_alg.h"
#include "lin_alg_data.h"

TEST_CASE("Vector: Float creation", "[vector],[creation]")
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

    std::unique_ptr<OSQPFloat[]> val = std::make_unique<OSQPFloat[]>(data->test_vec_ops_n);

    OSQPVectorf_to_raw(val.get(), v2.get());

    OSQPInt res = 1;
    for(OSQPInt i = 0; i < data->test_vec_ops_n; i++)
    {
      if(val[i] != 0.0)
        res = 0;
    }

    mu_assert("Vector not assigned properly",
              res == 1);
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

    std::unique_ptr<OSQPFloat[]> val = std::make_unique<OSQPFloat[]>(data->test_vec_ops_n);

    OSQPVectorf_to_raw(val.get(), v2.get());

    OSQPInt res = 1;
    for(OSQPInt i = 0; i < data->test_vec_ops_n; i++)
    {
      if(c_absval(val[i] - data->test_vec_ops_v2[i]) > TESTS_TOL)
        res = 0;
    }

    mu_assert("Vector not assigned properly",
              res == 1);
  }

  SECTION("New from data")
  {
    OSQPVectorf_ptr v1{OSQPVectorf_new(OSQP_NULL, 0)};
    OSQPVectorf_ptr v2{OSQPVectorf_new(data->test_vec_ops_v2, data->test_vec_ops_n)};

    mu_assert("Vector not allocated",
              v1.get());

    mu_assert("Vector not allocated",
              v2.get());

    mu_assert("Vector not correct length",
              OSQPVectorf_length(v1.get()) == 0);

    mu_assert("Vector not correct length",
              OSQPVectorf_length(v2.get()) == data->test_vec_ops_n);

    std::unique_ptr<OSQPFloat[]> val = std::make_unique<OSQPFloat[]>(data->test_vec_ops_n);

    OSQPVectorf_to_raw(val.get(), v2.get());

    OSQPInt res = 1;
    for(OSQPInt i = 0; i < data->test_vec_ops_n; i++)
    {
      if(c_absval(val[i] - data->test_vec_ops_v2[i]) > TESTS_TOL)
        res = 0;
    }

    mu_assert("Vector not assigned properly",
              res == 1);
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

TEST_CASE("Vector: Int creation", "[vector],[creation]")
{
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  SECTION("Malloc")
  {
    OSQPVectori_ptr v1{OSQPVectori_malloc(0)};
    OSQPVectori_ptr v2{OSQPVectori_malloc(data->test_vec_ops_n)};

    mu_assert("Vector not allocated",
              v1.get());

    mu_assert("Vector not allocated",
              v2.get());

    mu_assert("Vector not correct length",
              OSQPVectori_length(v1.get()) == 0);

    mu_assert("Vector not correct length",
              OSQPVectori_length(v2.get()) == data->test_vec_ops_n);
  }

  SECTION("Calloc")
  {
    OSQPVectori_ptr v1{OSQPVectori_calloc(0)};
    OSQPVectori_ptr v2{OSQPVectori_calloc(data->test_vec_ops_n)};
    OSQPVectori_ptr z{OSQPVectori_new(data->test_vec_ops_zero_int, data->test_vec_ops_n)};

    mu_assert("Vector not allocated",
              v1.get());

    mu_assert("Vector not allocated",
              v2.get());

    mu_assert("Vector not correct length",
              OSQPVectori_length(v1.get()) == 0);

    mu_assert("Vector not correct length",
              OSQPVectori_length(v2.get()) == data->test_vec_ops_n);

    std::unique_ptr<OSQPInt[]> val = std::make_unique<OSQPInt[]>(data->test_vec_ops_n);

    OSQPVectori_to_raw(val.get(), v2.get());

    OSQPInt res = 1;
    for(OSQPInt i = 0; i < data->test_vec_ops_n; i++)
    {
      if(val[i] != 0)
        res = 0;
    }

    mu_assert("Vector not zero-initialized",
              res == 1);
  }

  SECTION("Assignment")
  {
    OSQPVectori_ptr v1{OSQPVectori_malloc(0)};
    OSQPVectori_ptr v2{OSQPVectori_malloc(data->test_vec_ops_n)};

    OSQPVectori_from_raw(v1.get(), data->test_vec_ops_zero_int);
    OSQPVectori_from_raw(v2.get(), data->test_vec_subvec_ind5);

    mu_assert("Vector not allocated",
              v1.get());

    mu_assert("Vector not allocated",
              v2.get());

    mu_assert("Vector not correct length",
              OSQPVectori_length(v1.get()) == 0);

    mu_assert("Vector not correct length",
              OSQPVectori_length(v2.get()) == data->test_vec_ops_n);

    std::unique_ptr<OSQPInt[]> val = std::make_unique<OSQPInt[]>(data->test_vec_ops_n);

    OSQPVectori_to_raw(val.get(), v2.get());

    OSQPInt res = 1;
    for(OSQPInt i = 0; i < data->test_vec_ops_n; i++)
    {
      if(val[i] != data->test_vec_subvec_ind5[i])
        res = 0;
    }

    mu_assert("Vector not assigned properly",
              res == 1);
  }

  SECTION("New from data")
  {
    OSQPVectori_ptr v1{OSQPVectori_new(OSQP_NULL, 0)};
    OSQPVectori_ptr v2{OSQPVectori_new(data->test_vec_subvec_ind5, data->test_vec_ops_n)};

    mu_assert("Vector not allocated",
              v1.get());

    mu_assert("Vector not allocated",
              v2.get());

    mu_assert("Vector not correct length",
              OSQPVectori_length(v1.get()) == 0);

    mu_assert("Vector not correct length",
              OSQPVectori_length(v2.get()) == data->test_vec_ops_n);

    std::unique_ptr<OSQPInt[]> val = std::make_unique<OSQPInt[]>(data->test_vec_ops_n);

    OSQPVectori_to_raw(val.get(), v2.get());

    OSQPInt res = 1;
    for(OSQPInt i = 0; i < data->test_vec_ops_n; i++)
    {
      if(val[i] != data->test_vec_subvec_ind5[i])
        res = 0;
    }

    mu_assert("Vector not assigned properly",
              res == 1);
  }
}

TEST_CASE("Vector: Set values", "[vector],[assignment]")
{
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  OSQPVectorf_ptr v{OSQPVectorf_new(data->test_vec_ops_v1, data->test_vec_ops_n)};

  SECTION("Scalar set")
  {
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_vec_ops_same, data->test_vec_ops_n)};

    OSQPVectorf_set_scalar(v.get(), data->test_vec_ops_sc1);

    mu_assert("Error setting all values",
              OSQPVectorf_is_eq(ref.get(), v.get(), TESTS_TOL));
  }

  SECTION("Scalar set 0")
  {
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_vec_ops_zero, data->test_vec_ops_n)};

    OSQPVectorf_set_scalar(v.get(), 0.0);

    mu_assert("Error setting all values",
              OSQPVectorf_is_eq(ref.get(), v.get(), TESTS_TOL));
  }

  SECTION("Conditional scalar set")
  {
    OSQPVectori_ptr ind{OSQPVectori_new(data->test_vec_ops_sca_cond, data->test_vec_ops_n)};
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_vec_ops_sca_cond_res, data->test_vec_ops_n)};

    OSQPVectorf_set_scalar_conditional(v.get(),
                                       ind.get(),
                                       data->test_vec_ops_sc1,
                                       data->test_vec_ops_sc2,
                                       data->test_vec_ops_sc3);

    mu_assert("Error setting all values",
              OSQPVectorf_is_eq(ref.get(), v.get(), TESTS_TOL));
  }

  SECTION("Set less than")
  {
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_vec_ops_sca_lt, data->test_vec_ops_n)};
    OSQPVectorf_ptr res{OSQPVectorf_malloc(data->test_vec_ops_n)};

    OSQPVectorf_set_scalar_if_lt(res.get(), v.get(), data->test_vec_ops_sc1, data->test_vec_ops_sc2);

   mu_assert("Error setting values",
              OSQPVectorf_is_eq(ref.get(), res.get(), TESTS_TOL));
  }

  SECTION("Set greater than")
  {
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_vec_ops_sca_gt, data->test_vec_ops_n)};
    OSQPVectorf_ptr res{OSQPVectorf_malloc(data->test_vec_ops_n)};

    OSQPVectorf_set_scalar_if_gt(res.get(), v.get(), data->test_vec_ops_sc1, data->test_vec_ops_sc2);

   mu_assert("Error setting values",
              OSQPVectorf_is_eq(ref.get(), res.get(), TESTS_TOL));
  }
}

/* This is used by the derivatives right now */
#ifdef OSQP_ALGEBRA_BUILTIN
TEST_CASE("Vector: Concat", "[vector],[creation]")
{
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  OSQPVectorf_ptr v1{OSQPVectorf_new(data->test_vec_ops_v1, data->test_vec_ops_n)};
  OSQPVectorf_ptr v2{OSQPVectorf_new(data->test_vec_ops_v2, data->test_vec_ops_n)};

  OSQPVectorf_ptr res{OSQPVectorf_concat(v1.get(), v2.get())};

  std::unique_ptr<OSQPFloat[]> val = std::make_unique<OSQPFloat[]>(2*data->test_vec_ops_n);

  OSQPVectorf_to_raw(val.get(), res.get());

  OSQPInt same = 1;
  OSQPInt i = 0;
  for(i = 0; i < data->test_vec_ops_n; i++)
  {
    if(c_absval(val[i] - data->test_vec_ops_v1[i]) > TESTS_TOL)
      same = 0;
  }
  for(i = 0; i < data->test_vec_ops_n; i++)
  {
    if(c_absval(val[data->test_vec_ops_n + i] - data->test_vec_ops_v2[i]) > TESTS_TOL)
      same = 0;
  }

  mu_assert("Vector not concatenated properly",
            same == 1);
}

TEST_CASE("Vector: Subvector assignment", "[vector],[assignment]")
{
  lin_alg_sols_data_ptr data{generate_problem_lin_alg_sols_data()};

  OSQPVectorf_ptr v{OSQPVectorf_new(data->test_vec_ops_v1, data->test_vec_ops_n)};

  SECTION("No rows assigned")
  {
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_vec_ops_v1, data->test_vec_ops_n)};

    OSQPVectorf_subvector_assign(v.get(), data->test_vec_subvec_assign_5, 2, 0, 1);

    mu_assert("Subvector assignment failed",
              OSQPVectorf_is_eq(v.get(), ref.get(), TESTS_TOL));
  }

  SECTION("Partial assignment")
  {
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_vec_subvec_assign_5, data->test_vec_ops_n)};

    OSQPVectorf_subvector_assign(v.get(), data->test_vec_subvec_5, 2, 5, 1);

    mu_assert("Subvector assignment failed",
              OSQPVectorf_is_eq(v.get(), ref.get(), TESTS_TOL));
  }

  SECTION("Fully reassign")
  {
    OSQPVectorf_ptr ref{OSQPVectorf_new(data->test_vec_ops_v2, data->test_vec_ops_n)};

    OSQPVectorf_subvector_assign(v.get(), data->test_vec_ops_v2, 0, data->test_vec_ops_n, 1);

    mu_assert("Subvector assignment failed",
              OSQPVectorf_is_eq(v.get(), ref.get(), TESTS_TOL));
  }
}

TEST_CASE("Vector: Subvector creation", "[vector],[creation]")
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
