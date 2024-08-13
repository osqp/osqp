#include <catch2/catch.hpp>

#include "osqp_api.h"    /* OSQP API wrapper (public + some private) */
#include "osqp_tester.h" /* Tester helpers */
#include "test_utils.h"  /* Testing Helper functions */

TEST_CASE("CSC zero matrix", "[csc]" )
{
  OSQPCscMatrix* test;

  SECTION("Square")
  {
    OSQPInt n = 5;
    OSQPInt m = 5;

    test = OSQPCscMatrix_zeros(m, n);

    mu_assert("CSC zero matrix: Error in matrix size", test->m == m);
    mu_assert("CSC zero matrix: Error in matrix size", test->n == n);
    mu_assert("CSC zero matrix: Error in matrix ownership", test->owned == 1);

    mu_assert("CSC zero matrix: Error in x vector", test->x == OSQP_NULL);
    mu_assert("CSC zero matrix: Error in i vector", test->i == OSQP_NULL);

    for(int i=0; i < n; i++)
    {
        mu_assert("CSC zero matrix: Error in p vector", test->p[i] == 0);
    }

    mu_assert("CSC zero matrix: Error in p vector", test->p[n] == 0);
  }

  SECTION("Fewer rows than columns")
  {
    OSQPInt n = 5;
    OSQPInt m = 3;

    test = OSQPCscMatrix_zeros(m, n);

    mu_assert("CSC zero matrix: Error in matrix size", test->m == m);
    mu_assert("CSC zero matrix: Error in matrix size", test->n == n);
    mu_assert("CSC zero matrix: Error in matrix ownership", test->owned == 1);

    mu_assert("CSC zero matrix: Error in x vector", test->x == OSQP_NULL);
    mu_assert("CSC zero matrix: Error in i vector", test->i == OSQP_NULL);

    for(int i=0; i < n; i++)
    {
        mu_assert("CSC zero matrix: Error in p vector", test->p[i] == 0);
    }

    mu_assert("CSC zero matrix: Error in p vector", test->p[n] == 0);
  }

  SECTION("More rows than columns")
  {
    OSQPInt n = 5;
    OSQPInt m = 7;

    test = OSQPCscMatrix_zeros(m, n);

    mu_assert("CSC zero matrix: Error in matrix size", test->m == m);
    mu_assert("CSC zero matrix: Error in matrix size", test->n == n);
    mu_assert("CSC zero matrix: Error in matrix ownership", test->owned == 1);

    mu_assert("CSC zero matrix: Error in x vector", test->x == OSQP_NULL);
    mu_assert("CSC zero matrix: Error in i vector", test->i == OSQP_NULL);

    for(int i=0; i < n; i++)
    {
        mu_assert("CSC zero matrix: Error in p vector", test->p[i] == 0);
    }

    mu_assert("CSC zero matrix: Error in p vector", test->p[n] == 0);
  }

  OSQPCscMatrix_free(test);
}

TEST_CASE("CSC Diagonal scalar matrix", "[csc]" )
{
  OSQPCscMatrix* test;

  SECTION("Square")
  {
    OSQPInt n = 5;
    OSQPInt m = 5;

    test = OSQPCscMatrix_diag_scalar(n, m, 3.0);

    mu_assert("CSC diagonal matrix: Error in matrix size", test->m == n);
    mu_assert("CSC diagonal matrix: Error in matrix size", test->n == n);
    mu_assert("CSC diagonal matrix: Error in matrix ownership", test->owned == 1);

    for(int i=0; i < (n+1); i++)
    {
        mu_assert("CSC diagonal matrix: Error in p vector", test->p[i] == i);
    }

    for(int i=0; i < n; i++)
    {
        mu_assert("CSC diagonal matrix: Error in i vector", test->i[i] == i);
        mu_assert("CSC diagonal matrix: Error in x vector", test->x[i] == 3.0);
    }
  }

  SECTION("Fewer rows than columns")
  {
    OSQPInt n = 7;  // Number of columns
    OSQPInt m = 4;  // Number of rows

    test = OSQPCscMatrix_diag_scalar(m, n, 3.0);

    mu_assert("CSC diagonal matrix: Error in matrix size", test->m == m);
    mu_assert("CSC diagonal matrix: Error in matrix size", test->n == n);
    mu_assert("CSC diagonal matrix: Error in matrix ownership", test->owned == 1);

    // Column locations
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[0] == 0);
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[1] == 1);
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[2] == 2);
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[3] == 3);
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[4] == 4);
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[5] == 4);
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[6] == 4);
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[7] == 4);

    // Values loop
    for(int i=0; i < m; i++)
    {
        mu_assert("CSC diagonal matrix: Error in i vector", test->i[i] == i);
        mu_assert("CSC diagonal matrix: Error in x vector", test->x[i] == 3.0);
    }
  }

  SECTION("Fewer columns than rows")
  {
    OSQPInt n = 4;  // Number of columns
    OSQPInt m = 5;  // Number of rows

    test = OSQPCscMatrix_diag_scalar(m, n, 3.0);

    mu_assert("CSC diagonal matrix: Error in matrix size", test->m == m);
    mu_assert("CSC diagonal matrix: Error in matrix size", test->n == n);
    mu_assert("CSC diagonal matrix: Error in matrix ownership", test->owned == 1);

    // Column locations
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[0] == 0);
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[1] == 1);
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[2] == 2);
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[3] == 3);
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[4] == 4);

    // Values loop
    for(int i=0; i < n; i++)
    {
        mu_assert("CSC diagonal matrix: Error in i vector", test->i[i] == i);
        mu_assert("CSC diagonal matrix: Error in x vector", test->x[i] == 3.0);
    }
  }

  OSQPCscMatrix_free(test);
}


TEST_CASE("CSC Diagonal matrix", "[csc]" )
{
  OSQPCscMatrix* test;

  SECTION("Square")
  {
    OSQPInt   n = 5;
    OSQPInt   m = 5;
    OSQPFloat val[] = {1.0, 2.0, 3.0, 4.0, 5.0};

    test = OSQPCscMatrix_diag_vec(n, m, val);

    mu_assert("CSC diagonal matrix: Error in matrix size", test->m == n);
    mu_assert("CSC diagonal matrix: Error in matrix size", test->n == n);
    mu_assert("CSC diagonal matrix: Error in matrix ownership", test->owned == 1);

    for(int i=0; i < (n+1); i++)
    {
        mu_assert("CSC diagonal matrix: Error in p vector", test->p[i] == i);
    }

    for(int i=0; i < n; i++)
    {
        mu_assert("CSC diagonal matrix: Error in i vector", test->i[i] == i);
        mu_assert("CSC diagonal matrix: Error in x vector", test->x[i] == val[i]);
    }
  }

  SECTION("Fewer rows than columns")
  {
    OSQPInt n = 7;  // Number of columns
    OSQPInt m = 4;  // Number of rows
    OSQPFloat val[] = {1.0, 2.0, 3.0, 4.0};

    test = OSQPCscMatrix_diag_vec(m, n, val);

    mu_assert("CSC diagonal matrix: Error in matrix size", test->m == m);
    mu_assert("CSC diagonal matrix: Error in matrix size", test->n == n);
    mu_assert("CSC diagonal matrix: Error in matrix ownership", test->owned == 1);

    // Column locations
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[0] == 0);
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[1] == 1);
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[2] == 2);
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[3] == 3);
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[4] == 4);
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[5] == 4);
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[6] == 4);
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[7] == 4);

    // Values loop
    for(int i=0; i < m; i++)
    {
        mu_assert("CSC diagonal matrix: Error in i vector", test->i[i] == i);
        mu_assert("CSC diagonal matrix: Error in x vector", test->x[i] == val[i]);
    }
  }

  SECTION("Fewer columns than rows")
  {
    OSQPInt n = 4;  // Number of columns
    OSQPInt m = 5;  // Number of rows
    OSQPFloat val[] = {1.0, 2.0, 3.0, 4.0};

    test = OSQPCscMatrix_diag_vec(m, n, val);

    mu_assert("CSC diagonal matrix: Error in matrix size", test->m == m);
    mu_assert("CSC diagonal matrix: Error in matrix size", test->n == n);
    mu_assert("CSC diagonal matrix: Error in matrix ownership", test->owned == 1);

    // Column locations
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[0] == 0);
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[1] == 1);
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[2] == 2);
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[3] == 3);
    mu_assert("CSC diagonal matrix: Error in p vector", test->p[4] == 4);

    // Values loop
    for(int i=0; i < n; i++)
    {
        mu_assert("CSC diagonal matrix: Error in i vector", test->i[i] == i);
        mu_assert("CSC diagonal matrix: Error in x vector", test->x[i] == val[i]);
    }
  }

  OSQPCscMatrix_free(test);
}

TEST_CASE("CSC Identity matrix", "[csc]" )
{
  OSQPCscMatrix* test;

  SECTION("Square")
  {
    OSQPInt n = 5;

    test = OSQPCscMatrix_identity(n);

    mu_assert("CSC identity matrix: Error in matrix size", test->m == n);
    mu_assert("CSC identity matrix: Error in matrix size", test->n == n);
    mu_assert("CSC identity matrix: Error in matrix ownership", test->owned == 1);

    for(int i=0; i < (n+1); i++)
    {
        mu_assert("CSC identity matrix: Error in p vector", test->p[i] == i);
    }

    for(int i=0; i < n; i++)
    {
        mu_assert("CSC identity matrix: Error in i vector", test->i[i] == i);
        mu_assert("CSC identity matrix: Error in x vector", test->x[i] == 1.0);
    }
  }

  OSQPCscMatrix_free(test);
}