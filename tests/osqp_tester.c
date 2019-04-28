/* OSQP TESTER MODULE */

/* THE CODE FOR MINIMAL UNIT TESTING HAS BEEN TAKEN FROM
   http://www.jera.com/techinfo/jtns/jtn002.html */

#include <stdio.h>

#include "minunit.h"
#include "osqp.h"
#include "osqp_tester.h"

// Include tests
#include "lin_alg/test_lin_alg.h"
#include "solve_linsys/test_solve_linsys.h"
#include "basic_qp/test_basic_qp.h"
#include "basic_qp2/test_basic_qp2.h"
#include "non_cvx/test_non_cvx.h"
#include "primal_dual_infeasibility/test_primal_dual_infeasibility.h"
#include "primal_infeasibility/test_primal_infeasibility.h"
#include "unconstrained/test_unconstrained.h"
#include "update_matrices/test_update_matrices.h"

void* custom_malloc(size_t);
void* custom_calloc(size_t, size_t);
void  custom_free(void*);
void* custom_realloc(void*, size_t);

int check_custom_allocators_called();
int check_custom_allocator_count_zero();

void reset_custom_allocator_counts();

int osqp_tests_run = 0;

int custom_malloc_called = 0;
int custom_free_called = 0;
int custom_realloc_called = 0;

void* custom_malloc(size_t size)
{
  custom_malloc_called++;
  return malloc(size);
}

void* custom_calloc(size_t count, size_t size)
{
  void* ptr = custom_malloc(count * size);
  memset(ptr, 0, count * size);
  return ptr;
}

void custom_free(void* ptr)
{
  custom_free_called++;
  return free(ptr);
}

void* custom_realloc(void* ptr, size_t size)
{
  custom_realloc_called++;
  return realloc(ptr, size);
}

int check_custom_allocators_called()
{
  printf("Custom malloc called: %d\n", custom_malloc_called);
  printf("Custom realloc called: %d\n", custom_realloc_called);
  printf("Custom free called: %d\n", custom_free_called);

  return (custom_malloc_called && custom_realloc_called && custom_free_called);
}

int check_custom_allocator_count_zero()
{
  return !(custom_malloc_called || custom_realloc_called || custom_free_called);
}

void reset_custom_allocator_counts()
{
  custom_malloc_called = custom_realloc_called = custom_free_called = 0;
}

static char* all_tests() {
  // ensure that memory allocation works if
  // osqp_set_global_options() has not been
  // called yet.
  mu_run_test(test_lin_alg);
  mu_run_test(test_solve_linsys);

  // set custom allocator
  OSQPGlobalOptions opts = { 
    custom_malloc, 
    custom_calloc, 
    custom_free, 
    custom_realloc 
  };
  osqp_set_global_options(&opts);

  // ensure counts are 0
  mu_assert("Custom malloc/realloc/free counts nonzero.", check_custom_allocator_count_zero());

  // run tests with the custom allocator
  mu_run_test(test_basic_qp);
  mu_run_test(test_basic_qp2);
  
  // remove custom allocator
  memset(&opts, 0, sizeof(OSQPGlobalOptions));
  osqp_set_global_options(&opts);

  // ensure counts are nonzero
  mu_assert("Custom malloc/realloc/free were not called", check_custom_allocators_called());
  // reset counts to zero
  reset_custom_allocator_counts();

  // check that default memory allocation works
  // after allocators have been reset
  mu_run_test(test_non_cvx);
  mu_run_test(test_primal_infeasibility);
  mu_run_test(test_primal_dual_infeasibility);
  mu_run_test(test_unconstrained);
  mu_run_test(test_update_matrices);

  // ensure counts remained zero
  mu_assert("Custom malloc/realloc/free counts nonzero.", check_custom_allocator_count_zero());

  return 0;
}

int main(void) {
  char *result = all_tests();

  if (result != 0) {
    printf("%s\n", result);
  }
  else {
    printf("ALL TESTS PASSED\n");
  }
  printf("Tests run: %d\n", osqp_tests_run);

  return result != 0;
}
