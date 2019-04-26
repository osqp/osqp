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


int tests_run = 0;

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

static char* all_tests() {
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

  mu_run_test(test_basic_qp);
  mu_run_test(test_basic_qp2);
  
  // remove custom allocator
  memset(&opts, 0, sizeof(OSQPGlobalOptions));
  osqp_set_global_options(&opts);

  mu_run_test(test_non_cvx);
  mu_run_test(test_primal_infeasibility);
  mu_run_test(test_primal_dual_infeasibility);
  mu_run_test(test_unconstrained);
  mu_run_test(test_update_matrices);
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
  printf("Tests run: %d\n", tests_run);

  if ((custom_malloc_called > 0) && (custom_realloc_called > 0) && (custom_free_called > 0)) {
    printf("Custom malloc called: %d\n", custom_malloc_called);
    printf("Custom realloc called: %d\n", custom_realloc_called);
    printf("Custom free called: %d\n", custom_free_called);
    printf("Custom memory management checks PASSED\n");
  }
  else {
    printf("Custom malloc/realloc/free were not called");
    result = "Custom malloc/free failed.";
  }

  return result != 0;
}
