#include "osqp.h"
#include "minunit.h"
#include "qptests/chain80w/chain80w.h"
#ifndef TEST_CHAIN80W_H
#define TEST_CHAIN80W_H

static char * test_chain80w()
{
/* local variables */
c_int exitflag = 0;  // No errors

// Problem settings
Settings * settings = (Settings *)c_malloc(sizeof(Settings));

// Structures
Work * work;  // Workspace

// Generate problem data
Data * data = generate_problem_diesel();

c_print("\nTest chain80w\n");
c_print("-------------\n");

// Define Solver settings as default
set_default_settings(settings);

// Setup workspace
work = osqp_setup(data, settings);

if (!work) {
c_print("Setup error!\n");
exitflag = 1;
}
else {
// Solve Problem
osqp_solve(work);

// Clean workspace
osqp_cleanup(work);

}

// Cleanup data
clean_problem_diesel(data);

mu_assert("\nError in chain80w test.", exitflag == 0 );

//Cleanup
c_free(settings);

return 0;
}

#endif
