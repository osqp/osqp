#include "osqp.h"     // OSQP API
#include "cs.h"       // CSC data structure
#include "util.h"     // Utilities for testing
#include "minunit.h"  // Basic testing script header

#include "infeasibility/data.h"


static char * test_infeasible_qp_solve()
{
    // Problem settings
    OSQPSettings * settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

    // Structures
    OSQPWorkspace * work;  // Workspace
    OSQPData * data;  // Data
    infeasibility_sols_data *  sols_data;

    // Populate data
    data = generate_problem_infeasibility();
    sols_data = generate_problem_infeasibility_sols_data();


    // Define Solver settings as default
    set_default_settings(settings);
    settings->max_iter = 10000;
    settings->alpha = 1.6;
    settings->polish = 1;
    settings->scaling = 0;
    settings->verbose = 0;
    settings->warm_start = 0;

    // Setup workspace
    work = osqp_setup(data, settings);

    // Setup correct
    mu_assert("Infeasible QP test solve: Setup error!", work != OSQP_NULL);

    // Solve Problem
    osqp_solve(work);

    // Compare solver statuses
    mu_assert("Infeasible QP test solve: Error in solver status!",
              work->info->status_val == sols_data->status_test );


    // Clean workspace
    osqp_cleanup(work);


    // Cleanup data
    clean_problem_infeasibility(data);
    clean_problem_infeasibility_sols_data(sols_data);

    // Cleanup
    c_free(settings);

    return 0;
}


static char * test_infeasibility()
{

    mu_run_test(test_infeasible_qp_solve);


    return 0;
}
