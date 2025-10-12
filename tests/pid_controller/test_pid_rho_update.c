#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "osqp.h"
#include "osqp.h"

/* Test data for a simple QP problem */
static OSQPFloat P_x[] = {4.0, 1.0, 2.0};
static OSQPInt P_i[] = {0, 0, 1};
static OSQPInt P_p[] = {0, 1, 3, 3};

static OSQPFloat A_x[] = {1.0, 1.0, 1.0, 1.0};
static OSQPInt A_i[] = {0, 1, 0, 2};
static OSQPInt A_p[] = {0, 2, 3, 4};

static OSQPFloat q[] = {1.0, 1.0, 1.0};
static OSQPFloat l[] = {1.0, 0.0, 0.0};
static OSQPFloat u[] = {1.0, 0.7, 0.7};

int test_pid_controller_rho_update(void) {
    OSQPSolver* solver;
    OSQPSettings* settings;
    OSQPCscMatrix P, A;
    OSQPInt exitflag;
    OSQPInt i;
    
    printf("Testing PID Controller Rho Update\n");
    printf("================================\n");
    
    /* Setup matrices */
    OSQPCscMatrix_set_data(&P, 3, 3, 3, P_x, P_i, P_p);
    OSQPCscMatrix_set_data(&A, 3, 3, 4, A_x, A_i, A_p);
    
    /* Create settings */
    settings = OSQPSettings_new();
    if (!settings) {
        printf("ERROR: Failed to create settings\n");
        return -1;
    }
    
    /* Configure PID controller settings */
    settings->pid_controller = 1;
    settings->KP = 0.1;
    settings->KI = 0.01;
    settings->KD = 0.05;
    settings->negate_K = 0;
    settings->pid_controller_sqrt = 0;
    settings->pid_controller_sqrt_mult = 0;
    settings->pid_controller_sqrt_mult_2 = 0;
    settings->pid_controller_log = 0;
    
    /* Other solver settings */
    settings->adaptive_rho = OSQP_ADAPTIVE_RHO_UPDATE_ITERATIONS;
    settings->adaptive_rho_interval = 25;
    settings->rho = 0.1;
    settings->max_iter = 200;
    settings->verbose = 1;
    settings->eps_abs = 1e-4;
    settings->eps_rel = 1e-4;
    
    printf("Initial settings:\n");
    printf("  PID Controller: %s\n", settings->pid_controller ? "enabled" : "disabled");
    printf("  KP: %.6f\n", settings->KP);
    printf("  KI: %.6f\n", settings->KI);
    printf("  KD: %.6f\n", settings->KD);
    printf("  Initial rho: %.6f\n", settings->rho);
    printf("\n");
    
    /* Setup solver */
    exitflag = osqp_setup(&solver, &P, q, &A, l, u, 3, 3, settings);
    if (exitflag != 0) {
        printf("ERROR: Setup failed with code %d\n", (int)exitflag);
        OSQPSettings_free(settings);
        return -1;
    }
    
    printf("Setup completed successfully\n");
    
    /* Solve the problem */
    printf("Starting solve with PID controller...\n");
    exitflag = osqp_solve(solver);
    
    if (exitflag != 0) {
        printf("ERROR: Solve failed with code %d\n", (int)exitflag);
    } else {
        printf("Solve completed successfully\n");
        printf("Final status: %s\n", solver->info->status);
        printf("Iterations: %d\n", (int)solver->info->iter);
        printf("Final rho estimate: %.6f\n", solver->info->rho_estimate);
        printf("Rho updates performed: %d\n", (int)solver->info->rho_updates);
        printf("Final objective value: %.6f\n", solver->info->obj_val);
        printf("Primal residual: %.2e\n", solver->info->prim_res);
        printf("Dual residual: %.2e\n", solver->info->dual_res);
        
        if (solver->solution) {
            printf("Solution x: [");
            for (i = 0; i < 3; i++) {
                printf("%.6f", solver->solution->x[i]);
                if (i < 2) printf(", ");
            }
            printf("]\n");
        }
    }
    
    /* Cleanup */
    osqp_cleanup(solver);
    OSQPSettings_free(settings);
    
    return exitflag;
}

int test_comparison_without_pid(void) {
    OSQPSolver* solver;
    OSQPSettings* settings;
    OSQPCscMatrix P, A;
    OSQPInt exitflag;
    
    printf("\nTesting without PID Controller (for comparison)\n");
    printf("==============================================\n");
    
    /* Setup matrices */
    OSQPCscMatrix_set_data(&P, 3, 3, 3, P_x, P_i, P_p);
    OSQPCscMatrix_set_data(&A, 3, 3, 4, A_x, A_i, A_p);
    
    /* Create settings */
    settings = OSQPSettings_new();
    if (!settings) {
        printf("ERROR: Failed to create settings\n");
        return -1;
    }
    
    /* Configure without PID controller */
    settings->pid_controller = 0;
    settings->adaptive_rho = OSQP_ADAPTIVE_RHO_UPDATE_ITERATIONS;
    settings->adaptive_rho_interval = 25;
    settings->rho = 0.1;
    settings->max_iter = 200;
    settings->verbose = 1;
    settings->eps_abs = 1e-4;
    settings->eps_rel = 1e-4;
    
    /* Setup solver */
    exitflag = osqp_setup(&solver, &P, q, &A, l, u, 3, 3, settings);
    if (exitflag != 0) {
        printf("ERROR: Setup failed with code %d\n", (int)exitflag);
        OSQPSettings_free(settings);
        return -1;
    }
    
    /* Solve the problem */
    printf("Starting solve without PID controller...\n");
    exitflag = osqp_solve(solver);
    
    if (exitflag != 0) {
        printf("ERROR: Solve failed with code %d\n", (int)exitflag);
    } else {
        printf("Solve completed successfully\n");
        printf("Final status: %s\n", solver->info->status);
        printf("Iterations: %d\n", (int)solver->info->iter);
        printf("Final rho estimate: %.6f\n", solver->info->rho_estimate);
        printf("Rho updates performed: %d\n", (int)solver->info->rho_updates);
        printf("Final objective value: %.6f\n", solver->info->obj_val);
        printf("Primal residual: %.2e\n", solver->info->prim_res);
        printf("Dual residual: %.2e\n", solver->info->dual_res);
    }
    
    /* Cleanup */
    osqp_cleanup(solver);
    OSQPSettings_free(settings);
    
    return exitflag;
}

int main(void) {
    int result1, result2;
    
    printf("OSQP PID Controller Rho Update Test\n");
    printf("===================================\n");
    printf("OSQP Version: %s\n\n", osqp_version());
    
    /* Test with PID controller */
    result1 = test_pid_controller_rho_update();
    
    /* Test without PID controller for comparison */
    result2 = test_comparison_without_pid();
    
    printf("\n=== Test Summary ===\n");
    printf("PID Controller test: %s\n", result1 == 0 ? "PASSED" : "FAILED");
    printf("Standard test: %s\n", result2 == 0 ? "PASSED" : "FAILED");
    
    if (result1 == 0 && result2 == 0) {
        printf("\nAll tests completed successfully!\n");
        printf("The PID controller rho update functionality appears to be working.\n");
        return 0;
    } else {
        printf("\nSome tests failed. Check the output above for details.\n");
        return 1;
    }
}