#include "stdio.h"
#include "osqp.h"

#include "workspace.h"
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {

    int i;
    float gamma_step, gamma[21];

    // gamma = logspace(1e-2, 1e2, 21)
    gamma[0] = log(1e-2);
    gamma[20] = log(1e2);
    gamma_step = (gamma[20] - gamma[0])/20;
    for (i=1; i<20; i++){
        gamma[i] = gamma[i-1] + gamma_step;
    }
    for (i=0; i<=20; i++){
        gamma[i] = exp(gamma[i]);
        printf("%f\n", gamma[i]);   // Check if logspace was computed correctly
    }


    // Load problem
    load_workspace(&workspace);

    // Solve Problem
    osqp_solve(&workspace);

    // Print status
    printf("Status:                %s\n", (&workspace)->info->status);
    printf("Number of iterations:  %d\n", (&workspace)->info->iter);
    printf("Objective value:       %.4e\n", (&workspace)->info->obj_val);
    printf("Primal residual:       %.4e\n", (&workspace)->info->pri_res);
    printf("Dual residual:         %.4e\n", (&workspace)->info->dua_res);

    return 0;
};
