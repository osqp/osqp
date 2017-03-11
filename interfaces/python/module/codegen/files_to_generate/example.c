#include "stdio.h"
#include "osqp.h"

#include "workspace.h"
#include <string.h>

int main(int argc, char **argv) {

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
