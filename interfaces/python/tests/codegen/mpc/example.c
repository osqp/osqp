#include "stdio.h"
#include "osqp.h"
#include "workspace.h"
#include <string.h>

#include <time.h>

static clock_t tic_timestart;
void tic(void) {
  tic_timestart = clock();
}
float tocq(void) {
  clock_t tic_timestop;
  tic_timestop = clock();
  return (float)(tic_timestop - tic_timestart) / CLOCKS_PER_SEC;
}


int main(int argc, char **argv) {

    int i;
    int n_trials = 100;

    // Load problem
    load_workspace(&workspace);

    // Solve Problem
    tic();
    for (i =0; i<n_trials; i++){
        osqp_solve(&workspace);
    }
    printf("Time per solve, for %d solves: %f ms\n", n_trials, tocq() / n_trials * 1000);

    // Print status
    printf("Status:                %s\n", (&workspace)->info->status);
    printf("Number of iterations:  %d\n", (&workspace)->info->iter);
    printf("Objective value:       %.4e\n", (&workspace)->info->obj_val);
    printf("Primal residual:       %.4e\n", (&workspace)->info->pri_res);
    printf("Dual residual:         %.4e\n", (&workspace)->info->dua_res);

    return 0;
};
