#include "stdio.h"
#include <stdlib.h>
#include "osqp.h"

#include "workspace.h"
#include <string.h>
#include <math.h>

#define N (20)
#define N_plus_K (22)
#define N_GAMMA (20)
#define N_GAMMA_PLUS_1 (21)

// Update linear cost for portfolio example
void update_q(c_float *q_new, c_float *mu, c_float gamma, int n){
    int j;
    for(j = 0; j < n; j++){
        q_new[j] = -mu[j] / gamma;
    }
}


int main(int argc, char **argv) {

    FILE *myfile;
    int i;
    c_float q_new[N_plus_K] = {0.0};
    c_float mu[N];
    c_float gamma_step, gamma[N_GAMMA_PLUS_1];
    
    // Load mu from file
    myfile = fopen("portfolio_data.txt", "r");
    if (myfile == NULL){
        printf("Can not open the data file.");
        return 1;
    }
    for(i = 0; i < N; i++){
        if(sizeof(c_float) == 4){
            fscanf(myfile, "%f", &mu[i]);
            // printf("%f\n", mu[i]);
        }else{
            fscanf(myfile, "%lf", &mu[i]);
            // printf("%lf\n", mu[i]);
        }
    }
    fclose(myfile);
    printf("\n");
    
    // Generate gamma parameters: (logspace(-2,2,N_GAMMA_PLUS_1))
    gamma[0] = log(1e-2);
    gamma[20] = log(1e2);
    gamma_step = (gamma[N_GAMMA] - gamma[0]) / N_GAMMA;
    for (i=1; i<N_GAMMA; i++){
        gamma[i] = gamma[i-1] + gamma_step;
    }
    for (i=0; i<=20; i++){
        gamma[i] = exp(gamma[i]);
    }
    
    // Load problem
    load_workspace(&workspace);

    // Solve a sequence of problems for varying risk aversion parameter gamma
    for(i = 0; i <= N_GAMMA; i++){
        // Update linear cost
        update_q(q_new, mu, gamma[i], N);
        osqp_update_lin_cost(&workspace, q_new);

        // Solve Problem
        osqp_solve(&workspace);

        // Print status
        printf("Risk aversion parameter:  %f\n", gamma[i]);
        printf("Status:                   %s\n", (&workspace)->info->status);
        printf("Number of iterations:     %d\n", (&workspace)->info->iter);
        printf("Objective value:          %f\n", (&workspace)->info->obj_val);
        printf("Primal residual:          %f\n", (&workspace)->info->pri_res);
        printf("Dual residual:            %f\n", (&workspace)->info->dua_res);
        printf("\n");
    }


    return 0;
};
