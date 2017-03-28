#include "mex.h"
#include "osqp.h"
#include "workspace.h"


// internal utility functions
c_float* copyToCfloatVector(double * vecData, c_int numel);
void castToDoubleArr(c_float *arr, double* arr_out, c_int len);
void setToNaN(double* arr_out, c_int len);


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Get the command string
    char cmd[64];
	  if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
		mexErrMsgTxt("First input should be a command string less than 64 characters long.");


    // SOLVE
    if (!strcmp("solve", cmd)) {
        if (nlhs != 4 || nrhs != 1){
            mexErrMsgTxt("Solve : wrong number of inputs / outputs");
        }
        if(!(&workspace)){
            mexErrMsgTxt("No problem data has been given.");
        }
        // solve the problem
        osqp_solve(&workspace);


        // Allocate space for solution
        // primal variables
        plhs[0] = mxCreateDoubleMatrix((&workspace)->data->n, 1, mxREAL);
        // dual variables
        plhs[1] = mxCreateDoubleMatrix((&workspace)->data->m, 1, mxREAL);
        // status value
        plhs[2] = mxCreateDoubleScalar((&workspace)->info->status_val);
        // number of iterations
        plhs[3] = mxCreateDoubleScalar((&workspace)->info->iter);


        //copy results to mxArray outputs
        //assume that three outputs will always
        //be returned to matlab-side class wrapper
        if (((&workspace)->info->status_val != OSQP_PRIMAL_INFEASIBLE) &&
            ((&workspace)->info->status_val != OSQP_DUAL_INFEASIBLE)){

            //primal variables
            castToDoubleArr((&workspace)->solution->x, mxGetPr(plhs[0]), (&workspace)->data->n);

            //dual variables
            castToDoubleArr((&workspace)->solution->y, mxGetPr(plhs[1]), (&workspace)->data->m);

        } else { // Problem is primal or dual infeasible -> NaN values

            // Set primal and dual variables to NaN
            setToNaN(mxGetPr(plhs[0]), (&workspace)->data->n);
            setToNaN(mxGetPr(plhs[1]), (&workspace)->data->m);
        }

        return;
    }

    
    // update linear cost
    if (!strcmp("update_lin_cost", cmd)) {

        // Fill q
        const mxArray *q = prhs[1];

        // Copy vector to ensure it is cast as c_float
        c_float *q_vec;
        if(!mxIsEmpty(q)){
            q_vec = copyToCfloatVector(mxGetPr(q), (&workspace)->data->n);
        }

        if(!mxIsEmpty(q)){
          osqp_update_lin_cost(&workspace, q_vec);
        }

        // Free
        if(!mxIsEmpty(q)) mxFree(q_vec);

        return;
    }
    
    
    // update lower bound
    if (!strcmp("update_lower_bound", cmd)) {

        // Fill l
        const mxArray *l = prhs[1];

        // Copy vector to ensure it is cast as c_float
        c_float *l_vec;
        if(!mxIsEmpty(l)){
            l_vec = copyToCfloatVector(mxGetPr(l), (&workspace)->data->m);
        }

        if(!mxIsEmpty(l)){
          osqp_update_lower_bound(&workspace, l_vec);
        }

        // Free
        if(!mxIsEmpty(l)) mxFree(l_vec);

        return;
    }


    // update upper bound
    if (!strcmp("update_upper_bound", cmd)) {

        // Fill l
        const mxArray *u = prhs[1];

        // Copy vector to ensure it is cast as c_float
        c_float *u_vec;
        if(!mxIsEmpty(u)){
            u_vec = copyToCfloatVector(mxGetPr(u), (&workspace)->data->m);
        }

        if(!mxIsEmpty(u)){
          osqp_update_upper_bound(&workspace, u_vec);
        }

        // Free
        if(!mxIsEmpty(u)) mxFree(u_vec);

        return;
    }


    // update bounds
    if (!strcmp("update_bounds", cmd)) {

        // Fill l, u
        const mxArray *l = prhs[1];
        const mxArray *u = prhs[2];

        // Copy vectors to ensure they are cast as c_float
        c_float *l_vec;
        c_float *u_vec;
        if(!mxIsEmpty(l)){
            l_vec = copyToCfloatVector(mxGetPr(l), (&workspace)->data->m);
        }
        if(!mxIsEmpty(u)){
            u_vec = copyToCfloatVector(mxGetPr(u), (&workspace)->data->m);
        }

        if(!mxIsEmpty(u)){
            osqp_update_bounds(&workspace, l_vec, u_vec);
        }

        // Free
        if(!mxIsEmpty(l)) mxFree(l_vec);
        if(!mxIsEmpty(u)) mxFree(u_vec);

        return;
    }


    // Got here, so command not recognized
    mexErrMsgTxt("Command not recognized.");
}


c_float* copyToCfloatVector(double* vecData, c_int numel){
    // This memory needs to be freed!

    c_float* out;
    c_int i;

    out = mxMalloc(numel * sizeof(c_float));

    //copy data
    for(i=0; i < numel; i++){
        out[i] = (c_float)vecData[i];
    }
    return out;

}

void castToDoubleArr(c_float *arr, double* arr_out, c_int len){
    c_int i;
    for (i = 0; i < len; i++) {
        arr_out[i] = (double)arr[i];
    }
}

void setToNaN(double* arr_out, c_int len){
    c_int i;
    for (i = 0; i < len; i++) {
        arr_out[i] = mxGetNaN();
    }
}
