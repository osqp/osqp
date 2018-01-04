#include <string.h>
#include <mex.h>
#include "osqp.h"
#include "workspace.h"



/*********************************
 * Timer Structs and Functions * *
 *********************************/

// Windows
#ifdef IS_WINDOWS

#include <windows.h>

typedef struct {
    LARGE_INTEGER tic;
	LARGE_INTEGER toc;
	LARGE_INTEGER freq;
} PyTimer;

// Mac
#elif defined IS_MAC

#include <mach/mach_time.h>

/* Use MAC OSX  mach_time for timing */
typedef struct {
	uint64_t tic;
	uint64_t toc;
	mach_timebase_info_data_t tinfo;
} PyTimer;

// Linux
#else

/* Use POSIX clocl_gettime() for timing on non-Windows machines */
#include <time.h>
#include <sys/time.h>

typedef struct {
	struct timespec tic;
	struct timespec toc;
} PyTimer;

#endif


/**
 * Timer Methods
 */

// Windows
#ifdef IS_WINDOWS

void tic(PyTimer* t)
{
    QueryPerformanceFrequency(&t->freq);
    QueryPerformanceCounter(&t->tic);
}

c_float toc(PyTimer* t)
{
    QueryPerformanceCounter(&t->toc);
    return ((t->toc.QuadPart - t->tic.QuadPart) / (c_float)t->freq.QuadPart);
}

// Mac
#elif defined IS_MAC

void tic(PyTimer* t)
{
    /* read current clock cycles */
    t->tic = mach_absolute_time();
}

c_float toc(PyTimer* t)
{
	uint64_t duration; /* elapsed time in clock cycles*/

    t->toc = mach_absolute_time();
    duration = t->toc - t->tic;

    /*conversion from clock cycles to nanoseconds*/
    mach_timebase_info(&(t->tinfo));
    duration *= t->tinfo.numer;
    duration /= t->tinfo.denom;

    return (c_float)duration / 1e9;
}


// Linux
#else

/* read current time */
void tic(PyTimer* t)
{
    clock_gettime(CLOCK_MONOTONIC, &t->tic);
}


/* return time passed since last call to tic on this timer */
c_float toc(PyTimer* t)
{
    struct timespec temp;

    clock_gettime(CLOCK_MONOTONIC, &t->toc);

    if ((t->toc.tv_nsec - t->tic.tv_nsec)<0) {
        temp.tv_sec = t->toc.tv_sec - t->tic.tv_sec-1;
        temp.tv_nsec = 1e9+t->toc.tv_nsec - t->tic.tv_nsec;
    } else {
        temp.tv_sec = t->toc.tv_sec - t->tic.tv_sec;
        temp.tv_nsec = t->toc.tv_nsec - t->tic.tv_nsec;
    }
    return (c_float)temp.tv_sec + (c_float)temp.tv_nsec / 1e9;
}


#endif

/****************************************
 * END( Timer Structs and Functions ) * *
 ****************************************/



// Internal utility functions
c_float* copyToCfloatVector(double * vecData, c_int numel);
void     castToDoubleArr(c_float *arr, double* arr_out, c_int len);
void     setToNaN(double* arr_out, c_int len);
#if EMBEDDED != 1
c_int*   copyDoubleToCintVector(double* vecData, c_int numel);
void     change1To0Indexing(c_int *vecData, c_int numel);
#endif


// Function handler
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{    
    // Get the command string
    char cmd[64];
	  if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
		mexErrMsgTxt("First input should be a command string less than 64 characters long.");


    // SOLVE
    if (!strcmp("solve", cmd)) {

        // Allocate timer
        double solve_time;
        PyTimer * timer;
        timer = mxMalloc(sizeof(PyTimer));

        if (nlhs != 5 || nrhs != 1){
            mexErrMsgTxt("Solve : wrong number of inputs / outputs");
        }

        // solve the problem
        tic(timer);                 // start timer
        osqp_solve(&workspace);
        solve_time = toc(timer);    // stop timer


        // Allocate space for solution
        // primal variables
        plhs[0] = mxCreateDoubleMatrix((&workspace)->data->n, 1, mxREAL);
        // dual variables
        plhs[1] = mxCreateDoubleMatrix((&workspace)->data->m, 1, mxREAL);
        // status value
        plhs[2] = mxCreateDoubleScalar((&workspace)->info->status_val);
        // number of iterations
        plhs[3] = mxCreateDoubleScalar((&workspace)->info->iter);
        // solve time
        plhs[4] = mxCreateDoubleScalar(solve_time);


        //copy results to mxArray outputs
        //assume that three outputs will always
        //be returned to matlab-side class wrapper
        if (((&workspace)->info->status_val != OSQP_PRIMAL_INFEASIBLE) &&
            ((&workspace)->info->status_val != OSQP_PRIMAL_INFEASIBLE_INACCURATE) &&
            ((&workspace)->info->status_val != OSQP_DUAL_INFEASIBLE) &&
            ((&workspace)->info->status_val != OSQP_DUAL_INFEASIBLE_INACCURATE)){

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

    #if EMBEDDED != 1
    // update matrix P
    if (!strcmp("update_P", cmd)) {

        // Fill Px and Px_idx
        const mxArray *Px = prhs[1];
        const mxArray *Px_idx = prhs[2];

        int Px_n = mxGetScalar(prhs[3]);

        // Copy vectors to ensure they are cast as c_float and c_int
        c_float *Px_vec;
        c_int *Px_idx_vec = NULL;
        if(!mxIsEmpty(Px)){
            Px_vec = copyToCfloatVector(mxGetPr(Px), Px_n);
        }
        if(!mxIsEmpty(Px_idx)){
            Px_idx_vec = copyDoubleToCintVector(mxGetPr(Px_idx), Px_n);
            
            // Change indexing to match C 0-based one
            change1To0Indexing(Px_idx_vec, Px_n);
            
        }

        if(!mxIsEmpty(Px)){
            osqp_update_P((&workspace), Px_vec, Px_idx_vec, Px_n);
        }

        // Free
        if(!mxIsEmpty(Px)) mxFree(Px_vec);
        if(!mxIsEmpty(Px_idx)) mxFree(Px_idx_vec);

        return;
    }

    // update matrix A
    if (!strcmp("update_A", cmd)) {

        // Fill Ax and Ax_idx
        const mxArray *Ax = prhs[1];
        const mxArray *Ax_idx = prhs[2];

        int Ax_n = mxGetScalar(prhs[3]);

        // Copy vectors to ensure they are cast as c_float and c_int
        c_float *Ax_vec;
        c_int *Ax_idx_vec = NULL;
        if(!mxIsEmpty(Ax)){
            Ax_vec = copyToCfloatVector(mxGetPr(Ax), Ax_n);
        }
        if(!mxIsEmpty(Ax_idx)){
            Ax_idx_vec = copyDoubleToCintVector(mxGetPr(Ax_idx), Ax_n);
            
            // Change indexing to match C 0-based one
            change1To0Indexing(Ax_idx_vec, Ax_n);
        }

        if(!mxIsEmpty(Ax)){
            osqp_update_A((&workspace), Ax_vec, Ax_idx_vec, Ax_n);
        }

        // Free
        if(!mxIsEmpty(Ax)) mxFree(Ax_vec);
        if(!mxIsEmpty(Ax_idx)) mxFree(Ax_idx_vec);

        return;
    }

    // update matrices P and A
    if (!strcmp("update_P_A", cmd)) {

        // Fill vectors
        const mxArray *Px = prhs[1];
        const mxArray *Px_idx = prhs[2];
        const mxArray *Ax = prhs[4];
        const mxArray *Ax_idx = prhs[5];

        int Px_n = mxGetScalar(prhs[3]);
        int Ax_n = mxGetScalar(prhs[6]);

        // Copy vectors to ensure they are cast as c_float and c_int
        c_float *Px_vec, *Ax_vec;
        c_int *Px_idx_vec = NULL;
        c_int *Ax_idx_vec = NULL;
        if(!mxIsEmpty(Px)){
            Px_vec = copyToCfloatVector(mxGetPr(Px), Px_n);
        }
        if(!mxIsEmpty(Px_idx)){
            Px_idx_vec = copyDoubleToCintVector(mxGetPr(Px_idx), Px_n);
            
            // Change indexing to match C 0-based one
            change1To0Indexing(Px_idx_vec, Px_n);
        }
        if(!mxIsEmpty(Ax)){
            Ax_vec = copyToCfloatVector(mxGetPr(Ax), Ax_n);
        }
        if(!mxIsEmpty(Ax_idx)){
            Ax_idx_vec = copyDoubleToCintVector(mxGetPr(Ax_idx), Ax_n);
            
            // Change indexing to match C 0-based one
            change1To0Indexing(Ax_idx_vec, Ax_n);
        }

        if(!mxIsEmpty(Ax) && !mxIsEmpty(Px)){
            osqp_update_P_A((&workspace), Px_vec, Px_idx_vec, Px_n,
                                          Ax_vec, Ax_idx_vec, Ax_n);
        }

        // Free
        if(!mxIsEmpty(Px)) mxFree(Px_vec);
        if(!mxIsEmpty(Px_idx)) mxFree(Px_idx_vec);
        if(!mxIsEmpty(Ax)) mxFree(Ax_vec);
        if(!mxIsEmpty(Ax_idx)) mxFree(Ax_idx_vec);

        return;
    }
    #endif  // end EMBEDDED

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

#if EMBEDDED != 1
c_int* copyDoubleToCintVector(double* vecData, c_int numel){
  // This memory needs to be freed!

  c_int* out;
  c_int i;

  out = mxMalloc(numel * sizeof(c_int));

  //copy data
  for(i=0; i < numel; i++){
      out[i] = (c_int)vecData[i];
  }
  return out;

}

void change1To0Indexing(c_int *vecData, c_int numel){
c_int i; // Indexing
for(i=0; i < numel; i++){
vecData[i] -= 1;  // Decrease index by 1
}
}


#endif  // end EMBEDDED
