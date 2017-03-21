#include "mex.h"
#include "matrix.h"
#include "osqp_mex.hpp"
#include "osqp.h"

// all of the OSQP_INFO fieldnames as strings
const char *OSQP_INFO_FIELDS[] = {"iter",         //c_int
                                "status" ,        //char*
                                "status_val" ,    //c_int
                                "status_polish",  //c_int
                                "obj_val",        //c_float
                                "pri_res",        //c_float
                                "dua_res",        //c_float
                                "setup_time",     //c_float, only used if PROFILING
                                "solve_time",     //c_float, only used if PROFILING
                                "polish_time",    //c_float, only used if PROFILING
                                "run_time"};      //c_float, only used if PROFILING

const char* OSQP_SETTINGS_FIELDS[] =
                                //the following subset can't be changed after initilization
                               {"rho",            //c_float
                                "sigma",            //c_float
                                "scaling",        //c_int
                                "scaling_norm",   //c_int
                                "scaling_iter",   //c_int
                                //the following subset can be changed after initilization
                                "max_iter",                     //c_int
                                "eps_abs",                      //c_float
                                "eps_rel",                      //c_float
                                "eps_prim_inf",                      //c_float
                                "eps_dual_inf",                      //c_float
                                "alpha",                        //c_float
                                "delta",                        //c_float
                                "polish",                       //c_int
                                "pol_refine_iter",              //c_int
                                "verbose",                      //c_int
                                "early_terminate",              //c_int
                                "early_terminate_interval",     //c_int
                                "warm_start"};                  //c_int



// wrapper class for all osqp data and settings
class OsqpData
{
public:
  OsqpData(){
    work = NULL;
  }
  OSQPWorkspace     * work;     // Workspace
};

// internal utility functions
c_int*    copyToCintVector(mwIndex * vecData, c_int numel);
c_float*  copyToCfloatVector(double * vecData, c_int numel);
void castToDoubleArr(c_float *arr, double* arr_out, c_int len);
void setToNaN(double* arr_out, c_int len);
mxArray*  copyInfoToMxStruct(OSQPInfo*);
mxArray*  copySettingsToMxStruct(OSQPSettings*);
void      copyMxStructToSettings(const mxArray*,OSQPSettings*);
void      copyUpdatedSettingsToWork(const mxArray* ,OsqpData*);


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Get the command string
    char cmd[64];
	  if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
		mexErrMsgTxt("First input should be a command string less than 64 characters long.");

    // new object
    if (!strcmp("new", cmd)) {
        // Check parameters
        if (nlhs != 1){
            mexErrMsgTxt("New: One output expected.");
          }
        // Return a handle to a new C++ wrapper instance
        plhs[0] = convertPtr2Mat<OsqpData>(new OsqpData);
        return;
    }

    // Check for a second input, which should be the class instance handle
    if (nrhs < 2)
		mexErrMsgTxt("Second input should be a class instance handle.");

    // Get the class instance pointer from the second input
    OsqpData* osqpData = convertMat2Ptr<OsqpData>(prhs[1]);

    // delete the object and its data
    if (!strcmp("delete", cmd)) {

        //clean up the problem workspace
        if(osqpData->work){
            osqp_cleanup(osqpData->work);
        }

        //clean up the handle object
        destroyObject<OsqpData>(prhs[1]);
        // Warn if other commands were ignored
        if (nlhs != 0 || nrhs != 2)
            mexWarnMsgTxt("Delete: Unexpected arguments ignored.");
        return;
    }

    // report the current settings
    if (!strcmp("current_settings", cmd)) {

      //throw an error if this is called before solver is configured
      if(!osqpData->work){
        mexErrMsgTxt("Solver is uninitialized.  No settings have been configured.");
      }
      //report the current settings
      plhs[0] = copySettingsToMxStruct(osqpData->work->settings);
      return;
    }

    // write_settings
    if (!strcmp("update_settings", cmd)) {

      //overwrite the current settings.  Mex function is responsible
      //for disallowing overwrite of selected settings after initialization,
      //and for all error checking
      //throw an error if this is called before solver is configured
      if(!osqpData->work){
        mexErrMsgTxt("Solver is uninitialized.  No settings have been configured.");
      }
      copyUpdatedSettingsToWork(prhs[2],osqpData);
      return;
    }

    // report the default settings
    if (!strcmp("default_settings", cmd)) {

      //Create a Settings structure in default form and report the results
      //Useful for external solver packages (e.g. Yalmip) that want to
      //know which solver settings are supported
      OSQPSettings* defaults = (OSQPSettings *)mxCalloc(1,sizeof(OSQPSettings));
      set_default_settings(defaults);
      plhs[0] = copySettingsToMxStruct(defaults);
      mxFree(defaults);
      return;
    }

    // setup
    if (!strcmp("setup", cmd)) {

        //throw an error if this is called more than once
        if(osqpData->work){
          mexErrMsgTxt("Solver is already initialized with problem data.");
        }

        //Create data and settings containers
        OSQPSettings* settings = (OSQPSettings *)mxCalloc(1,sizeof(OSQPSettings));
        OSQPData*     data     = (OSQPData *)mxCalloc(1,sizeof(OSQPData));

        // handle the problem data first.  Matlab-side
        // class wrapper is responsible for ensuring that
        // P and A are sparse matrices,  everything
        // else is a dense vector and all inputs are
        // of compatible dimension

        // Get mxArrays
        const mxArray* P  = prhs[4];
        const mxArray* q  = prhs[5];
        const mxArray* A  = prhs[6];
        const mxArray* l = prhs[7];
        const mxArray* u = prhs[8];

        // Create Data Structure
        data->n  = mxGetScalar(prhs[2]);
        data->m  = mxGetScalar(prhs[3]);
        data->q = copyToCfloatVector(mxGetPr(q), data->n);
        data->l = copyToCfloatVector(mxGetPr(l), data->m);
        data->u = copyToCfloatVector(mxGetPr(u), data->m);

        // Matrix P:  nnz = P->p[n]
        c_int * Pp = copyToCintVector(mxGetJc(P), data->n + 1);
        c_int * Pi = copyToCintVector(mxGetIr(P), Pp[data->n]);
        c_float * Px = copyToCfloatVector(mxGetPr(P), Pp[data->n]);
        data->P  = csc_matrix(data->n, data->n, Pp[data->n], Px, Pi, Pp);

        // Matrix A: nnz = A->p[n]
        c_int * Ap = copyToCintVector(mxGetJc(A), data->n + 1);
        c_int * Ai = copyToCintVector(mxGetIr(A), Ap[data->n]);
        c_float * Ax = copyToCfloatVector(mxGetPr(A), Ap[data->n]);
        data->A  = csc_matrix(data->m, data->n, Ap[data->n], Ax, Ai, Ap);

        // Create Settings
        const mxArray* mxSettings = prhs[9];
        if(mxIsEmpty(mxSettings)){
          // use defaults
          set_default_settings(settings);
        } else {
          //populate settings structure from mxArray input
          copyMxStructToSettings(mxSettings, settings);
        }

        // Setup workspace
        osqpData->work = osqp_setup(data, settings);
        if(!osqpData->work){
           mexErrMsgTxt("Invalid problem setup");
         }

         //cleanup temporary structures
         // Data
         if (data->q) mxFree(data->q);
         if (data->l) mxFree(data->l);
         if (data->u) mxFree(data->u);
         if (Px) mxFree(Px);
         if (Pi) mxFree(Pi);
         if (Pp) mxFree(Pp);
         if (Ax) mxFree(Ax);
         if (Ai) mxFree(Ai);
         if (Ap) mxFree(Ap);
         mxFree(data);
         // Settings
         mxFree(settings);
        return;

    }

    // get #constraints and variables
    if (!strcmp("get_dimensions", cmd)) {

        //throw an error if this is called before solver is configured
        if(!osqpData->work){
          mexErrMsgTxt("Solver has not been initialized.");
        }

        plhs[0] = mxCreateDoubleScalar(osqpData->work->data->n);
        plhs[1] = mxCreateDoubleScalar(osqpData->work->data->m);

        return;
    }

    // report the version
    if (!strcmp("version", cmd)) {

        plhs[0] = mxCreateString(osqp_version());

        return;
    }

    // update linear cost and bounds
    if (!strcmp("update", cmd)) {

        //throw an error if this is called before solver is configured
        if(!osqpData->work){
          mexErrMsgTxt("Solver has not been initialized.");
        }

        // Fill q, l, u
        const mxArray *q = prhs[2];
        const mxArray *l = prhs[3];
        const mxArray *u = prhs[4];


        // Copy vectors to ensure they are cast as c_float
        c_float *q_vec;
        c_float *l_vec;
        c_float *u_vec;
        if(!mxIsEmpty(q)){
            q_vec = copyToCfloatVector(mxGetPr(q),
                                       osqpData->work->data->n);
        }
        if(!mxIsEmpty(l)){
            l_vec = copyToCfloatVector(mxGetPr(l),
                                       osqpData->work->data->m);
        }
        if(!mxIsEmpty(u)){
            u_vec = copyToCfloatVector(mxGetPr(u),
                                       osqpData->work->data->m);
        }

        if(!mxIsEmpty(q)){
          osqp_update_lin_cost(osqpData->work,q_vec);
        }
        if(!mxIsEmpty(l) && !mxIsEmpty(u) ){
          osqp_update_bounds(osqpData->work,l_vec,u_vec);
        }
        else if(!mxIsEmpty(l)){
          osqp_update_lower_bound(osqpData->work,l_vec);
        }
        else if(!mxIsEmpty(u)){
          osqp_update_upper_bound(osqpData->work,u_vec);
        }

        // Free vectors
        if(!mxIsEmpty(q)) mxFree(q_vec);
        if(!mxIsEmpty(l)) mxFree(l_vec);
        if(!mxIsEmpty(u)) mxFree(u_vec);

        return;
    }


    // Warm start x and y variables
    if (!strcmp("warm_start", cmd)) {

        //throw an error if this is called before solver is configured
        if(!osqpData->work){
          mexErrMsgTxt("Solver has not been initialized.");
        }

        // Fill x and y
        const mxArray *x = prhs[2];
        const mxArray *y = prhs[3];


        // Copy vectors to ensure they are cast as c_float
        c_float *x_vec;
        c_float *y_vec;

        if(!mxIsEmpty(x)){
            x_vec = copyToCfloatVector(mxGetPr(x),
                                       osqpData->work->data->n);
        }
        if(!mxIsEmpty(y)){
            y_vec = copyToCfloatVector(mxGetPr(y),
                                       osqpData->work->data->m);
        }

        // Warm start x and y
        osqp_warm_start(osqpData->work, x_vec, y_vec);

        // Free vectors
        if(!mxIsEmpty(x)) mxFree(x_vec);
        if(!mxIsEmpty(y)) mxFree(y_vec);

        return;
    }


    // Warm start x variable
    if (!strcmp("warm_start_x", cmd)) {

        //throw an error if this is called before solver is configured
        if(!osqpData->work){
          mexErrMsgTxt("Solver has not been initialized.");
        }

        // Fill x and y
        const mxArray *x = prhs[2];


        // Copy vectors to ensure they are cast as c_float
        c_float *x_vec;

        if(!mxIsEmpty(x)){
            x_vec = copyToCfloatVector(mxGetPr(x),
                                       osqpData->work->data->n);
        }

        // Warm start x
        osqp_warm_start_x(osqpData->work, x_vec);

        // Free vectors
        if(!mxIsEmpty(x)) mxFree(x_vec);

        return;
    }


    // Warm start y variable
    if (!strcmp("warm_start_y", cmd)) {

        //throw an error if this is called before solver is configured
        if(!osqpData->work){
          mexErrMsgTxt("Solver has not been initialized.");
        }

        // Fill x and y
        const mxArray *y = prhs[2];


        // Copy vectors to ensure they are cast as c_float
        c_float *y_vec;

        if(!mxIsEmpty(y)){
            y_vec = copyToCfloatVector(mxGetPr(y),
                                       osqpData->work->data->m);
        }

        // Warm start x
        osqp_warm_start_y(osqpData->work, y_vec);

        // Free vectors
        if(!mxIsEmpty(y)) mxFree(y_vec);

        return;
    }



    // SOLVE
    if (!strcmp("solve", cmd)) {
        if (nlhs != 3 || nrhs != 2){
          mexErrMsgTxt("Solve : wrong number of inputs / outputs");
        }
        if(!osqpData->work){
            mexErrMsgTxt("No problem data has been given.");
        }
        // solve the problem
        osqp_solve(osqpData->work);


        // Allocate space for solution
        // primal variables
        plhs[0] = mxCreateDoubleMatrix(osqpData->work->data->n,1,mxREAL);
        // dual variables
        plhs[1] = mxCreateDoubleMatrix(osqpData->work->data->m,1,mxREAL);

        //copy results to mxArray outputs
        //assume that three outputs will always
        //be returned to matlab-side class wrapper
        if ((osqpData->work->info->status_val != OSQP_PRIMAL_INFEASIBLE) &&
            (osqpData->work->info->status_val != OSQP_DUAL_INFEASIBLE)){

            //primal variables
            castToDoubleArr(osqpData->work->solution->x, mxGetPr(plhs[0]), osqpData->work->data->n);

            //dual variables
            castToDoubleArr(osqpData->work->solution->y, mxGetPr(plhs[1]), osqpData->work->data->m);

        } else { // Problem is primal or dual infeasible -> NaN values

            // Set primal and dual variables to NaN
            setToNaN(mxGetPr(plhs[0]), osqpData->work->data->n);
            setToNaN(mxGetPr(plhs[1]), osqpData->work->data->m);
        }

        // If problem is primal infeasible, set objective value to infinity
        if (osqpData->work->info->status_val == OSQP_PRIMAL_INFEASIBLE){
            osqpData->work->info->obj_val = mxGetInf();
        }

        // If problem is dual infeasible, set objective value to -infinity
        if (osqpData->work->info->status_val == OSQP_DUAL_INFEASIBLE){
            osqpData->work->info->obj_val = -mxGetInf();
        }

        plhs[2] = copyInfoToMxStruct(osqpData->work->info); // Info structure

        return;
    }

    if (!strcmp("constant", cmd)) { // Return solver constants

        char constant[32];
        mxGetString(prhs[2], constant, sizeof(constant));

        if (!strcmp("OSQP_INFTY", constant)){
            plhs[0] = mxCreateDoubleScalar(OSQP_INFTY);
            return;
        }
        if (!strcmp("OSQP_NAN", constant)){
            plhs[0] = mxCreateDoubleScalar(OSQP_NAN);
            return;
        }

        if (!strcmp("OSQP_SOLVED", constant)){
            plhs[0] = mxCreateDoubleScalar(OSQP_SOLVED);
            return;
        }

        if (!strcmp("OSQP_UNSOLVED", constant)){
            plhs[0] = mxCreateDoubleScalar(OSQP_UNSOLVED);
            return;
        }

        if (!strcmp("OSQP_PRIMAL_INFEASIBLE", constant)){
            plhs[0] = mxCreateDoubleScalar(OSQP_PRIMAL_INFEASIBLE);
            return;
        }

        if (!strcmp("OSQP_DUAL_INFEASIBLE", constant)){
            plhs[0] = mxCreateDoubleScalar(OSQP_DUAL_INFEASIBLE);
            return;
        }

        if (!strcmp("OSQP_MAX_ITER_REACHED", constant)){
            plhs[0] = mxCreateDoubleScalar(OSQP_MAX_ITER_REACHED);
            return;
        }


        mexErrMsgTxt("Constant not recognized.");

        return;
    }


    // Got here, so command not recognized
    mexErrMsgTxt("Command not recognized.");
}


c_float*  copyToCfloatVector(double* vecData, c_int numel){
  // This memory needs to be freed!
  c_float* out = (c_float*)c_malloc(numel * sizeof(c_float));

  //copy data
  for(c_int i=0; i < numel; i++){
      out[i] = (c_float)vecData[i];
  }
  return out;

}

c_int*  copyToCintVector(mwIndex* vecData, c_int numel){
  // This memory needs to be freed!
  c_int* out = (c_int*)c_malloc(numel * sizeof(c_int));

  //copy data
  for(c_int i=0; i < numel; i++){
      out[i] = (c_int)vecData[i];
  }
  return out;

}

void castToDoubleArr(c_float *arr, double* arr_out, c_int len) {
    for (c_int i = 0; i < len; i++) {
        arr_out[i] = (double)arr[i];
    }
}

void setToNaN(double* arr_out, c_int len){
    c_int i;
    for (i = 0; i < len; i++) {
        arr_out[i] = mxGetNaN();
    }
}


mxArray* copyInfoToMxStruct(OSQPInfo* info){

  //create mxArray with the right number of fields
  int nfields  = sizeof(OSQP_INFO_FIELDS) / sizeof(OSQP_INFO_FIELDS[0]);
  mxArray* mxPtr = mxCreateStructMatrix(1,1,nfields,OSQP_INFO_FIELDS);

  //map the OSQP_INFO fields one at a time into mxArrays
  //matlab all numeric values as doubles
  mxSetField(mxPtr, 0, "iter",          mxCreateDoubleScalar(info->iter));
  mxSetField(mxPtr, 0, "status",        mxCreateString(info->status));
  mxSetField(mxPtr, 0, "status_val",    mxCreateDoubleScalar(info->status_val));
  mxSetField(mxPtr, 0, "status_polish", mxCreateDoubleScalar(info->status_polish));
  mxSetField(mxPtr, 0, "obj_val",       mxCreateDoubleScalar(info->obj_val));
  mxSetField(mxPtr, 0, "pri_res",       mxCreateDoubleScalar(info->pri_res));
  mxSetField(mxPtr, 0, "dua_res",       mxCreateDoubleScalar(info->dua_res));

  #if PROFILING > 0
  //if not profiling, these fields will be empty
  mxSetField(mxPtr, 0, "setup_time",  mxCreateDoubleScalar(info->setup_time));
  mxSetField(mxPtr, 0, "solve_time",  mxCreateDoubleScalar(info->solve_time));
  mxSetField(mxPtr, 0, "polish_time", mxCreateDoubleScalar(info->polish_time));
  mxSetField(mxPtr, 0, "run_time",    mxCreateDoubleScalar(info->run_time));
  #endif

  return mxPtr;

}


mxArray* copySettingsToMxStruct(OSQPSettings* settings){

  int nfields  = sizeof(OSQP_SETTINGS_FIELDS) / sizeof(OSQP_SETTINGS_FIELDS[0]);
  mxArray* mxPtr = mxCreateStructMatrix(1,1,nfields,OSQP_SETTINGS_FIELDS);

  //map the OSQP_SETTINGS fields one at a time into mxArrays
  //matlab handles everything as a double
  mxSetField(mxPtr, 0, "rho",             mxCreateDoubleScalar(settings->rho));
  mxSetField(mxPtr, 0, "sigma",             mxCreateDoubleScalar(settings->sigma));
  mxSetField(mxPtr, 0, "scaling",         mxCreateDoubleScalar(settings->scaling));
  mxSetField(mxPtr, 0, "scaling_norm",    mxCreateDoubleScalar(settings->scaling_norm));
  mxSetField(mxPtr, 0, "scaling_iter",    mxCreateDoubleScalar(settings->scaling_iter));
  mxSetField(mxPtr, 0, "max_iter",        mxCreateDoubleScalar(settings->max_iter));
  mxSetField(mxPtr, 0, "eps_abs",         mxCreateDoubleScalar(settings->eps_abs));
  mxSetField(mxPtr, 0, "eps_rel",         mxCreateDoubleScalar(settings->eps_rel));
  mxSetField(mxPtr, 0, "eps_prim_inf",         mxCreateDoubleScalar(settings->eps_prim_inf));
  mxSetField(mxPtr, 0, "eps_dual_inf",         mxCreateDoubleScalar(settings->eps_dual_inf));
  mxSetField(mxPtr, 0, "alpha",           mxCreateDoubleScalar(settings->alpha));
  mxSetField(mxPtr, 0, "delta",           mxCreateDoubleScalar(settings->delta));
  mxSetField(mxPtr, 0, "polish",       mxCreateDoubleScalar(settings->polish));
  mxSetField(mxPtr, 0, "pol_refine_iter", mxCreateDoubleScalar(settings->pol_refine_iter));
  mxSetField(mxPtr, 0, "verbose",         mxCreateDoubleScalar(settings->verbose));
  mxSetField(mxPtr, 0, "early_terminate", mxCreateDoubleScalar(settings->early_terminate));
  mxSetField(mxPtr, 0, "early_terminate_interval", mxCreateDoubleScalar(settings->early_terminate_interval));
  mxSetField(mxPtr, 0, "warm_start",      mxCreateDoubleScalar(settings->warm_start));

  return mxPtr;
}

void copyMxStructToSettings(const mxArray* mxPtr, OSQPSettings* settings){

  //this function assumes that only a complete and validated structure
  //will be passed.  matlab mex-side function is responsible for checking
  //structure validity

  //map the OSQP_SETTINGS fields one at a time into mxArrays
  //matlab handles everything as a double
  settings->rho             = (c_float)mxGetScalar(mxGetField(mxPtr, 0, "rho"));
  settings->sigma             = (c_float)mxGetScalar(mxGetField(mxPtr, 0, "sigma"));
  settings->scaling         = (c_int)mxGetScalar(mxGetField(mxPtr, 0, "scaling"));
  settings->scaling_norm    = (c_int)mxGetScalar(mxGetField(mxPtr, 0, "scaling_norm"));
  settings->scaling_iter    = (c_int)mxGetScalar(mxGetField(mxPtr, 0, "scaling_iter"));
  settings->max_iter        = (c_int)mxGetScalar(mxGetField(mxPtr, 0, "max_iter"));
  settings->eps_abs         = (c_float)mxGetScalar(mxGetField(mxPtr, 0, "eps_abs"));
  settings->eps_rel         = (c_float)mxGetScalar(mxGetField(mxPtr, 0, "eps_rel"));
  settings->eps_prim_inf        = (c_float)mxGetScalar(mxGetField(mxPtr, 0, "eps_dual_inf"));
  settings->eps_dual_inf         = (c_float)mxGetScalar(mxGetField(mxPtr, 0, "eps_dual_inf"));
  settings->alpha           = (c_float)mxGetScalar(mxGetField(mxPtr, 0, "alpha"));
  settings->delta           = (c_float)mxGetScalar(mxGetField(mxPtr, 0, "delta"));
  settings->polish       = (c_int)mxGetScalar(mxGetField(mxPtr, 0, "polish"));
  settings->pol_refine_iter = (c_int)mxGetScalar(mxGetField(mxPtr, 0, "pol_refine_iter"));
  settings->verbose         = (c_int)mxGetScalar(mxGetField(mxPtr, 0, "verbose"));
  settings->early_terminate = (c_int)mxGetScalar(mxGetField(mxPtr, 0, "early_terminate"));
  settings->early_terminate_interval = (c_int)mxGetScalar(mxGetField(mxPtr, 0, "early_terminate_interval"));
  settings->warm_start      = (c_int)mxGetScalar(mxGetField(mxPtr, 0, "warm_start"));

}

void copyUpdatedSettingsToWork(const mxArray* mxPtr ,OsqpData* osqpData){

  //This does basically the same job as copyMxStructToSettings which was used
  //during setup, but uses the provided update functions in osqp.h to update
  //settings in the osqp workspace.  Protects against bad parameter writes
  //or future modifications to updated settings handling

  osqp_update_max_iter(osqpData->work,
    (c_int)mxGetScalar(mxGetField(mxPtr, 0, "max_iter")));
  osqp_update_eps_abs(osqpData->work,
    (c_float)mxGetScalar(mxGetField(mxPtr, 0, "eps_abs")));
  osqp_update_eps_rel(osqpData->work,
    (c_float)mxGetScalar(mxGetField(mxPtr, 0, "eps_rel")));
  osqp_update_eps_prim_inf(osqpData->work,
    (c_float)mxGetScalar(mxGetField(mxPtr, 0, "eps_prim_inf")));
  osqp_update_eps_dual_inf(osqpData->work,
    (c_float)mxGetScalar(mxGetField(mxPtr, 0, "eps_dual_inf")));
  osqp_update_alpha(osqpData->work,
    (c_float)mxGetScalar(mxGetField(mxPtr, 0, "alpha")));
  osqp_update_delta(osqpData->work,
    (c_float)mxGetScalar(mxGetField(mxPtr, 0, "delta")));
  osqp_update_polish(osqpData->work,
    (c_int)mxGetScalar(mxGetField(mxPtr, 0, "polish")));
  osqp_update_pol_refine_iter(osqpData->work,
    (c_int)mxGetScalar(mxGetField(mxPtr, 0, "pol_refine_iter")));
  osqp_update_verbose(osqpData->work,
    (c_int)mxGetScalar(mxGetField(mxPtr, 0, "verbose")));
  osqp_update_early_terminate(osqpData->work,
    (c_int)mxGetScalar(mxGetField(mxPtr, 0, "early_terminate")));
  osqp_update_early_terminate_interval(osqpData->work,
      (c_int)mxGetScalar(mxGetField(mxPtr, 0, "early_terminate_interval")));
  osqp_update_warm_start(osqpData->work,
    (c_int)mxGetScalar(mxGetField(mxPtr, 0, "warm_start")));
}
