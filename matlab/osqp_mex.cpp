#include "mex.h"
#include "matrix.h"
#include "osqp_mex.hpp"
extern "C" {
    #include "osqp.h"
  }

// wrapper class for all osqp data and settings
class OsqpData
{
public:
  Settings * settings = (Settings *)c_malloc(sizeof(Settings)); //Settings
  Data     * data     = (Data *)c_malloc(sizeof(Data));         // Data
  Work     * work;                                              // Workspace
  int      hasProblemData = 0;
};

// internal helper functions
csc*      mxCopySparseMatrix(const mxArray*);
c_float*  mxCopyDenseVector(const mxArray*);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Get the command string
    char cmd[64];
	if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
		mexErrMsgTxt("First input should be a command string less than 64 characters long.");

    // New
    if (!strcmp("new", cmd)) {
        // Check parameters
        if (nlhs != 1){
            mexErrMsgTxt("New: One output expected.");
          }
        // Return a handle to a new C++ wrapper instance
        plhs[0] = convertPtr2Mat<OsqpData>(new OsqpData);
        return;
    }

    // Check there is a second input, which should be the class instance handle
    if (nrhs < 2)
		mexErrMsgTxt("Second input should be a class instance handle.");

    // Delete
    if (!strcmp("delete", cmd)) {
        // Destroy the C++ object
        destroyObject<OsqpData>(prhs[1]);
        // Warn if other commands were ignored
        if (nlhs != 0 || nrhs != 2)
            mexWarnMsgTxt("Delete: Unexpected arguments ignored.");
        return;
    }

    // Get the class instance pointer from the second input
    OsqpData* osqpData = convertMat2Ptr<OsqpData>(prhs[1]);

    // Call the various class methods
    // SETUP
    if (!strcmp("setup", cmd)) {

        // handle the problem data first.  Matlab-side
        // class wrapper is responsible for ensuring that
        // P and A are sparse matrices,  everything
        // else is a dense vector and all inputs are
        // of compatible dimension
        const mxArray* P  = prhs[2];
        const mxArray* q  = prhs[3];
        const mxArray* A  = prhs[4];
        const mxArray* lA = prhs[5];
        const mxArray* uA = prhs[6];

        mexPrintf("Configuring Data\n");
        osqpData->data->m  = mxGetM(A);
        osqpData->data->n  = mxGetN(A);
        osqpData->data->q  = mxCopyDenseVector(q);
        osqpData->data->lA = mxCopyDenseVector(lA);
        osqpData->data->uA = mxCopyDenseVector(uA);
        osqpData->data->P  = mxCopySparseMatrix(P);
        osqpData->data->A  = mxCopySparseMatrix(A);

        //DEBUG : validate the data
        mexPrintf("Checking Data\n");
        mexPrintf("Data validation check %i\n",validate_data(osqpData->data));
        mexPrintf("Done Checking Data\n");

        // Define Solver settings as default
        mexPrintf("DEBUG: configuring default settings\n");
        set_default_settings(osqpData->settings);

        // Setup workspace
        osqpData->work = osqp_setup(osqpData->data, osqpData->settings);
        if(!osqpData->work){
           mexErrMsgTxt("Invalid problem data.");
         }

        // Problem data has been populated
        osqpData->hasProblemData = true;
        mexPrintf("DEBUG: has Problem Data ? : %i\n",osqpData->hasProblemData);

        return;

    }
    // SOLVE
    if (!strcmp("solve", cmd)) {
        if(!osqpData->work || !osqpData->hasProblemData){
            mexErrMsgTxt("No problem data has been given.");
        }
        // solve the problem
        osqp_solve(osqpData->work);
        return;
    }

    // Got here, so command not recognized
    mexErrMsgTxt("Command not recognized.");
}


csc*  mxCopySparseMatrix(const mxArray* mxPtr){

  mwIndex* mxIdx;      //mathworks type for sparse matrix index arrays
  double*  mxValues;   //mathworks type for sparse matrix value arrays
  csc* out = csc_spalloc(mxGetM(mxPtr), mxGetN(mxPtr), mxGetNzmax(mxPtr), 1, 1);

  mexPrintf("csc_spalloc result : %p\n",(void*)out);
  mexPrintf("csc_spalloc nzmax %i\n",mxGetNzmax(mxPtr));

  // Copy column indices in compressed column format
  mxIdx      = mxGetJc(mxPtr); // column indices, size nzmax starting from 0
  for(int i=0; i < (out->n+1); i++){
      out->p[i] = mxIdx[i];
  }

  // Copy row indices in standard sparse index format
  mxIdx      = mxGetIr(mxPtr); // row indices, size nzmax starting from 0
  for(int i=0; i < out->nzmax; i++){
      out->i[i] = mxIdx[i];
  }

  // Copy sparse data values

  out->x       = (c_float*)c_malloc(out->nzmax * sizeof(c_float));
  mxValues     = mxGetPr(mxPtr);
  for(int i=0; i < out->nzmax; i++){
      out->x[i] = mxValues[i];
  }

  return out;
}


c_float*  mxCopyDenseVector(const mxArray* mxPtr){

  int numel       = mxGetNumberOfElements(mxPtr);
  c_float* out    = (c_float*)c_malloc(numel * sizeof(c_float));
  double*  mxData = mxGetPr(mxPtr);

  //copy data
  for(int i=0; i < numel; i++){
      out[i] = mxData[i];
  }
  return out;

}
