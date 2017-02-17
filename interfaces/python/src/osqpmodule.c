// Use not deprecated Numpy API (numpy > 1.7)
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"                // Python API
#include "structmember.h"          // Python members structure (to store results)
#include "numpy/arrayobject.h"     // Numpy C API
#include "numpy/npy_math.h"        // For infinity values
#include "osqp.h"                  // OSQP API


/* The PyInt variable is a PyLong in Python3.x.
 */
#if PY_MAJOR_VERSION >= 3
#define PyInt_AsLong PyLong_AsLong
#define PyInt_Check PyLong_Check
#endif


/****************************************
 * Utilities for Main API Functions     *
 ****************************************/


/* OSQP Problem data in Python arrays */
typedef struct {
    c_int          n;
    c_int          m;
    PyArrayObject *Px;
    PyArrayObject *Pi;
    PyArrayObject *Pp;
    PyArrayObject *q;
    PyArrayObject *Ax;
    PyArrayObject *Ai;
    PyArrayObject *Ap;
    PyArrayObject *l;
    PyArrayObject *u;
} PyData;

// Get integer type from OSQP setup
static int get_int_type(void) {
    switch (sizeof(c_int)) {
    case 1:
        return NPY_INT8;
    case 2:
        return NPY_INT16;
    case 4:
        return NPY_INT32;
    case 8:
        return NPY_INT64;
    default:
        return NPY_INT32; /* defaults to 4 byte int */
    }
}

// Get float type from OSQP setup
static int get_float_type(void) {
    switch (sizeof(c_float)) {
    case 2:
        return NPY_FLOAT16;
    case 4:
        return NPY_FLOAT32;
    case 8:
        return NPY_FLOAT64;
    default:
        return NPY_FLOAT64; /* defaults to double */
    }
}

/* gets the pointer to the block of contiguous C memory
 * the overhead should be small unless the numpy array has been
 * reordered in some way or the data type doesn't quite match
 */
static PyArrayObject *get_contiguous(PyArrayObject *array, int typenum) {
        /*
        * the "tmp_arr" pointer has to have Py_DECREF called on it; new_owner
        * owns the "new" array object created by PyArray_Cast
        */
        PyArrayObject *tmp_arr;
        PyArrayObject *new_owner;
        tmp_arr = PyArray_GETCONTIGUOUS(array);
        new_owner = (PyArrayObject *) PyArray_Cast(tmp_arr, typenum);
        Py_DECREF(tmp_arr);
        return new_owner;
}


static PyData * create_pydata(c_int n, c_int m,
                     PyArrayObject *Px, PyArrayObject *Pi, PyArrayObject *Pp,
                     PyArrayObject *q, PyArrayObject *Ax, PyArrayObject *Ai,
                     PyArrayObject *Ap, PyArrayObject *l, PyArrayObject *u){

        // Get int and float types
        int int_type = get_int_type();
        int float_type = get_float_type();

        // Populate PyData structure
        PyData * py_d = (PyData *)c_malloc(sizeof(PyData));
        py_d->n = n;
        py_d->m = m;
        py_d->Px = get_contiguous(Px, float_type);
        py_d->Pi = get_contiguous(Pi, int_type);
        py_d->Pp = get_contiguous(Pp, int_type);
        py_d->q  = get_contiguous(q, float_type);
        py_d->Ax = get_contiguous(Ax, float_type);
        py_d->Ai = get_contiguous(Ai, int_type);
        py_d->Ap = get_contiguous(Ap, int_type);
        py_d->l = get_contiguous(l, float_type);
        py_d->u = get_contiguous(u, float_type);

        // Retrun
        return py_d;

}

// Create data structure from arrays
static Data * create_data(PyData * py_d){

        // Allocate Data structure
        Data * data = (Data *)c_malloc(sizeof(Data));

        // Populate Data structure
        data->n = py_d->n;
        data->m = py_d->m;
        data->P = csc_matrix(data->n, data->n,
                             PyArray_DIM(py_d->Px, 0),  // nnz
                             (c_float *)PyArray_DATA(py_d->Px),
                             (c_int *)PyArray_DATA(py_d->Pi),
                             (c_int *)PyArray_DATA(py_d->Pp));
        data->q = (c_float *)PyArray_DATA(py_d->q);
        data->A = csc_matrix(data->m, data->n,
                             PyArray_DIM(py_d->Ax, 0),  // nnz
                             (c_float *)PyArray_DATA(py_d->Ax),
                             (c_int *)PyArray_DATA(py_d->Ai),
                             (c_int *)PyArray_DATA(py_d->Ap));
        data->l = (c_float *)PyArray_DATA(py_d->l);
        data->u = (c_float *)PyArray_DATA(py_d->u);

        return data;
}


static c_int free_data(Data *data, PyData * py_d){

    // Clean contiguous PyArrayObjects
    Py_DECREF(py_d->Px);
    Py_DECREF(py_d->Pi);
    Py_DECREF(py_d->Pp);
    Py_DECREF(py_d->q);
    Py_DECREF(py_d->Ax);
    Py_DECREF(py_d->Ai);
    Py_DECREF(py_d->Ap);
    Py_DECREF(py_d->l);
    Py_DECREF(py_d->u);
    c_free(py_d);

    // Clean data structure
    if (data){
        if (data->P){
            c_free(data->P);
        }

        if (data->A){
            c_free(data->A);
        }

        c_free(data);
        return 0;
    }
    else{
        return 1;
    }

}

/*******************************************
 * INFO Object definition and methods   *
 *******************************************/

 typedef struct {
    PyObject_HEAD
    c_int iter;                /* number of iterations taken */
    PyUnicodeObject * status;  /* status unicode string, e.g. 'Solved' */
    c_int status_val;          /* status as c_int, defined in constants.h */
    c_int status_polish;       /* polish status: successful (1), not (0) */
    c_float obj_val;           /* primal objective */
    c_float pri_res;           /* norm of primal residual */
    c_float dua_res;           /* norm of dual residual */

    #ifdef PROFILING
    c_float setup_time;        /* time taken for setup phase (milliseconds) */
    c_float solve_time;        /* time taken for solve phase (milliseconds) */
    c_float polish_time;       /* time taken for polish phase (milliseconds) */
    c_float run_time;          /* total time taken (milliseconds) */
    #endif

} OSQP_info;


static PyMemberDef OSQP_info_members[] = {
    {"iter", T_INT, offsetof(OSQP_info, iter), READONLY, "Primal solution"},
    {"status", T_OBJECT, offsetof(OSQP_info, status), READONLY, "Solver status"},
    {"status_val", T_INT, offsetof(OSQP_info, status_val), READONLY, "Solver status value"},
    {"status_polish", T_INT, offsetof(OSQP_info, status_polish), READONLY, "Polishing status value"},
    {"obj_val", T_DOUBLE, offsetof(OSQP_info, obj_val), READONLY, "Objective value"},
    {"pri_res", T_DOUBLE, offsetof(OSQP_info, pri_res), READONLY, "Primal residual"},
    {"dua_res", T_DOUBLE, offsetof(OSQP_info, dua_res), READONLY, "Dual residual"},
    #ifdef PROFILING
    {"setup_time", T_DOUBLE, offsetof(OSQP_info, setup_time), READONLY, "Setup time"},
    {"solve_time", T_DOUBLE, offsetof(OSQP_info, solve_time), READONLY, "Solve time"},
    {"polish_time", T_DOUBLE, offsetof(OSQP_info, polish_time), READONLY, "Polish time"},
    {"run_time", T_DOUBLE, offsetof(OSQP_info, run_time), READONLY, "Total run time"},
    #endif
    {NULL}
};


// Initialize results structure assigning arguments
static c_int OSQP_info_init( OSQP_info * self, PyObject *args)
{
    #ifdef PROFILING

    #ifdef DLONG

    #ifdef DFLOAT
    static char * argparse_string = "lUllfffffff";
    #else
    static char * argparse_string = "lUllddddddd";
    #endif

    #else

    #ifdef DFLOAT
    static char * argparse_string = "iUiifffffff";
    #else
    static char * argparse_string = "iUiiddddddd";
    #endif

    #endif
    // Parse argumentrs
    if( !PyArg_ParseTuple(args, argparse_string,
                          &(self->iter),
                          &(self->status),
                          &(self->status_val),
                          &(self->status_polish),
                          &(self->obj_val),
                          &(self->pri_res),
                          &(self->dua_res),
                          &(self->setup_time),
                          &(self->solve_time),
                          &(self->polish_time),
                          &(self->run_time))) {
            return -1;
    }
    #else

    #ifdef DLONG

    #ifdef DFLOAT
    static char * argparse_string = "lUllfff";
    #else
    static char * argparse_string = "lUllddd";
    #endif

    #else

    #ifdef DFLOAT
    static char * argparse_string = "iUiifff";
    #else
    static char * argparse_string = "iUiiddd";
    #endif

    #endif

    // Parse argumentrs
    if( !PyArg_ParseTuple(args, argparse_string,
                          &(self->iter),
                          &(self->status),
                          &(self->status_val),
                          &(self->status_polish),
                          &(self->obj_val),
                          &(self->pri_res),
                          &(self->dua_res))) {
            return -1;
    }

    #endif


	return 0;
}


static c_int OSQP_info_dealloc(OSQP_info *self){

    // Delete Python string status
    Py_DECREF(self->status);

    // Deallocate object
    PyObject_Del(self);

    return 0;
}


// Define info type object
static PyTypeObject OSQP_info_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "osqp.OSQP_info",                       /* tp_name*/
    sizeof(OSQP_info),                      /* tp_basicsize*/
    0,                                         /* tp_itemsize*/
    (destructor)OSQP_info_dealloc,          /* tp_dealloc*/
    0,                                         /* tp_print*/
    0,                                         /* tp_getattr*/
    0,                                         /* tp_setattr*/
    0,                                         /* tp_compare*/
    0,                                         /* tp_repr*/
    0,                                         /* tp_as_number*/
    0,                                         /* tp_as_sequence*/
    0,                                         /* tp_as_mapping*/
    0,                                         /* tp_hash */
    0,                                         /* tp_call*/
    0,                                         /* tp_str*/
    0,                                         /* tp_getattro*/
    0,                                         /* tp_setattro*/
    0,                                         /* tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,                        /* tp_flags*/
    "OSQP solver info",                     /* tp_doc */
    0,		                                   /* tp_traverse */
    0,		                                   /* tp_clear */
    0,		                                   /* tp_richcompare */
    0,		                                   /* tp_weaklistoffset */
    0,		                                   /* tp_iter */
    0,		                                   /* tp_iternext */
    0,                                         /* tp_methods */
    OSQP_info_members,                      /* tp_members */
    0,                                         /* tp_getset */
    0,                                         /* tp_base */
    0,                                         /* tp_dict */
    0,                                         /* tp_descr_get */
    0,                                         /* tp_descr_set */
    0,                                         /* tp_dictoffset */
    (initproc)OSQP_info_init,               /* tp_init */
    0,                                         /* tp_alloc */
    0,                                         /* tp_new */
};

/*******************************************
 * RESULTS Object definition and methods   *
 *******************************************/

 typedef struct {
    PyObject_HEAD
    PyArrayObject * x;     // Primal solution
    PyArrayObject * y;     // Dual solution
    OSQP_info * info;      // Solver information
} OSQP_results;

static PyMemberDef OSQP_results_members[] = {
    {"x", T_OBJECT, offsetof(OSQP_results, x), READONLY, "Primal solution"},
    {"y", T_OBJECT, offsetof(OSQP_results, y), READONLY, "Dual solution"},
    {"info", T_OBJECT, offsetof(OSQP_results, info), READONLY, "Solver Information"},
    {NULL}
};

// Initialize results structure assigning arguments
static c_int OSQP_results_init( OSQP_results * self, PyObject *args)
{
    static char * argparse_string = "O!O!O!";

    // Parse argumentrs
    if( !PyArg_ParseTuple(args, argparse_string,
                          &PyArray_Type, &(self->x),
                          &PyArray_Type, &(self->y),
                          &OSQP_info_Type, &(self->info))) {
            return -1;
    }

	return 0;
}


static c_int OSQP_results_dealloc(OSQP_results *self){

    // Delete Python arrays
    Py_DECREF(self->x);
    Py_DECREF(self->y);

    // Delete info object
    Py_DECREF(self->info);

    // Deallocate object
    PyObject_Del(self);

    return 0;
}


// Define results type object
static PyTypeObject OSQP_results_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "osqp.OSQP_results",                       /* tp_name*/
    sizeof(OSQP_results),                      /* tp_basicsize*/
    0,                                         /* tp_itemsize*/
    (destructor)OSQP_results_dealloc,          /* tp_dealloc*/
    0,                                         /* tp_print*/
    0,                                         /* tp_getattr*/
    0,                                         /* tp_setattr*/
    0,                                         /* tp_compare*/
    0,                                         /* tp_repr*/
    0,                                         /* tp_as_number*/
    0,                                         /* tp_as_sequence*/
    0,                                         /* tp_as_mapping*/
    0,                                         /* tp_hash */
    0,                                         /* tp_call*/
    0,                                         /* tp_str*/
    0,                                         /* tp_getattro*/
    0,                                         /* tp_setattro*/
    0,                                         /* tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,                        /* tp_flags*/
    "OSQP solver results",                     /* tp_doc */
    0,		                                   /* tp_traverse */
    0,		                                   /* tp_clear */
    0,		                                   /* tp_richcompare */
    0,		                                   /* tp_weaklistoffset */
    0,		                                   /* tp_iter */
    0,		                                   /* tp_iternext */
    0,                                         /* tp_methods */
    OSQP_results_members,                      /* tp_members */
    0,                                         /* tp_getset */
    0,                                         /* tp_base */
    0,                                         /* tp_dict */
    0,                                         /* tp_descr_get */
    0,                                         /* tp_descr_set */
    0,                                         /* tp_dictoffset */
    (initproc)OSQP_results_init,               /* tp_init */
    0,                                         /* tp_alloc */
    0,                                         /* tp_new */
};




/****************************************
 * OSQP Object definition and methods   *
 ****************************************/

typedef struct {
    PyObject_HEAD
    Work * workspace;  // Pointer to C workspace structure
} OSQP;

static PyTypeObject OSQP_Type;


/* Create new OSQP Object */
static c_int OSQP_init( OSQP * self, PyObject *args, PyObject *kwds)
{
	// OSQP *self;
	// self = PyObject_New(OSQP, &OSQP_Type);
	if (self == NULL)
		return -1;
	self->workspace = NULL;
	// return self;
	return 0;
}



// Deallocate OSQP object
static c_int OSQP_dealloc(OSQP* self)
{
    // Cleanup workspace if not null
    if (self->workspace)
        osqp_cleanup(self->workspace);

    // Cleanup python object
    PyObject_Del(self);

    return 0;
}

// Solve Optimization Problem
static PyObject * OSQP_solve(OSQP *self)
{
    if (self->workspace){
        // Get int and float types
        // int int_type = get_int_type();
        int float_type = get_float_type();

        // Create status object
        PyObject * status;

        // Create solution objects
        PyObject * x, *y;

        // Define info related variables
        static char *argparse_string;
        PyObject *info_list;
        PyObject *info;

        // Results
        PyObject *results_list;
        PyObject *results;

        // Temporary solution
        c_float *x_arr, *y_arr; // Primal dual solutions
        npy_intp nd[] = {(npy_intp)self->workspace->data->n};  // Dimensions in R^n
        npy_intp md[] = {(npy_intp)self->workspace->data->m};  // Dimensions in R^m

        /**
         *  Solve QP Problem
         */
        osqp_solve(self->workspace);

        // If solution is not Infeasible or Unbounded store it
        if ((self->workspace->info->status_val != OSQP_INFEASIBLE) &&
            (self->workspace->info->status_val != OSQP_UNBOUNDED)){
            // Store solution into temporary arrays
            // N.B. Needed to be able to store RESULTS even when OSQP structure is deleted
            x_arr = vec_copy(self->workspace->solution->x, self->workspace->data->n);
            y_arr = vec_copy(self->workspace->solution->y, self->workspace->data->m);


            // Get primal dual solution PyArrayObjects
            x = PyArray_SimpleNewFromData(1, nd, float_type, x_arr);
            // Set x to own x_arr so that it is freed when x is freed
            PyArray_ENABLEFLAGS((PyArrayObject *) x, NPY_ARRAY_OWNDATA);

            y = PyArray_SimpleNewFromData(1, md, float_type, y_arr);
            // Set y to own y_arr so that it is freed when y is freed
            PyArray_ENABLEFLAGS((PyArrayObject *) y, NPY_ARRAY_OWNDATA);
        } else { // Problem infeasible or unbounded -> None values for x,y
            x = PyArray_EMPTY(1, nd, NPY_OBJECT, 0);
            y = PyArray_EMPTY(1, nd, NPY_OBJECT, 0);
        }

        // If problem infeasible, set objective value to numpy infinity
        if (self->workspace->info->status_val == OSQP_INFEASIBLE){
            self->workspace->info->obj_val = NPY_INFINITY;
        }

        // If problem unbounded, set objective value to numpy -infinity
        if (self->workspace->info->status_val == OSQP_UNBOUNDED){
            self->workspace->info->obj_val = -NPY_INFINITY;
        }


        /*  CREATE INFO OBJECT */
        // Store status string
        status = PyUnicode_FromString(self->workspace->info->status);

        // Create info_list
        #ifdef PROFILING
        #ifdef DLONG

        #ifdef DFLOAT
        argparse_string = "lOllfffffff";
        #else
        argparse_string = "lOllddddddd";
        #endif

        #else

        #ifdef DFLOAT
        argparse_string = "iOiifffffff";
        #else
        argparse_string = "iOiiddddddd";
        #endif

        #endif

        info_list = Py_BuildValue(argparse_string,
                                  self->workspace->info->iter,
                                  status,
                                  self->workspace->info->status_val,
                                  self->workspace->info->status_polish,
                                  self->workspace->info->obj_val,
                                  self->workspace->info->pri_res,
                                  self->workspace->info->dua_res,
                                  self->workspace->info->setup_time,
                                  self->workspace->info->solve_time,
                                  self->workspace->info->polish_time,
                                  self->workspace->info->run_time);
        #else

        #ifdef DLONG

        #ifdef DFLOAT
        argparse_string = "lOllfff";
        #else
        argparse_string = "lOllddd";
        #endif

        #else

        #ifdef DFLOAT
        argparse_string = "iOiifff";
        #else
        argparse_string = "iOiiddd";
        #endif

        #endif

        info_list = Py_BuildValue(argparse_string,
                                            self->workspace->info->iter,
                                            status,
                                            self->workspace->info->status_val,
                                            self->workspace->info->status_polish,
                                            self->workspace->info->obj_val,
                                            self->workspace->info->pri_res,
                                            self->workspace->info->dua_res);
        #endif

        info = PyObject_CallObject((PyObject *) &OSQP_info_Type, info_list);

        /* Release the info argument list. */
        Py_DECREF(info_list);

        /*  CREATE RESULTS OBJECT */
        results_list = Py_BuildValue("OOO", x, y, info);

        // /* Call the class object. */
        results = PyObject_CallObject((PyObject *) &OSQP_results_Type, results_list);

        /* Release the argument list. */
        Py_DECREF(results_list);

    	// Py_INCREF(Py_None);
    	// return Py_None;
        return results;
        // return x;
    }
    else {
        PyErr_SetString(PyExc_ValueError, "Workspace not initialized!");
         return (PyObject *) NULL;
    }
}


// Setup optimization problem
static PyObject * OSQP_setup(OSQP *self, PyObject *args, PyObject *kwargs) {
        c_int n, m;  // Problem dimensions
        PyArrayObject *Px, *Pi, *Pp, *q, *Ax, *Ai, *Ap, *l, *u;
        static char *kwlist[] = {"dims",                          // nvars and ncons
                                 "Px", "Pi", "Pp", "q",           // Cost function
                                 "Ax", "Ai", "Ap", "l", "u",      // Constraints
                                 "scaling", "scaling_norm", "scaling_iter",
                                 "rho", "sigma", "max_iter",
                                 "eps_abs", "eps_rel", "eps_inf", "eps_unb", "alpha",
                                 "delta", "polishing", "pol_refine_iter", "verbose",
                                 "warm_start", NULL};               // Settings


        #ifdef DLONG

        #ifdef DFLOAT
        static char * argparse_string = "(ll)O!O!O!O!O!O!O!O!O!|lllfflffffffllll";
        #else
        static char * argparse_string = "(ll)O!O!O!O!O!O!O!O!O!|lllddlddddddllll";
        #endif

        #else

        #ifdef DFLOAT
        static char * argparse_string = "(ii)O!O!O!O!O!O!O!O!O!|iiiffiffffffiiii";
        #else
        static char * argparse_string = "(ii)O!O!O!O!O!O!O!O!O!|iiiddiddddddiiii";
        #endif

        #endif

        // Data and settings
        PyData *pydata;
        Data * data;
        Settings * settings = (Settings *)c_malloc(sizeof(Settings));
        set_default_settings(settings);

        if( !PyArg_ParseTupleAndKeywords(args, kwargs, argparse_string, kwlist,
                                         &n, &m,
                                         &PyArray_Type, &Px,
                                         &PyArray_Type, &Pi,
                                         &PyArray_Type, &Pp,
                                         &PyArray_Type, &q,
                                         &PyArray_Type, &Ax,
                                         &PyArray_Type, &Ai,
                                         &PyArray_Type, &Ap,
                                         &PyArray_Type, &l,
                                         &PyArray_Type, &u,
                                         &settings->scaling,
                                         &settings->scaling_norm,
                                         &settings->scaling_iter,
                                         &settings->rho,
                                         &settings->sigma,
                                         &settings->max_iter,
                                         &settings->eps_abs,
                                         &settings->eps_rel,
                                         &settings->eps_inf,
                                         &settings->eps_unb,
                                         &settings->alpha,
                                         &settings->delta,
                                         &settings->polishing,
                                         &settings->pol_refine_iter,
                                         &settings->verbose,
                                         &settings->warm_start)) {
                return NULL;
        }

        // Create Data from parsed vectors
        pydata = create_pydata(n, m, Px, Pi, Pp, q, Ax, Ai, Ap, l, u);
        data = create_data(pydata);

        // Create Workspace object
        self->workspace = osqp_setup(data, settings);

        // Cleanup data and settings
        free_data(data, pydata);
        c_free(settings);

        if (self->workspace){ // Workspace allocation correct
            // Return workspace
            Py_INCREF(Py_None);
        	return Py_None;
        }
        else{
            PyErr_SetString(PyExc_ValueError, "Workspace allocation error!");
            return (PyObject *) NULL;
        }
}

static PyObject *OSQP_version(OSQP *self) {
    return Py_BuildValue("s", osqp_version());
}


static PyObject *OSQP_dimensions(OSQP *self){
    #ifdef DLONG
    return Py_BuildValue("ll", self->workspace->data->n, self->workspace->data->m);
    #else
    return Py_BuildValue("ii", self->workspace->data->n, self->workspace->data->m);
    #endif
}


static PyObject *OSQP_constant(OSQP *self, PyObject *args) {


    char * constant_name;  // String less than 32 chars

    // Parse argumentrs
    if( !PyArg_ParseTuple(args, "s", &(constant_name))) {
            return NULL;
    }


    if(!strcmp(constant_name, "OSQP_INFTY")){
        #ifdef DFLOAT
        return Py_BuildValue("f", OSQP_INFTY);
        #else
        return Py_BuildValue("d", OSQP_INFTY);
        #endif
    }

    if(!strcmp(constant_name, "OSQP_NAN")){
        #ifdef DFLOAT
        return Py_BuildValue("f", OSQP_NAN);
        #else
        return Py_BuildValue("d", OSQP_NAN);
        #endif
    }

    if(!strcmp(constant_name, "OSQP_SOLVED")){
        return Py_BuildValue("i", OSQP_SOLVED);
    }

    if(!strcmp(constant_name, "OSQP_UNSOLVED")){
        return Py_BuildValue("i", OSQP_UNSOLVED);
    }

    if(!strcmp(constant_name, "OSQP_INFEASIBLE")){
        return Py_BuildValue("i", OSQP_INFEASIBLE);
    }

    if(!strcmp(constant_name, "OSQP_UNBOUNDED")){
        return Py_BuildValue("i", OSQP_UNBOUNDED);
    }

    if(!strcmp(constant_name, "OSQP_MAX_ITER_REACHED")){
        return Py_BuildValue("i", OSQP_MAX_ITER_REACHED);
    }


    // If reached here error
    PyErr_SetString(PyExc_ValueError, "Constant not recognized");
    return (PyObject *) NULL;

}




static PyObject *OSQP_update_lin_cost(OSQP *self, PyObject *args){
    PyArrayObject *q, *q_cont;
    c_float * q_arr;
    int float_type = get_float_type();

    static char * argparse_string = "O!";

    // Parse argumentrs
    if( !PyArg_ParseTuple(args, argparse_string,
                          &PyArray_Type, &q)) {
            return NULL;
    }

    // Get contiguous data structure
    q_cont = get_contiguous(q, float_type);

    // Copy array into c_float array
    q_arr = (c_float *)PyArray_DATA(q_cont);

    // Update linear cost
    osqp_update_lin_cost(self->workspace, q_arr);

    // Free data
    Py_DECREF(q_cont);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}

static PyObject *OSQP_update_lower_bound(OSQP *self, PyObject *args){
    PyArrayObject *l, *l_cont;
    c_float * l_arr;
    int float_type = get_float_type();

    static char * argparse_string = "O!";

    // Parse argumentrs
    if( !PyArg_ParseTuple(args, argparse_string,
                          &PyArray_Type, &l)) {
            return NULL;
    }

    // Get contiguous data structure
    l_cont = get_contiguous(l, float_type);

    // Copy array into c_float array
    l_arr = (c_float *)PyArray_DATA(l_cont);

    // Update linear cost
    osqp_update_lower_bound(self->workspace, l_arr);

    // Free data
    Py_DECREF(l_cont);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}

static PyObject *OSQP_update_upper_bound(OSQP *self, PyObject *args){
    PyArrayObject *u, *u_cont;
    c_float * u_arr;
    int float_type = get_float_type();

    static char * argparse_string = "O!";

    // Parse argumentrs
    if( !PyArg_ParseTuple(args, argparse_string,
                          &PyArray_Type, &u)) {
            return NULL;
    }

    // Get contiguous data structure
    u_cont = get_contiguous(u, float_type);

    // Copy array into c_float array
    u_arr = (c_float *)PyArray_DATA(u_cont);

    // Update linear cost
    osqp_update_upper_bound(self->workspace, u_arr);

    // Free data
    Py_DECREF(u_cont);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}


static PyObject *OSQP_update_bounds(OSQP *self, PyObject *args){
    PyArrayObject *l, *l_cont, *u, *u_cont;
    c_float * l_arr, * u_arr;
    int float_type = get_float_type();

    static char * argparse_string = "O!O!";

    // Parse argumentrs
    if( !PyArg_ParseTuple(args, argparse_string,
                          &PyArray_Type, &l,
                          &PyArray_Type, &u)) {
            return NULL;
    }

    // Get contiguous data structure
    l_cont = get_contiguous(l, float_type);
    u_cont = get_contiguous(u, float_type);

    // Copy array into c_float array
    l_arr = (c_float *)PyArray_DATA(l_cont);
    u_arr = (c_float *)PyArray_DATA(u_cont);

    // Update linear cost
    osqp_update_bounds(self->workspace, l_arr, u_arr);

    // Free data
    Py_DECREF(l_cont);
    Py_DECREF(u_cont);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}

static PyObject *OSQP_warm_start(OSQP *self, PyObject *args){
    PyArrayObject *x, *x_cont, *y, *y_cont;
    c_float * x_arr, * y_arr;
    int float_type = get_float_type();

    static char * argparse_string = "O!O!";

    // Parse argumentrs
    if( !PyArg_ParseTuple(args, argparse_string,
                          &PyArray_Type, &x,
                          &PyArray_Type, &y)) {
            return NULL;
    }

    // Get contiguous data structure
    x_cont = get_contiguous(x, float_type);
    y_cont = get_contiguous(y, float_type);

    // Copy array into c_float array
    x_arr = (c_float *)PyArray_DATA(x_cont);
    y_arr = (c_float *)PyArray_DATA(y_cont);

    // Update linear cost
    osqp_warm_start(self->workspace, x_arr, y_arr);

    // Free data
    Py_DECREF(x_cont);
    Py_DECREF(y_cont);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}

static PyObject *OSQP_warm_start_x(OSQP *self, PyObject *args){
    PyArrayObject *x, *x_cont;
    c_float * x_arr;
    int float_type = get_float_type();

    static char * argparse_string = "O!";

    // Parse argumentrs
    if( !PyArg_ParseTuple(args, argparse_string,
                          &PyArray_Type, &x)) {
            return NULL;
    }

    // Get contiguous data structure
    x_cont = get_contiguous(x, float_type);

    // Copy array into c_float array
    x_arr = (c_float *)PyArray_DATA(x_cont);

    // Update linear cost
    osqp_warm_start_x(self->workspace, x_arr);

    // Free data
    Py_DECREF(x_cont);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}

static PyObject *OSQP_warm_start_y(OSQP *self, PyObject *args){
    PyArrayObject *y, *y_cont;
    c_float * y_arr;
    int float_type = get_float_type();

    static char * argparse_string = "O!";

    // Parse argumentrs
    if( !PyArg_ParseTuple(args, argparse_string,
                          &PyArray_Type, &y)) {
            return NULL;
    }

    // Get contiguous data structure
    y_cont = get_contiguous(y, float_type);

    // Copy array into c_float array
    y_arr = (c_float *)PyArray_DATA(y_cont);

    // Update linear cost
    osqp_warm_start_y(self->workspace, y_arr);

    // Free data
    Py_DECREF(y_cont);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}


static PyObject *OSQP_update_max_iter(OSQP *self, PyObject *args){
    c_int max_iter_new;

    #ifdef DLONG
    static char * argparse_string = "l";
    #else
    static char * argparse_string = "i";
    #endif
    // Parse argumentrs
    if( !PyArg_ParseTuple(args, argparse_string, &max_iter_new)) {
            return NULL;
    }


    // Perform Update
    osqp_update_max_iter(self->workspace, max_iter_new);


    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}


static PyObject *OSQP_update_eps_abs(OSQP *self, PyObject *args){
    c_float eps_abs_new;

    #ifdef DFLOAT
    static char * argparse_string = "f";
    #else
    static char * argparse_string = "d";
    #endif

    // Parse argumentrs
    if( !PyArg_ParseTuple(args, argparse_string, &eps_abs_new)) {
            return NULL;
    }

    // Perform Update
    osqp_update_eps_abs(self->workspace, eps_abs_new);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}

static PyObject *OSQP_update_eps_rel(OSQP *self, PyObject *args){
    c_float eps_rel_new;

    #ifdef DFLOAT
    static char * argparse_string = "f";
    #else
    static char * argparse_string = "d";
    #endif

    // Parse argumentrs
    if( !PyArg_ParseTuple(args, argparse_string, &eps_rel_new)) {
            return NULL;
    }

    // Perform Update
    osqp_update_eps_rel(self->workspace, eps_rel_new);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}

static PyObject *OSQP_update_alpha(OSQP *self, PyObject *args){
    c_float alpha_new;


    #ifdef DFLOAT
    static char * argparse_string = "f";
    #else
    static char * argparse_string = "d";
    #endif

    // Parse argumentrs
    if( !PyArg_ParseTuple(args, argparse_string, &alpha_new)) {
            return NULL;
    }

    // Perform Update
    osqp_update_alpha(self->workspace, alpha_new);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}


static PyObject *OSQP_update_delta(OSQP *self, PyObject *args){
    c_float delta_new;

    #ifdef DFLOAT
    static char * argparse_string = "f";
    #else
    static char * argparse_string = "d";
    #endif

    // Parse argumentrs
    if( !PyArg_ParseTuple(args, argparse_string, &delta_new)) {
            return NULL;
    }

    // Perform Update
    osqp_update_delta(self->workspace, delta_new);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}


static PyObject *OSQP_update_polishing(OSQP *self, PyObject *args){
    c_int polishing_new;

    #ifdef DLONG
    static char * argparse_string = "l";
    #else
    static char * argparse_string = "i";
    #endif

    // Parse argumentrs
    if( !PyArg_ParseTuple(args, argparse_string, &polishing_new)) {
            return NULL;
    }

    // Perform Update
    osqp_update_polishing(self->workspace, polishing_new);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}

static PyObject *OSQP_update_pol_refine_iter(OSQP *self, PyObject *args){
    c_int pol_refine_iter_new;

    #ifdef DLONG
    static char * argparse_string = "l";
    #else
    static char * argparse_string = "i";
    #endif

    // Parse argumentrs
    if( !PyArg_ParseTuple(args, argparse_string, &pol_refine_iter_new)) {
            return NULL;
    }

    // Perform Update
    osqp_update_pol_refine_iter(self->workspace, pol_refine_iter_new);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}

static PyObject *OSQP_update_verbose(OSQP *self, PyObject *args){
    c_int verbose_new;

    #ifdef DLONG
    static char * argparse_string = "l";
    #else
    static char * argparse_string = "i";
    #endif

    // Parse argumentrs
    if( !PyArg_ParseTuple(args, argparse_string, &verbose_new)) {
            return NULL;
    }

    // Perform Update
    osqp_update_verbose(self->workspace, verbose_new);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}

static PyObject *OSQP_update_warm_start(OSQP *self, PyObject *args){
    c_int warm_start_new;

    #ifdef DLONG
    static char * argparse_string = "l";
    #else
    static char * argparse_string = "i";
    #endif

    // Parse argumentrs
    if( !PyArg_ParseTuple(args, argparse_string, &warm_start_new)) {
            return NULL;
    }

    // Perform Update
    osqp_update_warm_start(self->workspace, warm_start_new);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}


static PyMethodDef OSQP_methods[] = {
    {"setup",	(PyCFunction)OSQP_setup,METH_VARARGS|METH_KEYWORDS, PyDoc_STR("Setup OSQP problem")},
	{"solve",	(PyCFunction)OSQP_solve, METH_VARARGS, PyDoc_STR("Solve OSQP problem")},
    {"version",	(PyCFunction)OSQP_version, METH_NOARGS, PyDoc_STR("OSQP version")},
    {"constant",	(PyCFunction)OSQP_constant, METH_VARARGS, PyDoc_STR("Return internal OSQP constant")},
    {"dimensions",	(PyCFunction)OSQP_dimensions, METH_NOARGS, PyDoc_STR("Return problem dimensions (n, m)")},
    {"update_lin_cost",	(PyCFunction)OSQP_update_lin_cost, METH_VARARGS, PyDoc_STR("Update OSQP problem linear cost")},
    {"update_lower_bound",	(PyCFunction)OSQP_update_lower_bound, METH_VARARGS, PyDoc_STR("Update OSQP problem lower bound")},
    {"update_upper_bound",	(PyCFunction)OSQP_update_upper_bound, METH_VARARGS, PyDoc_STR("Update OSQP problem upper bound")},
    {"update_bounds",	(PyCFunction)OSQP_update_bounds, METH_VARARGS, PyDoc_STR("Update OSQP problem bounds")},
    {"warm_start",	(PyCFunction)OSQP_warm_start, METH_VARARGS, PyDoc_STR("Warm start primal and dual variables")},
    {"warm_start_x",	(PyCFunction)OSQP_warm_start_x, METH_VARARGS, PyDoc_STR("Warm start primal variable")},
    {"warm_start_y",	(PyCFunction)OSQP_warm_start_y, METH_VARARGS, PyDoc_STR("Warm start dual variable")},
    {"update_max_iter",	(PyCFunction)OSQP_update_max_iter, METH_VARARGS, PyDoc_STR("Update OSQP solver setting max_iter")},
    {"update_eps_abs",	(PyCFunction)OSQP_update_eps_abs, METH_VARARGS, PyDoc_STR("Update OSQP solver setting eps_abs")},
    {"update_eps_rel",	(PyCFunction)OSQP_update_eps_rel, METH_VARARGS, PyDoc_STR("Update OSQP solver setting eps_rel")},
    {"update_alpha",	(PyCFunction)OSQP_update_alpha, METH_VARARGS, PyDoc_STR("Update OSQP solver setting alpha")},
    {"update_delta",	(PyCFunction)OSQP_update_delta, METH_VARARGS, PyDoc_STR("Update OSQP solver setting delta")},
    {"update_polishing",	(PyCFunction)OSQP_update_polishing, METH_VARARGS, PyDoc_STR("Update OSQP solver setting polishing")},
    {"update_pol_refine_iter",	(PyCFunction)OSQP_update_pol_refine_iter, METH_VARARGS, PyDoc_STR("Update OSQP solver setting pol_refine_iter")},
    {"update_verbose",	(PyCFunction)OSQP_update_verbose, METH_VARARGS, PyDoc_STR("Update OSQP solver setting verbose")},
    {"update_warm_start",	(PyCFunction)OSQP_update_warm_start, METH_VARARGS, PyDoc_STR("Update OSQP solver setting warm_start")},
    {NULL,		NULL}		/* sentinel */
};


// Define workspace type object
static PyTypeObject OSQP_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "osqp.OSQP",                               /*tp_name*/
    sizeof(OSQP),                              /*tp_basicsize*/
    0,                                         /*tp_itemsize*/
    (destructor)OSQP_dealloc,                  /*tp_dealloc*/
    0,                                         /*tp_print*/
    0,                                         /*tp_getattr*/
    0,                                         /*tp_setattr*/
    0,                                         /*tp_compare*/
    0,                                         /*tp_repr*/
    0,                                         /*tp_as_number*/
    0,                                         /*tp_as_sequence*/
    0,                                         /*tp_as_mapping*/
    0,                                         /*tp_hash */
    0,                                         /*tp_call*/
    0,                                         /*tp_str*/
    0,                                         /*tp_getattro*/
    0,                                         /*tp_setattro*/
    0,                                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,                        /*tp_flags*/
    "OSQP solver",                             /* tp_doc */
    0,		                                   /* tp_traverse */
    0,		                                   /* tp_clear */
    0,		                                   /* tp_richcompare */
    0,		                                   /* tp_weaklistoffset */
    0,		                                   /* tp_iter */
    0,		                                   /* tp_iternext */
    OSQP_methods,                              /* tp_methods */
    0,                                         /* tp_members */
    0,                                         /* tp_getset */
    0,                                         /* tp_base */
    0,                                         /* tp_dict */
    0,                                         /* tp_descr_get */
    0,                                         /* tp_descr_set */
    0,                                         /* tp_dictoffset */
    (initproc)OSQP_init,                       /* tp_init */
    0,                                         /* tp_alloc */
    0,                                         /* tp_new */
};






/************************
 * Interface Methods    *
 ************************/


 /* Module initialization for Python 3*/
 #if PY_MAJOR_VERSION >= 3
 static struct PyModuleDef moduledef = {
     PyModuleDef_HEAD_INIT, "_osqp",           /* m_name */
     NULL,         /* m_doc */
     -1,                                       /* m_size */
     OSQP_methods,                             /* m_methods */
     NULL,                                 /* m_reload */
     NULL,                                 /* m_traverse */
     NULL,                                 /* m_clear */
     NULL,                                 /* m_free */
 };
 #endif



static PyObject * moduleinit(void){

    PyObject *m;

    // Initialize module (no methods. all inside OSQP object)
    #if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
    #else
    m = Py_InitModule3("_osqp", NULL, NULL);
    #endif
    if (m == NULL)
        return NULL;

    // Initialize OSQP_Type
    OSQP_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&OSQP_Type) < 0)     // Initialize OSQP_Type
        return NULL;

    // Add type to the module dictionary and initialize it
    Py_INCREF(&OSQP_Type);
    if (PyModule_AddObject(m, "OSQP", (PyObject *)&OSQP_Type) < 0)
        return NULL;


    // Initialize Info Type
    OSQP_info_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&OSQP_info_Type) < 0)
        return NULL;

    // Initialize Results Type
    OSQP_results_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&OSQP_results_Type) < 0)
        return NULL;

    return m;
}




// Init Osqp Internal module
#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__osqp(void)
#else
PyMODINIT_FUNC init_osqp(void)
#endif
{

        import_array(); /* for numpy arrays */

        // Module initialization is not a global variable in
        // Python 3
        #if PY_MAJOR_VERSION >= 3
        return moduleinit();
        #else
        moduleinit();
        #endif
}
