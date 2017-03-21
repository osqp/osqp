// Use not deprecated Numpy API (numpy > 1.7)
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"                // Python API
// #include "structmember.h"          // Python members structure (to store results)
#include "numpy/arrayobject.h"     // Numpy C API
#include "numpy/npy_math.h"        // For infinity values
#include "osqp.h"                  // OSQP API


#include "workspace.h"             // Include code-generated OSQP workspace




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
#elif IS_MAC

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
#if IS_WINDOWS

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
#elif IS_MAC

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




/* The PyInt variable is a PyLong in Python3.x.
 */
#if PY_MAJOR_VERSION >= 3
#define PyInt_AsLong PyLong_AsLong
#define PyInt_Check PyLong_Check
#endif


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


/************************
* Interface Methods    *
************************/


// Solve Optimization Problem
static PyObject * OSQP_solve(PyObject *self, PyObject *args)
{
    // Allocate timer
    PyTimer * timer;
    c_float solve_time;

    // Get float types
    int float_type = get_float_type();

    // Create solution objects
    PyObject * x, *y;

    // Temporary solution
    c_float *x_arr, *y_arr; // Primal dual solutions
    npy_intp nd[] = {(npy_intp)(&workspace)->data->n}; // Dimensions in R^n
    npy_intp md[] = {(npy_intp)(&workspace)->data->m}; // Dimensions in R^m


    // Allocate solutions
    x_arr = PyMem_Malloc((&workspace)->data->n * sizeof(c_float));
    y_arr = PyMem_Malloc((&workspace)->data->m * sizeof(c_float));

    // Initialize timer
    timer = PyMem_Malloc(sizeof(PyTimer));
    tic(timer);

    /**
     *  Solve QP Problem
     */
    osqp_solve((&workspace));

    // Stop timer
    solve_time = toc(timer);

    // If problem is not primal or dual infeasible store it
    if (((&workspace)->info->status_val != OSQP_PRIMAL_INFEASIBLE) &&
        ((&workspace)->info->status_val != OSQP_DUAL_INFEASIBLE)) {
            // Store solution into temporary arrays
            // N.B. Needed to be able to store RESULTS even when OSQP structure is deleted
            prea_vec_copy((&workspace)->solution->x, x_arr, (&workspace)->data->n);
            prea_vec_copy((&workspace)->solution->y, y_arr, (&workspace)->data->m);

            // Get primal dual solution PyArrayObjects
            x = PyArray_SimpleNewFromData(1, nd, float_type, x_arr);
            // Set x to own x_arr so that it is freed when x is freed
            PyArray_ENABLEFLAGS((PyArrayObject *) x, NPY_ARRAY_OWNDATA);

            y = PyArray_SimpleNewFromData(1, md, float_type, y_arr);
            // Set y to own y_arr so that it is freed when y is freed
            PyArray_ENABLEFLAGS((PyArrayObject *) y, NPY_ARRAY_OWNDATA);

    } else { // Problem primal or dual infeasible -> None values for x,y
            x = PyArray_EMPTY(1, nd, NPY_OBJECT, 0);
            y = PyArray_EMPTY(1, nd, NPY_OBJECT, 0);
    }

    // Free timer
    PyMem_Free(timer);

    // Return value
    return Py_BuildValue("OOiid", x, y, (&workspace)->info->status_val, (&workspace)->info->iter, solve_time);

}



static PyObject *OSQP_update_lin_cost(PyObject *self, PyObject *args){
    PyArrayObject *q, *q_cont;
    c_float * q_arr;
    int float_type = get_float_type();

    static char * argparse_string = "O!";

    // Parse arguments
    if( !PyArg_ParseTuple(args, argparse_string,
                          &PyArray_Type, &q)) {
            return NULL;
    }

    // Check dimension
    if (PyArray_DIM(q, 0) != (&workspace)->data->n){
        PySys_WriteStdout("Error in linear cost dimension!\n");
        return NULL;
    }

    // Get contiguous data structure
    q_cont = get_contiguous(q, float_type);

    // Copy array into c_float array
    q_arr = (c_float *)PyArray_DATA(q_cont);

    // Update linear cost
    osqp_update_lin_cost((&workspace), q_arr);

    // Free data
    Py_DECREF(q_cont);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}

static PyObject *OSQP_update_lower_bound(PyObject *self, PyObject *args){
    PyArrayObject *l, *l_cont;
    c_float * l_arr;
    int float_type = get_float_type();

    static char * argparse_string = "O!";

    // Parse arguments
    if( !PyArg_ParseTuple(args, argparse_string,
                          &PyArray_Type, &l)) {
            return NULL;
    }

    // Check dimension
    if (PyArray_DIM(l, 0) != (&workspace)->data->m){
        PySys_WriteStdout("Error in lower bound dimension!\n");
        return NULL;
    }

    // Get contiguous data structure
    l_cont = get_contiguous(l, float_type);

    // Copy array into c_float array
    l_arr = (c_float *)PyArray_DATA(l_cont);

    // Update linear cost
    osqp_update_lower_bound((&workspace), l_arr);

    // Free data
    Py_DECREF(l_cont);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}

static PyObject *OSQP_update_upper_bound(PyObject *self, PyObject *args){
    PyArrayObject *u, *u_cont;
    c_float * u_arr;
    int float_type = get_float_type();

    static char * argparse_string = "O!";

    // Parse arguments
    if( !PyArg_ParseTuple(args, argparse_string,
                          &PyArray_Type, &u)) {
            return NULL;
    }

    // Check dimension
    if (PyArray_DIM(u, 0) != (&workspace)->data->m){
        PySys_WriteStdout("Error in upper bound dimension!\n");
        return NULL;
    }

    // Get contiguous data structure
    u_cont = get_contiguous(u, float_type);

    // Copy array into c_float array
    u_arr = (c_float *)PyArray_DATA(u_cont);

    // Update linear cost
    osqp_update_upper_bound((&workspace), u_arr);

    // Free data
    Py_DECREF(u_cont);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}


static PyObject *OSQP_update_bounds(PyObject *self, PyObject *args){
    PyArrayObject *l, *l_cont, *u, *u_cont;
    c_float * l_arr, * u_arr;
    int float_type = get_float_type();

    static char * argparse_string = "O!O!";


    // Parse arguments
    if( !PyArg_ParseTuple(args, argparse_string,
                          &PyArray_Type, &l,
													&PyArray_Type, &u)) {
            return NULL;
    }

    // Check dimension
    if (PyArray_DIM(u, 0) != (&workspace)->data->m){
        PySys_WriteStdout("Error in upper bound dimension!\n");
        return NULL;
    }

    // Check dimension
    if (PyArray_DIM(l, 0) != (&workspace)->data->m){
        PySys_WriteStdout("Error in lower bound dimension!\n");
        return NULL;
    }


    // Get contiguous data structure
    u_cont = get_contiguous(u, float_type);

    // Get contiguous data structure
    l_cont = get_contiguous(l, float_type);

    // Copy array into c_float array
    u_arr = (c_float *)PyArray_DATA(u_cont);

    // Copy array into c_float array
    l_arr = (c_float *)PyArray_DATA(l_cont);

    // Update linear cost
    osqp_update_bounds((&workspace), l_arr, u_arr);

    // Free data
    Py_DECREF(u_cont);
    Py_DECREF(l_cont);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}



static PyMethodDef PYTHON_EXT_NAME_methods[] = {
        {"solve", (PyCFunction)OSQP_solve, METH_NOARGS, "Solve QP"},
        {"update_lin_cost", (PyCFunction)OSQP_update_lin_cost, METH_VARARGS, "Update linear cost"},
        {"update_lower_bound", (PyCFunction)OSQP_update_lower_bound, METH_VARARGS, "Update lower bound"},
        {"update_upper_bound", (PyCFunction)OSQP_update_upper_bound, METH_VARARGS, "Update upper bound"},
        {"update_bounds", (PyCFunction)OSQP_update_bounds, METH_VARARGS, "Update solver bounds"},
        {NULL, NULL, 0, NULL}
};



/* Module initialization for Python 3*/
 #if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "PYTHON_EXT_NAME",    /* m_name */
        "Embedded OSQP solver",             /* m_doc */
        -1,                                 /* m_size */
        PYTHON_EXT_NAME_methods,                     /* m_methods */
        NULL,                               /* m_reload */
        NULL,                               /* m_traverse */
        NULL,                               /* m_clear */
        NULL,                               /* m_free */
};
 #endif



static PyObject * moduleinit(void){

        PyObject *m;

        // Initialize module (no methods. all inside OSQP object)
    #if PY_MAJOR_VERSION >= 3
        m = PyModule_Create(&moduledef);
    #else
        m = Py_InitModule3("PYTHON_EXT_NAME", PYTHON_EXT_NAME_methods, "Embedded OSQP solver");
    #endif
        if (m == NULL)
                return NULL;

        return m;
}




// Init Osqp Internal module
#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_PYTHON_EXT_NAME(void)
#else
PyMODINIT_FUNC initPYTHON_EXT_NAME(void)
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
