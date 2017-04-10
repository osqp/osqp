#ifndef OSQPUTILSPY_H
#define OSQPUTILSPY_H

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
} PyOSQPData;

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
        return NPY_INT64; /* defaults to 4 byte int */
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


// Function working on Python 3.6
static PyArrayObject * PyArrayFromCArray(c_float *arrayin, npy_intp * nd){
    int i;
    PyArrayObject * arrayout;
    double * data;

    arrayout = (PyArrayObject *)PyArray_SimpleNew(1, nd, NPY_DOUBLE);
    data = PyArray_DATA(arrayout);

    // Copy array into Python array
    for (i=0; i< nd[0]; i++){
        data[i] = (double)arrayin[i];
    }

    return arrayout;

}

// Original function supposed to work (Not in Python 3.6)
// static PyObject * PyArrayFromCArray(c_float *arrayin, npy_intp * nd,
//                                          int typenum){
// 	int i;
// 	PyObject * arrayout;
// 	c_float *x_arr;
//
// 	 // Allocate solutions
//     x_arr = PyMem_Malloc(nd[0] * sizeof(c_float));
//
// 	// copy elements to x_arr
// 	for (i=0; i< nd[0]; i++){
// 		x_arr[i] = arrayin[i];
//     }
//
// 	arrayout = PyArray_SimpleNewFromData(1, nd, typenum, x_arr);
// 	// Set x to own x_arr so that it is freed when x is freed
// 	PyArray_ENABLEFLAGS((PyArrayObject *) arrayout, NPY_ARRAY_OWNDATA);
//
//
//     return arrayout;
//
// }


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


static PyOSQPData * create_pydata(c_int n, c_int m,
                     PyArrayObject *Px, PyArrayObject *Pi, PyArrayObject *Pp,
                     PyArrayObject *q, PyArrayObject *Ax, PyArrayObject *Ai,
                     PyArrayObject *Ap, PyArrayObject *l, PyArrayObject *u){

    // Get int and float types
    int int_type = get_int_type();
    int float_type = get_float_type();

    // Populate PyOSQPData structure
    PyOSQPData * py_d = (PyOSQPData *)c_malloc(sizeof(PyOSQPData));
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
static OSQPData * create_data(PyOSQPData * py_d){

    // Allocate OSQPData structure
    OSQPData * data = (OSQPData *)c_malloc(sizeof(OSQPData));

    // Populate OSQPData structure
    data->n = py_d->n;
    data->m = py_d->m;
    data->P = csc_matrix(data->n, data->n,
                         (c_int) PyArray_DIM(py_d->Px, 0),  // nnz
                         (c_float *)PyArray_DATA(py_d->Px),
                         (c_int *)PyArray_DATA(py_d->Pi),
                         (c_int *)PyArray_DATA(py_d->Pp));
    data->q = (c_float *)PyArray_DATA(py_d->q);
    data->A = csc_matrix(data->m, data->n,
                         (c_int) PyArray_DIM(py_d->Ax, 0),  // nnz
                         (c_float *)PyArray_DATA(py_d->Ax),
                         (c_int *)PyArray_DATA(py_d->Ai),
                         (c_int *)PyArray_DATA(py_d->Ap));
    data->l = (c_float *)PyArray_DATA(py_d->l);
    data->u = (c_float *)PyArray_DATA(py_d->u);

    return data;
}


static c_int free_data(OSQPData *data, PyOSQPData * py_d){

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

    return 0;

}

#endif
