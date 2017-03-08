#ifndef OSQPINFOPY_H
#define OSQPINFOPY_H

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
    // Parse arguments
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

    // Parse arguments
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

#endif
