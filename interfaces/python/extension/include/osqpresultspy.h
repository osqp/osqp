#ifndef OSQPRESULTSPY_H
#define OSQPRESULTSPY_H

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

    // Parse arguments
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

#endif
