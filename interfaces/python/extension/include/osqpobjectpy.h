/****************************************
 * OSQP Object definition and methods   *
 ****************************************/



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
                                 "delta", "polish", "pol_refine_iter", "verbose",
                                 "early_terminate", "warm_start", NULL};  // Settings


        #ifdef DLONG

        #ifdef DFLOAT
        static char * argparse_string = "(ll)O!O!O!O!O!O!O!O!O!|lllfflfffffflllll";
        #else
        static char * argparse_string = "(ll)O!O!O!O!O!O!O!O!O!|lllddlddddddlllll";
        #endif

        #else

        #ifdef DFLOAT
        static char * argparse_string = "(ii)O!O!O!O!O!O!O!O!O!|iiiffiffffffiiiii";
        #else
        static char * argparse_string = "(ii)O!O!O!O!O!O!O!O!O!|iiiddiddddddiiiii";
        #endif

        #endif

        // OSQPData and settings
        PyOSQPData *pydata;
        OSQPData * data;
        OSQPSettings * settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
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
                                         &settings->polish,
                                         &settings->pol_refine_iter,
                                         &settings->verbose,
                                         &settings->early_terminate,
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

    // Parse arguments
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

    // Parse arguments
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

    // Parse arguments
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

    // Parse arguments
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

    // Parse arguments
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

    // Parse arguments
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

    // Parse arguments
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

    // Parse arguments
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
    // Parse arguments
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

    // Parse arguments
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

    // Parse arguments
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

    // Parse arguments
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

    // Parse arguments
    if( !PyArg_ParseTuple(args, argparse_string, &delta_new)) {
            return NULL;
    }

    // Perform Update
    osqp_update_delta(self->workspace, delta_new);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}


static PyObject *OSQP_update_polish(OSQP *self, PyObject *args){
    c_int polish_new;

    #ifdef DLONG
    static char * argparse_string = "l";
    #else
    static char * argparse_string = "i";
    #endif

    // Parse arguments
    if( !PyArg_ParseTuple(args, argparse_string, &polish_new)) {
            return NULL;
    }

    // Perform Update
    osqp_update_polish(self->workspace, polish_new);

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

    // Parse arguments
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

    // Parse arguments
    if( !PyArg_ParseTuple(args, argparse_string, &verbose_new)) {
            return NULL;
    }

    // Perform Update
    osqp_update_verbose(self->workspace, verbose_new);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}

static PyObject *OSQP_update_early_terminate(OSQP *self, PyObject *args){
    c_int early_terminate_new;

    #ifdef DLONG
    static char * argparse_string = "l";
    #else
    static char * argparse_string = "i";
    #endif

    // Parse arguments
    if( !PyArg_ParseTuple(args, argparse_string, &early_terminate_new)) {
            return NULL;
    }

    // Perform Update
    osqp_update_early_terminate(self->workspace, early_terminate_new);

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

    // Parse arguments
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
    {"update_polish",	(PyCFunction)OSQP_update_polish, METH_VARARGS, PyDoc_STR("Update OSQP solver setting polish")},
    {"update_pol_refine_iter",	(PyCFunction)OSQP_update_pol_refine_iter, METH_VARARGS, PyDoc_STR("Update OSQP solver setting pol_refine_iter")},
    {"update_verbose",	(PyCFunction)OSQP_update_verbose, METH_VARARGS, PyDoc_STR("Update OSQP solver setting verbose")},
    {"update_early_terminate",	(PyCFunction)OSQP_update_early_terminate, METH_VARARGS, PyDoc_STR("Update OSQP solver setting early_terminate")},
    {"update_warm_start",	(PyCFunction)OSQP_update_warm_start, METH_VARARGS, PyDoc_STR("Update OSQP solver setting warm_start")},
    {"_get_workspace", (PyCFunction)OSQP_get_workspace, METH_VARARGS, PyDoc_STR("Returns the OSQP workspace struct as a Python dictionary.")},
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
