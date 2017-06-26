#ifndef OSQPOBJECTPY_H
#define OSQPOBJECTPY_H

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
        npy_intp nd[] = {(npy_intp)self->workspace->data->n};  // Dimensions in R^n
        npy_intp md[] = {(npy_intp)self->workspace->data->m};  // Dimensions in R^m

        /**
         *  Solve QP Problem
         */
        osqp_solve(self->workspace);

        // If problem is not primal or dual infeasible store it
        if ((self->workspace->info->status_val != OSQP_PRIMAL_INFEASIBLE) &&
            (self->workspace->info->status_val != OSQP_DUAL_INFEASIBLE)){

			// Construct primal and dual solution arrays
			x = (PyObject *)PyArrayFromCArray(self->workspace->solution->x,
						          nd);
			y = (PyObject *)PyArrayFromCArray(self->workspace->solution->y,
								  md);

        } else { // Problem primal or dual infeasible -> None values for x,y
            x = PyArray_EMPTY(1, nd, NPY_OBJECT, 0);
            y = PyArray_EMPTY(1, nd, NPY_OBJECT, 0);
        }

        // If problem primal infeasible, set objective value to numpy infinity
        if (self->workspace->info->status_val == OSQP_PRIMAL_INFEASIBLE){
            self->workspace->info->obj_val = NPY_INFINITY;
        }

        // If problem dual infeasible, set objective value to numpy -infinity
        if (self->workspace->info->status_val == OSQP_DUAL_INFEASIBLE){
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
                                 "scaling", "scaling_iter",
                                 "rho", "sigma", "max_iter",
                                 "eps_abs", "eps_rel", "eps_prim_inf", "eps_dual_inf", "alpha",
                                 "delta", "polish", "pol_refine_iter", "auto_rho", "verbose",
                                 "scaled_termination", "early_terminate", "early_terminate_interval",
								 "warm_start", NULL};  // Settings


        #ifdef DLONG

        #ifdef DFLOAT
        static char * argparse_string = "(ll)O!O!O!O!O!O!O!O!O!|llfflffffffllllllll";
        #else
        static char * argparse_string = "(ll)O!O!O!O!O!O!O!O!O!|llddlddddddllllllll";
        #endif

        #else

        #ifdef DFLOAT
        static char * argparse_string = "(ii)O!O!O!O!O!O!O!O!O!|iiffiffffffiiiiiiii";
        #else
        static char * argparse_string = "(ii)O!O!O!O!O!O!O!O!O!|iiddiddddddiiiiiiii";
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
                                         &settings->scaling_iter,
                                         &settings->rho,
                                         &settings->sigma,
                                         &settings->max_iter,
                                         &settings->eps_abs,
                                         &settings->eps_rel,
                                         &settings->eps_prim_inf,
                                         &settings->eps_dual_inf,
                                         &settings->alpha,
                                         &settings->delta,
                                         &settings->polish,
					 &settings->pol_refine_iter,
					 &settings->auto_rho,
                                         &settings->verbose,
                                         &settings->scaled_termination,
                                         &settings->early_terminate,
					 &settings->early_terminate_interval,
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

    if(!strcmp(constant_name, "OSQP_PRIMAL_INFEASIBLE")){
        return Py_BuildValue("i", OSQP_PRIMAL_INFEASIBLE);
    }

    if(!strcmp(constant_name, "OSQP_DUAL_INFEASIBLE")){
        return Py_BuildValue("i", OSQP_DUAL_INFEASIBLE);
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

    // Update lower bound
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

    // Update upper bound
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

    // Update bounds
    osqp_update_bounds(self->workspace, l_arr, u_arr);

    // Free data
    Py_DECREF(l_cont);
    Py_DECREF(u_cont);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}

// Update elements of matrix P
static PyObject * OSQP_update_P(OSQP *self, PyObject *args) {
		PyArrayObject *Px, *Px_cont, *Px_idx, *Px_idx_cont;
		c_float * Px_arr;
		c_int * Px_idx_arr;
		c_int Px_n;
		int float_type = get_float_type();
		int int_type = get_int_type();

		#ifdef DLONG
		static char * argparse_string = "OOl";
		#else
		static char * argparse_string = "OOi";
		#endif

		// Parse arguments
		if( !PyArg_ParseTuple(args, argparse_string, &Px, &Px_idx, &Px_n)) {
						return NULL;
		}

		// Check if Px_idx is passed
		if((PyObject *)Px_idx != Py_None){
				Px_idx_cont = get_contiguous(Px_idx, int_type);
				Px_idx_arr = (c_int *)PyArray_DATA(Px_idx_cont);
		} else {
				Px_idx_cont = OSQP_NULL;
				Px_idx_arr = OSQP_NULL;
		}

		// Get contiguous data structure
		Px_cont = get_contiguous(Px, float_type);

		// Copy array into c_float and c_int arrays
		Px_arr = (c_float *)PyArray_DATA(Px_cont);

		// Update matrix P
		osqp_update_P(self->workspace, Px_arr, Px_idx_arr, Px_n);

	  // Free data
	  Py_DECREF(Px_cont);
		if ((PyObject *)Px_idx != Py_None) Py_DECREF(Px_idx_cont);


    // Return None
    Py_INCREF(Py_None);
    return Py_None;
}

// Update elements of matrix A
static PyObject * OSQP_update_A(OSQP *self, PyObject *args) {
		PyArrayObject *Ax, *Ax_cont, *Ax_idx, *Ax_idx_cont;
		c_float * Ax_arr;
		c_int * Ax_idx_arr;
		c_int Ax_n;
		int float_type = get_float_type();
		int int_type = get_int_type();

		#ifdef DLONG
		static char * argparse_string = "OOl";
		#else
		static char * argparse_string = "OOi";
		#endif

		// Parse arguments
		if( !PyArg_ParseTuple(args, argparse_string, &Ax, &Ax_idx, &Ax_n)) {
						return NULL;
		}

		// Check if Ax_idx is passed
		if((PyObject *)Ax_idx != Py_None){
				Ax_idx_cont = get_contiguous(Ax_idx, int_type);
				Ax_idx_arr = (c_int *)PyArray_DATA(Ax_idx_cont);
		} else {
				Ax_idx_cont = OSQP_NULL;
				Ax_idx_arr = OSQP_NULL;
		}

		// Get contiguous data structure
		Ax_cont = get_contiguous(Ax, float_type);

		// Copy array into c_float and c_int arrays
		Ax_arr = (c_float *)PyArray_DATA(Ax_cont);

		// Update matrix A
		osqp_update_A(self->workspace, Ax_arr, Ax_idx_arr, Ax_n);

    // Free data
    Py_DECREF(Ax_cont);
	if ((PyObject *)Ax_idx != Py_None) Py_DECREF(Ax_idx_cont);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;
}

// Update elements of matrices P and A
static PyObject * OSQP_update_P_A(OSQP *self, PyObject *args) {
		PyArrayObject *Px, *Px_cont, *Px_idx, *Px_idx_cont;
		PyArrayObject *Ax, *Ax_cont, *Ax_idx, *Ax_idx_cont;
		c_float * Px_arr, * Ax_arr;
		c_int * Px_idx_arr, * Ax_idx_arr;
		c_int Px_n, Ax_n;
		int float_type = get_float_type();
		int int_type = get_int_type();

		#ifdef DLONG
		static char * argparse_string = "OOlOOl";
		#else
		static char * argparse_string = "OOiOOi";
		#endif

		// Parse arguments
		if( !PyArg_ParseTuple(args, argparse_string, &Px, &Px_idx, &Px_n,
													&Ax, &Ax_idx, &Ax_n)) {
						return NULL;
		}

		// Check if Ax_idx is passed
		if((PyObject *)Ax_idx != Py_None){
				Ax_idx_cont = get_contiguous(Ax_idx, int_type);
				Ax_idx_arr = (c_int *)PyArray_DATA(Ax_idx_cont);
		} else {
				Ax_idx_cont = OSQP_NULL;
				Ax_idx_arr = OSQP_NULL;
		}

		// Check if Px_idx is passed
		if((PyObject *)Px_idx != Py_None){
				Px_idx_cont = get_contiguous(Px_idx, int_type);
				Px_idx_arr = (c_int *)PyArray_DATA(Px_idx_cont);
		} else {
				Px_idx_cont = OSQP_NULL;
				Px_idx_arr = OSQP_NULL;
		}

		// Get contiguous data structure
		Px_cont = get_contiguous(Px, float_type);
		Ax_cont = get_contiguous(Ax, float_type);

		// Copy array into c_float and c_int arrays
		Px_arr = (c_float *)PyArray_DATA(Px_cont);
		Ax_arr = (c_float *)PyArray_DATA(Ax_cont);

		// Update matrices P and A
		osqp_update_P_A(self->workspace, Px_arr, Px_idx_arr, Px_n, Ax_arr, Ax_idx_arr, Ax_n);

    // Free data
    Py_DECREF(Px_cont);
		if ((PyObject *)Px_idx != Py_None) Py_DECREF(Px_idx_cont);
		Py_DECREF(Ax_cont);
		if ((PyObject *)Ax_idx != Py_None) Py_DECREF(Ax_idx_cont);

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


static PyObject *OSQP_update_eps_prim_inf(OSQP *self, PyObject *args){
    c_float eps_prim_inf_new;

    #ifdef DFLOAT
    static char * argparse_string = "f";
    #else
    static char * argparse_string = "d";
    #endif

    // Parse arguments
    if( !PyArg_ParseTuple(args, argparse_string, &eps_prim_inf_new)) {
        return NULL;
    }

    // Perform Update
    osqp_update_eps_prim_inf(self->workspace, eps_prim_inf_new);

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}

static PyObject *OSQP_update_eps_dual_inf(OSQP *self, PyObject *args){
    c_float eps_dual_inf_new;

    #ifdef DFLOAT
    static char * argparse_string = "f";
    #else
    static char * argparse_string = "d";
    #endif

    // Parse arguments
    if( !PyArg_ParseTuple(args, argparse_string, &eps_dual_inf_new)) {
        return NULL;
    }

    // Perform Update
    osqp_update_eps_dual_inf(self->workspace, eps_dual_inf_new);

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

static PyObject *OSQP_update_scaled_termination(OSQP *self, PyObject *args){
    c_int scaled_termination_new;

    #ifdef DLONG
    static char * argparse_string = "l";
    #else
    static char * argparse_string = "i";
    #endif

    // Parse arguments
    if( !PyArg_ParseTuple(args, argparse_string, &scaled_termination_new)) {
        return NULL;
    }

    // Perform Update
    osqp_update_scaled_termination(self->workspace, scaled_termination_new);

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


static PyObject *OSQP_update_early_terminate_interval(OSQP *self, PyObject *args){
    c_int early_terminate_interval_new;

    #ifdef DLONG
    static char * argparse_string = "l";
    #else
    static char * argparse_string = "i";
    #endif

    // Parse arguments
    if( !PyArg_ParseTuple(args, argparse_string, &early_terminate_interval_new)) {
        return NULL;
    }

    // Perform Update
    osqp_update_early_terminate_interval(self->workspace, early_terminate_interval_new);

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
		{"update_P",	(PyCFunction)OSQP_update_P, METH_VARARGS, PyDoc_STR("Update OSQP problem quadratic cost matrix")},
		{"update_P_A",	(PyCFunction)OSQP_update_P_A, METH_VARARGS, PyDoc_STR("Update OSQP problem matrices")},
		{"update_A",	(PyCFunction)OSQP_update_A, METH_VARARGS, PyDoc_STR("Update OSQP problem constraint matrix")},
    {"warm_start",	(PyCFunction)OSQP_warm_start, METH_VARARGS, PyDoc_STR("Warm start primal and dual variables")},
    {"warm_start_x",	(PyCFunction)OSQP_warm_start_x, METH_VARARGS, PyDoc_STR("Warm start primal variable")},
    {"warm_start_y",	(PyCFunction)OSQP_warm_start_y, METH_VARARGS, PyDoc_STR("Warm start dual variable")},
    {"update_max_iter",	(PyCFunction)OSQP_update_max_iter, METH_VARARGS, PyDoc_STR("Update OSQP solver setting max_iter")},
    {"update_eps_abs",	(PyCFunction)OSQP_update_eps_abs, METH_VARARGS, PyDoc_STR("Update OSQP solver setting eps_abs")},
    {"update_eps_rel",	(PyCFunction)OSQP_update_eps_rel, METH_VARARGS, PyDoc_STR("Update OSQP solver setting eps_rel")},
    {"update_eps_prim_inf",	(PyCFunction)OSQP_update_eps_prim_inf, METH_VARARGS, PyDoc_STR("Update OSQP solver setting eps_prim_inf")},
    {"update_eps_dual_inf",	(PyCFunction)OSQP_update_eps_dual_inf, METH_VARARGS, PyDoc_STR("Update OSQP solver setting eps_dual_inf")},
    {"update_alpha",	(PyCFunction)OSQP_update_alpha, METH_VARARGS, PyDoc_STR("Update OSQP solver setting alpha")},
    {"update_delta",	(PyCFunction)OSQP_update_delta, METH_VARARGS, PyDoc_STR("Update OSQP solver setting delta")},
    {"update_polish",	(PyCFunction)OSQP_update_polish, METH_VARARGS, PyDoc_STR("Update OSQP solver setting polish")},
    {"update_pol_refine_iter",	(PyCFunction)OSQP_update_pol_refine_iter, METH_VARARGS, PyDoc_STR("Update OSQP solver setting pol_refine_iter")},
    {"update_verbose",	(PyCFunction)OSQP_update_verbose, METH_VARARGS, PyDoc_STR("Update OSQP solver setting verbose")},
    {"update_scaled_termination",	(PyCFunction)OSQP_update_scaled_termination, METH_VARARGS, PyDoc_STR("Update OSQP solver setting scaled_termination")},
    {"update_early_terminate",	(PyCFunction)OSQP_update_early_terminate, METH_VARARGS, PyDoc_STR("Update OSQP solver setting early_terminate")},
    {"update_early_terminate_interval",	(PyCFunction)OSQP_update_early_terminate_interval, METH_VARARGS, PyDoc_STR("Update OSQP solver setting early_terminate_interval")},
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

#endif
