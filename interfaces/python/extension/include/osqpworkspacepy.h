/**********************************************
 * OSQP Workspace creation in Python objects  *
 **********************************************/


// TODO: Extract long integers and doubles (make sure of that!)
//


// Include private header to access to private structure
#include "lin_sys/direct/suitesparse/private.h"


 static PyObject *OSQP_get_scaling(OSQP *self){
     npy_intp n = (npy_intp)self->workspace->data->n;  // Dimensions in R^n
     npy_intp m = (npy_intp)self->workspace->data->m;  // Dimensions in R^m
     int float_type = get_float_type();

     /* Build Arrays. */
     OSQPScaling *scaling = self->workspace->scaling;
     PyObject *D    = PyArray_SimpleNewFromData(1, &n, float_type, scaling->D);
     PyObject *E    = PyArray_SimpleNewFromData(1, &m, float_type, scaling->E);
     PyObject *Dinv = PyArray_SimpleNewFromData(1, &n, float_type, scaling->Dinv);
     PyObject *Einv = PyArray_SimpleNewFromData(1, &m, float_type, scaling->Einv);

     /* Change data ownership. */
     PyArray_ENABLEFLAGS((PyArrayObject *) D, NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) E, NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) Dinv, NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) Einv, NPY_ARRAY_OWNDATA);

     /* Build Python dictionary. */
     PyObject *return_dict = Py_BuildValue("{s:O,s:O,s:O,s:O}",
                                           "D", D,
                                           "E", E,
                                           "Dinv", Dinv,
                                           "Einv", Einv);

     return return_dict;
 }


 static PyObject *OSQP_get_data(OSQP *self){
     OSQPData *data = self->workspace->data;
     npy_intp n = (npy_intp)data->n;
     npy_intp n_plus_1 = n+1;
     npy_intp m = (npy_intp)data->m;
     npy_intp m_plus_1 = m+1;
     npy_intp Pnzmax = (npy_intp)data->P->nzmax;
     npy_intp Anzmax = (npy_intp)data->A->nzmax;
     npy_intp Pnz = (npy_intp)data->P->nz;
     npy_intp Anz = (npy_intp)data->A->nz;

     int float_type = get_float_type();
     int int_type   = get_int_type();

     /* Build Arrays. */
     PyObject *Pp   = PyArray_SimpleNewFromData(1, &n_plus_1, int_type, data->P->p);
     PyObject *Pi   = PyArray_SimpleNewFromData(1, &Pnzmax, int_type, data->P->i);
     PyObject *Px   = PyArray_SimpleNewFromData(1, &Pnzmax, float_type, data->P->x);
     PyObject *Ap   = PyArray_SimpleNewFromData(1, &m_plus_1, int_type, data->A->p);
     PyObject *Ai   = PyArray_SimpleNewFromData(1, &Anzmax, int_type, data->A->i);
     PyObject *Ax   = PyArray_SimpleNewFromData(1, &Anzmax, float_type, data->A->x);
     PyObject *q    = PyArray_SimpleNewFromData(1, &n, float_type, data->q);
     PyObject *l    = PyArray_SimpleNewFromData(1, &m, float_type, data->l);
     PyObject *u    = PyArray_SimpleNewFromData(1, &m, float_type, data->u);

     /* Change data ownership. */
     PyArray_ENABLEFLAGS((PyArrayObject *) Pp, NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) Pi, NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) Px, NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) Ap, NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) Ai, NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) Ax, NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) q, NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) l, NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) u, NPY_ARRAY_OWNDATA);

     PyObject *return_dict = Py_BuildValue(
         "{s:i,s:i,"
         "s:{s:i,s:i,s:i,s:O,s:O,s:O,s:i},"
         "s:{s:i,s:i,s:i,s:O,s:O,s:O,s:i},"
         "s:O,s:O,s:O}",
         "n", data->n, "m", data->m,
         "P", "nzmax", Pnzmax, "m", n, "n", n, "p", Pp, "i", Pi, "x", Px, "nz", Pnz,
         "A", "nzmax", Anzmax, "m", m, "n", n, "p", Ap, "i", Ai, "x", Ax, "nz", Anz,
         "q", q, "l", l, "u", u);

     return return_dict;
 }


 static PyObject *OSQP_get_priv(OSQP *self){
     Priv *priv= self->workspace->priv;

     npy_intp Ln = (npy_intp)priv->L->n;
     npy_intp Ln_plus_1 = Ln+1;
     npy_intp Lnzmax = (npy_intp)priv->L->nzmax;
     npy_intp Lnz = (npy_intp)priv->L->nz;

     int float_type = get_float_type();
     int int_type   = get_int_type();

     /* Build Arrays. */
     PyObject *Lp   = PyArray_SimpleNewFromData(1, &Ln_plus_1, int_type, priv->L->p);
     PyObject *Li   = PyArray_SimpleNewFromData(1, &Lnzmax, int_type, priv->L->i);
     PyObject *Lx   = PyArray_SimpleNewFromData(1, &Lnzmax, float_type, priv->L->x);
     PyObject *Dinv = PyArray_SimpleNewFromData(1, &Ln, float_type, priv->Dinv);
     PyObject *P    = PyArray_SimpleNewFromData(1, &Ln, int_type, priv->P);
     PyObject *bp   = PyArray_SimpleNewFromData(1, &Ln, float_type, priv->bp);

     /* Change data ownership. */
     PyArray_ENABLEFLAGS((PyArrayObject *) Lp, NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) Li, NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) Lx, NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) Dinv, NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) P, NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) bp, NPY_ARRAY_OWNDATA);

     PyObject *return_dict = Py_BuildValue(
         "{s:{s:i,s:i,s:i,s:O,s:O,s:O,s:i},"
         "s:O,s:O,s:O}",
         "L", "nzmax", Lnzmax, "m", Ln, "n", Ln, "p", Lp, "i", Li, "x", Lx, "nz", Lnz,
         "Dinv", Dinv, "P", P, "bp", bp);

     return return_dict;
 }


 static PyObject *OSQP_get_settings(OSQP *self){
     OSQPSettings *settings = self->workspace->settings;

     PyObject *return_dict = Py_BuildValue(
         "{s:d,s:d,s:i,s:i,s:i,s:i,s:d,s:d,s:d,s:d,s:i,s:i,s:i,s:i}",
         "rho", (double)settings->rho,
         "sigma", (double)settings->sigma,
         "scaling", settings->scaling,
         "scaling_norm", settings->scaling_norm,
         "scaling_iter", settings->scaling_iter,
         "max_iter", settings->max_iter,
         "eps_abs", (double)settings->eps_abs,
         "eps_rel", (double)settings->eps_rel,
         "alpha", (double)settings->alpha,
         "delta", (double)settings->delta,
         "verbose", settings->verbose,
         "warm_start", settings->warm_start);

     return return_dict;
 }


static PyObject *OSQP_get_workspace(OSQP *self){
     PyObject *data_py    = OSQP_get_data(self);
     PyObject *priv_py    = OSQP_get_priv(self);
     PyObject *scaling_py = OSQP_get_scaling(self);
     PyObject *settings_py= OSQP_get_settings(self);

     PyObject *return_dict = Py_BuildValue("{s:O,s:O,s:O,s:O}",
                                           "data", data_py,
                                           "priv", priv_py,
                                           "scaling", scaling_py,
                                           "settings", settings_py);
     // TODO do we have to decref things here?
     return return_dict;
}
