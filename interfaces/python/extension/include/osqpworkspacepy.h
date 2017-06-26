#ifndef OSQPWORKSPACEPY_H
#define OSQPWORKSPACEPY_H

/**********************************************
 * OSQP Workspace creation in Python objects  *
 **********************************************/


// TODO: Extract long integers and doubles (make sure of that!)
//


// Include private header to access to private structure
#include "private.h"


 static PyObject *OSQP_get_scaling(OSQP *self){


     if(self->workspace->settings->scaling) { // if scaling enabled
         npy_intp n = (npy_intp)self->workspace->data->n;  // Dimensions in R^n
         npy_intp m = (npy_intp)self->workspace->data->m;  // Dimensions in R^m

         int float_type = get_float_type();

         PyObject *return_dict;

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
         return_dict = Py_BuildValue("{s:O,s:O,s:O,s:O}",
                                     "D", D, "E", E, "Dinv", Dinv, "Einv", Einv);

         return return_dict;
    } else { // Scaling disabled. Return None
        Py_INCREF(Py_None);
        return Py_None;
    }
 }


 static PyObject *OSQP_get_data(OSQP *self){
     OSQPData *data = self->workspace->data;
     npy_intp n = (npy_intp)data->n;
     npy_intp n_plus_1 = n+1;
     npy_intp m = (npy_intp)data->m;
     npy_intp Pnzmax = (npy_intp)data->P->p[n];
     npy_intp Anzmax = (npy_intp)data->A->p[n];
     npy_intp Pnz = (npy_intp)data->P->nz;
     npy_intp Anz = (npy_intp)data->A->nz;

     int float_type = get_float_type();
     int int_type   = get_int_type();

     PyObject *return_dict;

     /* Build Arrays. */
     PyObject *Pp   = PyArray_SimpleNewFromData(1, &n_plus_1, int_type, data->P->p);
     PyObject *Pi   = PyArray_SimpleNewFromData(1, &Pnzmax, int_type, data->P->i);
     PyObject *Px   = PyArray_SimpleNewFromData(1, &Pnzmax, float_type, data->P->x);
     PyObject *Ap   = PyArray_SimpleNewFromData(1, &n_plus_1, int_type, data->A->p);
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

     return_dict = Py_BuildValue(
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
     OSQPData *data = self->workspace->data;

     npy_intp Ln          = (npy_intp)priv->L->n;
     npy_intp Ln_plus_1   = Ln+1;
     npy_intp Lnzmax      = (npy_intp)priv->L->p[Ln];
     npy_intp Lnz         = (npy_intp)priv->L->nz;
     npy_intp Pdiag_n     = (npy_intp)priv->Pdiag_n;
     npy_intp KKTn        = (npy_intp)priv->KKT->n;
     npy_intp KKTn_plus_1 = KKTn+1;
     npy_intp KKTnzmax    = (npy_intp)priv->KKT->p[KKTn];
     npy_intp KKTnz       = (npy_intp)priv->KKT->nz;
     npy_intp Pnzmax      = (npy_intp)data->P->p[data->P->n];
     npy_intp Anzmax      = (npy_intp)data->A->p[data->A->n];
     npy_intp m_plus_n    = (npy_intp)(data->m + data->n);

     int float_type = get_float_type();
     int int_type   = get_int_type();

     PyObject *return_dict;

     /* Build Arrays. */
     PyObject *Lp        = PyArray_SimpleNewFromData(1, &Ln_plus_1,   int_type,   priv->L->p);
     PyObject *Li        = PyArray_SimpleNewFromData(1, &Lnzmax,      int_type,   priv->L->i);
     PyObject *Lx        = PyArray_SimpleNewFromData(1, &Lnzmax,      float_type, priv->L->x);
     PyObject *Dinv      = PyArray_SimpleNewFromData(1, &Ln,          float_type, priv->Dinv);
     PyObject *P         = PyArray_SimpleNewFromData(1, &Ln,          int_type,   priv->P);
     PyObject *bp        = PyArray_SimpleNewFromData(1, &Ln,          float_type, priv->bp);
     PyObject *Pdiag_idx = PyArray_SimpleNewFromData(1, &Pdiag_n,     int_type,   priv->Pdiag_idx);
     PyObject *KKTp      = PyArray_SimpleNewFromData(1, &KKTn_plus_1, int_type,   priv->KKT->p);
     PyObject *KKTi      = PyArray_SimpleNewFromData(1, &KKTnzmax,    int_type,   priv->KKT->i);
     PyObject *KKTx      = PyArray_SimpleNewFromData(1, &KKTnzmax,    float_type, priv->KKT->x);
     PyObject *PtoKKT    = PyArray_SimpleNewFromData(1, &Pnzmax,      int_type,   priv->PtoKKT);
     PyObject *AtoKKT    = PyArray_SimpleNewFromData(1, &Anzmax,      int_type,   priv->AtoKKT);
     PyObject *Lnz_vec   = PyArray_SimpleNewFromData(1, &m_plus_n,    int_type,   priv->Lnz);
     PyObject *Y         = PyArray_SimpleNewFromData(1, &m_plus_n,    float_type, priv->Y);
     PyObject *Pattern   = PyArray_SimpleNewFromData(1, &m_plus_n,    int_type,   priv->Pattern);
     PyObject *Flag      = PyArray_SimpleNewFromData(1, &m_plus_n,    int_type,   priv->Flag);
     PyObject *Parent    = PyArray_SimpleNewFromData(1, &m_plus_n,    int_type,   priv->Parent);

     /* Change data ownership. */
     PyArray_ENABLEFLAGS((PyArrayObject *) Lp,        NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) Li,        NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) Lx,        NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) Dinv,      NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) P,         NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) bp,        NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) Pdiag_idx, NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) KKTp,      NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) KKTi,      NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) KKTx,      NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) PtoKKT,    NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) AtoKKT,    NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) Lnz_vec,   NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) Y,         NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) Pattern,   NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) Flag,      NPY_ARRAY_OWNDATA);
     PyArray_ENABLEFLAGS((PyArrayObject *) Parent,    NPY_ARRAY_OWNDATA);

     return_dict = Py_BuildValue(
         "{s:{s:i,s:i,s:i,s:O,s:O,s:O,s:i},"  // L
         "s:O,s:O,s:O,"                       // Dinv, P, bp
         "s:O,s:i,"                           // Pdiag_idx, Pdiag_n
         "s:{s:i,s:i,s:i,s:O,s:O,s:O,s:i},"   // KKT
         "s:O,s:O,s:O,s:O,"                   // PtoKKT, AtoKKT, Lnz, Y
         "s:O,s:O,s:O}",                      // Pattern, Flag, Parent
         "L", "nzmax", Lnzmax, "m", Ln, "n", Ln, "p", Lp, "i", Li, "x", Lx, "nz", Lnz,
         "Dinv", Dinv, "P", P, "bp", bp,
         "Pdiag_idx", Pdiag_idx, "Pdiag_n", Pdiag_n,
         "KKT", "nzmax", KKTnzmax, "m", KKTn, "n", KKTn, "p", KKTp, "i", KKTi, "x", KKTx, "nz", KKTnz,
         "PtoKKT", PtoKKT, "AtoKKT", AtoKKT, "Lnz", Lnz_vec, "Y", Y,
         "Pattern", Pattern, "Flag", Flag, "Parent", Parent);

     return return_dict;
 }


 static PyObject *OSQP_get_settings(OSQP *self){
     OSQPSettings *settings = self->workspace->settings;

     PyObject *return_dict = Py_BuildValue(
         "{s:d,s:d,s:i,s:i,s:i,s:d,s:d,s:d, s:d, s:d, s:i, s:i, s:i, s:i}",
         "rho", (double)settings->rho,
         "sigma", (double)settings->sigma,
         "scaling", settings->scaling,
         "scaling_iter", settings->scaling_iter,
         "max_iter", settings->max_iter,
         "eps_abs", (double)settings->eps_abs,
         "eps_rel", (double)settings->eps_rel,
         "eps_prim_inf", (double)settings->eps_prim_inf,
         "eps_dual_inf", (double)settings->eps_dual_inf,
         "alpha", (double)settings->alpha,
         "warm_start", settings->warm_start,
         "scaled_termination", settings->scaled_termination,
         "early_terminate", settings->early_terminate,
         "early_terminate_interval", settings->early_terminate_interval);
     return return_dict;
 }


static PyObject *OSQP_get_workspace(OSQP *self){
     PyObject *data_py = OSQP_get_data(self);
     PyObject *priv_py = OSQP_get_priv(self);
     PyObject *scaling_py = OSQP_get_scaling(self);
     PyObject *settings_py = OSQP_get_settings(self);

     PyObject *return_dict = Py_BuildValue("{s:O,s:O,s:O,s:O}",
                                           "data", data_py,
                                           "priv", priv_py,
                                           "scaling", scaling_py,
                                           "settings", settings_py);
     return return_dict;
}

#endif
