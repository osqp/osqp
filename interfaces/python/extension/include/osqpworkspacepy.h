/**********************************************
 * OSQP Workspace creation in Python objects  *
 **********************************************/


typedef struct {
    PyObject_HEAD
    // Internal solver variables
    PyArrayObject *x, *y, *z, *xz_tilde; // Iterates
    PyArrayObject *x_prev, *z_prev;      // Previous x and z
    PyArrayObject *delta_y, *Adelta_y;   // Infeasibility variables
    PyArrayObject *delta_x, *Pdelta_y, *Adelta_x; // Unboundedness variables
    PyArrayObject *P_x; *A_x;          // Scaling workspace vectors
    PyArrayObject *D_temp, *E_temp;    // Temporary scaling vectors

} OSQP_workspace;


static PyMemberDef OSQP_workspace_members[] = {
    {"x", T_OBJECT, offsetof(OSQP_results, x), 0, "Primal solution"},
    {"y", T_OBJECT, offsetof(OSQP_results, y), 0, "Dual solution"},

    // TODO: Complete!

    // {"info", T_OBJECT, offsetof(OSQP_results, info), 0, "Solver Information"},
    {NULL}
};



# TO COMPLETE!!!
