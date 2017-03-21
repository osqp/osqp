#include "types.h"
#include "private.h"

// Redefine type of the structure in private
// N.B. this makes sure the right amount of memory is allocated
typedef struct c_priv Priv;

// Define data structure
OSQPData data;

csc Pdata;
c_int Pdata_i[OSQP_P_NNZ];
c_int Pdata_p[OSQP_NDIM + 1];
c_float Pdata_x[OSQP_P_NNZ];


csc Adata;
c_int Adata_i[OSQP_A_NNZ];
c_int Adata_p[OSQP_NDIM + 1];
c_float Adata_x[OSQP_A_NNZ];


c_float qdata[OSQP_NDIM];
c_float ldata[OSQP_MDIM];
c_float udata[OSQP_MDIM];




// Define settings structure
OSQPSettings settings;


// Define scaling
OSQPScaling scaling;

c_float Dscaling[OSQP_NDIM];
c_float Dinvscaling[OSQP_NDIM];
c_float Escaling[OSQP_MDIM];
c_float Einvscaling[OSQP_MDIM];


// Define private structure
Priv priv;

csc priv_L;
c_int priv_L_i[OSQP_L_NNZ];
c_int priv_L_p[OSQP_KKT_NDIM + 1];
c_float priv_L_x[OSQP_L_NNZ];


c_float priv_Dinv[OSQP_KKT_NDIM];
c_int priv_P[OSQP_KKT_NDIM];
c_float priv_bp[OSQP_KKT_NDIM];


// TODO: Add embedded_flag == 2 case!


// Define solution
OSQPSolution solution;
c_float xsolution[OSQP_NDIM];
c_float ysolution[OSQP_MDIM];


// Define info
OSQPInfo info;


// Define workspace
OSQPWorkspace workspace;

c_float work_x[OSQP_NDIM];
c_float work_y[OSQP_MDIM];
c_float work_z[OSQP_MDIM];
c_float work_xz_tilde[OSQP_KKT_NDIM];

c_float work_x_prev[OSQP_NDIM];
c_float work_z_prev[OSQP_MDIM];

c_float work_delta_y[OSQP_MDIM];
c_float work_Atdelta_y[OSQP_NDIM];

c_float work_delta_x[OSQP_NDIM];
c_float work_Pdelta_x[OSQP_NDIM];
c_float work_Adelta_x[OSQP_MDIM];

c_float work_P_x[OSQP_NDIM];
c_float work_A_x[OSQP_MDIM];

c_float work_D_temp[OSQP_NDIM];

c_float work_E_temp[OSQP_MDIM];




// Link structures in workspace
void link_structures(){

    Pdata.i = Pdata_i;
    Pdata.p = Pdata_p;
    Pdata.x = Pdata_x;

    Adata.i = Adata_i;
    Adata.p = Adata_p;
    Adata.x = Adata_x;

    data.P = &Pdata;
    data.q = qdata;
    data.A = &Adata;
    data.u = udata;
    data.l = ldata;


    scaling.D = Dscaling;
    scaling.Dinv = Dinvscaling;
    scaling.E = Escaling;
    scaling.Einv = Einvscaling;

    priv_L.i = priv_L_i;
    priv_L.p = priv_L_p;
    priv_L.x = priv_L_x;

    priv.L = &priv_L;
    priv.Dinv = priv_Dinv;
    priv.P = priv_P;
    priv.bp = priv_bp;


    solution.x = xsolution;
    solution.y = ysolution;


    workspace.x = work_x;
    workspace.y = work_y;
    workspace.z = work_z;
    workspace.xz_tilde = work_xz_tilde;
    workspace.x_prev = work_x_prev;
    workspace.z_prev = work_z_prev;
    workspace.delta_y = work_delta_y;
    workspace.Atdelta_y = work_Atdelta_y;
    workspace.delta_x = work_delta_x;
    workspace.Pdelta_x = work_Pdelta_x;
    workspace.Adelta_x = work_Adelta_x;
    workspace.P_x = work_P_x;
    workspace.A_x = work_A_x;
    workspace.D_temp = work_D_temp;
    workspace.E_temp = work_E_temp;

    workspace.data = &data;
    workspace.priv = &priv;
    workspace.settings = &settings;
    workspace.scaling = &scaling;
    workspace.solution = &solution;
    workspace.info = &info;


}
