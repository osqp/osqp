#ifndef LIN_ALG_DATA_H
#define LIN_ALG_DATA_H
#include "osqp.h"


/* create data and solutions structure */
typedef struct {
csc * test_mat_ops_ew_abs;
c_float * test_mat_vec_y;
c_int test_vec_ops_n;
c_float test_vec_ops_norm2;
c_float * test_vec_ops_v2;
csc * test_sp_matrix_A;
c_float * test_mat_vec_Px_cum;
c_float * test_mat_vec_Ax_cum;
csc * test_mat_ops_prem_diag;
c_int test_mat_vec_n;
c_float * test_qpform_x;
c_float * test_mat_vec_Ax;
c_float * test_mat_vec_x;
c_float * test_vec_ops_ew_reciprocal;
csc * test_mat_extr_triu_P;
c_float * test_sp_matrix_Adns;
c_int test_qpform_n;
c_float * test_mat_vec_ATy;
c_float * test_vec_ops_v1;
c_int test_mat_ops_n;
c_float test_qpform_value;
c_float test_vec_ops_sc;
c_float * test_vec_ops_add_scaled;
csc * test_mat_ops_A;
c_float * test_mat_vec_Px;
c_int test_mat_extr_triu_n;
csc * test_mat_extr_triu_Pu;
csc * test_mat_ops_postm_diag;
c_int test_mat_vec_m;
c_float * test_mat_vec_ATy_cum;
c_float * test_mat_ops_d;
csc * test_mat_vec_Pu;
csc * test_mat_vec_A;
c_float test_vec_ops_vec_prod;
csc * test_qpform_Pu;
csc * test_mat_ops_ew_square;
c_float test_vec_ops_norm2_diff;
} lin_alg_sols_data;

/* function to define problem data */
lin_alg_sols_data *  generate_problem_lin_alg_sols_data(){

lin_alg_sols_data * data = (lin_alg_sols_data *)c_malloc(sizeof(lin_alg_sols_data));


// Matrix test_mat_ops_ew_abs
//---------------------------
data->test_mat_ops_ew_abs = c_malloc(sizeof(csc));
data->test_mat_ops_ew_abs->m = 2;
data->test_mat_ops_ew_abs->n = 2;
data->test_mat_ops_ew_abs->nz = -1;
data->test_mat_ops_ew_abs->nzmax = 3;
data->test_mat_ops_ew_abs->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_ew_abs->x[0] = 0.62766513772076149014;
data->test_mat_ops_ew_abs->x[1] = 0.04834707402576599033;
data->test_mat_ops_ew_abs->x[2] = 0.78541755270078594808;
data->test_mat_ops_ew_abs->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_ew_abs->i[0] = 1;
data->test_mat_ops_ew_abs->i[1] = 0;
data->test_mat_ops_ew_abs->i[2] = 1;
data->test_mat_ops_ew_abs->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_ew_abs->p[0] = 0;
data->test_mat_ops_ew_abs->p[1] = 1;
data->test_mat_ops_ew_abs->p[2] = 3;

data->test_mat_vec_y = c_malloc(5 * sizeof(c_float));
data->test_mat_vec_y[0] = -1.60730660657315560513;
data->test_mat_vec_y[1] = 1.81209883383505210297;
data->test_mat_vec_y[2] = 0.69354219184880350202;
data->test_mat_vec_y[3] = -0.39090847345761670084;
data->test_mat_vec_y[4] = 0.38994886235845566125;
data->test_vec_ops_n = 10;
data->test_vec_ops_norm2 = 2.16960658713026877109;
data->test_vec_ops_v2 = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_v2[0] = 0.69301937149786896342;
data->test_vec_ops_v2[1] = 2.47126696451565397439;
data->test_vec_ops_v2[2] = 1.23172377300913016285;
data->test_vec_ops_v2[3] = 0.51977999040890776605;
data->test_vec_ops_v2[4] = 0.05970748086383371317;
data->test_vec_ops_v2[5] = 0.35870047671493032748;
data->test_vec_ops_v2[6] = 1.10781982310653059898;
data->test_vec_ops_v2[7] = 0.08402358755182567918;
data->test_vec_ops_v2[8] = 1.98015662690100624488;
data->test_vec_ops_v2[9] = 0.38472854379849191764;

// Matrix test_sp_matrix_A
//------------------------
data->test_sp_matrix_A = c_malloc(sizeof(csc));
data->test_sp_matrix_A->m = 5;
data->test_sp_matrix_A->n = 6;
data->test_sp_matrix_A->nz = -1;
data->test_sp_matrix_A->nzmax = 30;
data->test_sp_matrix_A->x = c_malloc(30 * sizeof(c_float));
data->test_sp_matrix_A->x[0] = 0.21939329715813443711;
data->test_sp_matrix_A->x[1] = 1.47093864883172265046;
data->test_sp_matrix_A->x[2] = -1.54385053983811215694;
data->test_sp_matrix_A->x[3] = 2.37528570346344025666;
data->test_sp_matrix_A->x[4] = 0.23499136827079761258;
data->test_sp_matrix_A->x[5] = 1.18283703252498639813;
data->test_sp_matrix_A->x[6] = 1.02383258482997518968;
data->test_sp_matrix_A->x[7] = 0.72535889498978922685;
data->test_sp_matrix_A->x[8] = -1.40697818529832008316;
data->test_sp_matrix_A->x[9] = 0.79070254268935802333;
data->test_sp_matrix_A->x[10] = 0.96801525093549589673;
data->test_sp_matrix_A->x[11] = -0.04630915204284631603;
data->test_sp_matrix_A->x[12] = 0.18655932252031626195;
data->test_sp_matrix_A->x[13] = 0.19757156162017830425;
data->test_sp_matrix_A->x[14] = -1.51290421496221183517;
data->test_sp_matrix_A->x[15] = -0.55177765203161199814;
data->test_sp_matrix_A->x[16] = 0.40686332710025829851;
data->test_sp_matrix_A->x[17] = -0.27907618539435108396;
data->test_sp_matrix_A->x[18] = 0.26253367053101106521;
data->test_sp_matrix_A->x[19] = -0.70854787905188398867;
data->test_sp_matrix_A->x[20] = 0.45701183335689427034;
data->test_sp_matrix_A->x[21] = -0.95502227401915618099;
data->test_sp_matrix_A->x[22] = 0.78979253303003360553;
data->test_sp_matrix_A->x[23] = 0.76374498320029493570;
data->test_sp_matrix_A->x[24] = 1.64419578755487827415;
data->test_sp_matrix_A->x[25] = 0.81134690438235879117;
data->test_sp_matrix_A->x[26] = -1.24815792991564933345;
data->test_sp_matrix_A->x[27] = 0.32751625294310976466;
data->test_sp_matrix_A->x[28] = 1.55807760642326798006;
data->test_sp_matrix_A->x[29] = -0.33144627698270023863;
data->test_sp_matrix_A->i = c_malloc(30 * sizeof(c_int));
data->test_sp_matrix_A->i[0] = 0;
data->test_sp_matrix_A->i[1] = 1;
data->test_sp_matrix_A->i[2] = 2;
data->test_sp_matrix_A->i[3] = 3;
data->test_sp_matrix_A->i[4] = 4;
data->test_sp_matrix_A->i[5] = 0;
data->test_sp_matrix_A->i[6] = 1;
data->test_sp_matrix_A->i[7] = 2;
data->test_sp_matrix_A->i[8] = 3;
data->test_sp_matrix_A->i[9] = 4;
data->test_sp_matrix_A->i[10] = 0;
data->test_sp_matrix_A->i[11] = 1;
data->test_sp_matrix_A->i[12] = 2;
data->test_sp_matrix_A->i[13] = 3;
data->test_sp_matrix_A->i[14] = 4;
data->test_sp_matrix_A->i[15] = 0;
data->test_sp_matrix_A->i[16] = 1;
data->test_sp_matrix_A->i[17] = 2;
data->test_sp_matrix_A->i[18] = 3;
data->test_sp_matrix_A->i[19] = 4;
data->test_sp_matrix_A->i[20] = 0;
data->test_sp_matrix_A->i[21] = 1;
data->test_sp_matrix_A->i[22] = 2;
data->test_sp_matrix_A->i[23] = 3;
data->test_sp_matrix_A->i[24] = 4;
data->test_sp_matrix_A->i[25] = 0;
data->test_sp_matrix_A->i[26] = 1;
data->test_sp_matrix_A->i[27] = 2;
data->test_sp_matrix_A->i[28] = 3;
data->test_sp_matrix_A->i[29] = 4;
data->test_sp_matrix_A->p = c_malloc((6 + 1) * sizeof(c_int));
data->test_sp_matrix_A->p[0] = 0;
data->test_sp_matrix_A->p[1] = 5;
data->test_sp_matrix_A->p[2] = 10;
data->test_sp_matrix_A->p[3] = 15;
data->test_sp_matrix_A->p[4] = 20;
data->test_sp_matrix_A->p[5] = 25;
data->test_sp_matrix_A->p[6] = 30;

data->test_mat_vec_Px_cum = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_Px_cum[0] = -3.39458158715514990433;
data->test_mat_vec_Px_cum[1] = -4.23821980493991112837;
data->test_mat_vec_Px_cum[2] = -7.22194780578144523986;
data->test_mat_vec_Px_cum[3] = -4.78101255385152512645;
data->test_mat_vec_Ax_cum = c_malloc(5 * sizeof(c_float));
data->test_mat_vec_Ax_cum[0] = -3.66014825946914390542;
data->test_mat_vec_Ax_cum[1] = -1.43265007570017099958;
data->test_mat_vec_Ax_cum[2] = -1.89437269091793547560;
data->test_mat_vec_Ax_cum[3] = -2.43386244094980952468;
data->test_mat_vec_Ax_cum[4] = -3.47556296219329619035;

// Matrix test_mat_ops_prem_diag
//------------------------------
data->test_mat_ops_prem_diag = c_malloc(sizeof(csc));
data->test_mat_ops_prem_diag->m = 2;
data->test_mat_ops_prem_diag->n = 2;
data->test_mat_ops_prem_diag->nz = -1;
data->test_mat_ops_prem_diag->nzmax = 3;
data->test_mat_ops_prem_diag->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_prem_diag->x[0] = -0.25877634133286092633;
data->test_mat_ops_prem_diag->x[1] = -0.02516881079275864325;
data->test_mat_ops_prem_diag->x[2] = -0.32381514997721688731;
data->test_mat_ops_prem_diag->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_prem_diag->i[0] = 1;
data->test_mat_ops_prem_diag->i[1] = 0;
data->test_mat_ops_prem_diag->i[2] = 1;
data->test_mat_ops_prem_diag->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_prem_diag->p[0] = 0;
data->test_mat_ops_prem_diag->p[1] = 1;
data->test_mat_ops_prem_diag->p[2] = 3;

data->test_mat_vec_n = 4;
data->test_qpform_x = c_malloc(4 * sizeof(c_float));
data->test_qpform_x[0] = 0.05993847541773818927;
data->test_qpform_x[1] = 0.44400529432695040599;
data->test_qpform_x[2] = 0.48810442934561415473;
data->test_qpform_x[3] = 0.90337743549422078182;
data->test_mat_vec_Ax = c_malloc(5 * sizeof(c_float));
data->test_mat_vec_Ax[0] = -2.05284165289598830029;
data->test_mat_vec_Ax[1] = -3.24474890953522310255;
data->test_mat_vec_Ax[2] = -2.58791488276673886659;
data->test_mat_vec_Ax[3] = -2.04295396749219282384;
data->test_mat_vec_Ax[4] = -3.86551182455175190711;
data->test_mat_vec_x = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_x[0] = -0.93762554770098138768;
data->test_mat_vec_x[1] = -1.30115063403557051913;
data->test_mat_vec_x[2] = -1.60510192464836110027;
data->test_mat_vec_x[3] = -1.69064568769894330025;
data->test_vec_ops_ew_reciprocal = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_ew_reciprocal[0] = -1.33542720050404817300;
data->test_vec_ops_ew_reciprocal[1] = 5.34240198781485720048;
data->test_vec_ops_ew_reciprocal[2] = 40.36982912262022438199;
data->test_vec_ops_ew_reciprocal[3] = 0.69202033330684531443;
data->test_vec_ops_ew_reciprocal[4] = -1.13236687750831843680;
data->test_vec_ops_ew_reciprocal[5] = 1.25213449081899819149;
data->test_vec_ops_ew_reciprocal[6] = 9.28478687420028769850;
data->test_vec_ops_ew_reciprocal[7] = 1.56650847173118568101;
data->test_vec_ops_ew_reciprocal[8] = 30.41761301966236175076;
data->test_vec_ops_ew_reciprocal[9] = -2.32640883396123010485;

// Matrix test_mat_extr_triu_P
//----------------------------
data->test_mat_extr_triu_P = c_malloc(sizeof(csc));
data->test_mat_extr_triu_P->m = 5;
data->test_mat_extr_triu_P->n = 5;
data->test_mat_extr_triu_P->nz = -1;
data->test_mat_extr_triu_P->nzmax = 21;
data->test_mat_extr_triu_P->x = c_malloc(21 * sizeof(c_float));
data->test_mat_extr_triu_P->x[0] = 1.14917142469609556699;
data->test_mat_extr_triu_P->x[1] = 0.31173688668596044593;
data->test_mat_extr_triu_P->x[2] = 0.73625464657738093344;
data->test_mat_extr_triu_P->x[3] = 0.89049293620335590038;
data->test_mat_extr_triu_P->x[4] = 0.31173688668596044593;
data->test_mat_extr_triu_P->x[5] = 0.23639581052096092506;
data->test_mat_extr_triu_P->x[6] = 1.48399997612500200717;
data->test_mat_extr_triu_P->x[7] = 1.30534889243361718059;
data->test_mat_extr_triu_P->x[8] = 0.13138286738822435584;
data->test_mat_extr_triu_P->x[9] = 1.48399997612500200717;
data->test_mat_extr_triu_P->x[10] = 1.88334741755529133656;
data->test_mat_extr_triu_P->x[11] = 1.04838550837143995587;
data->test_mat_extr_triu_P->x[12] = 1.30603140872840350895;
data->test_mat_extr_triu_P->x[13] = 0.73625464657738093344;
data->test_mat_extr_triu_P->x[14] = 1.30534889243361718059;
data->test_mat_extr_triu_P->x[15] = 1.04838550837143995587;
data->test_mat_extr_triu_P->x[16] = 1.06649871849946875635;
data->test_mat_extr_triu_P->x[17] = 0.89049293620335590038;
data->test_mat_extr_triu_P->x[18] = 0.13138286738822435584;
data->test_mat_extr_triu_P->x[19] = 1.30603140872840350895;
data->test_mat_extr_triu_P->x[20] = 1.06649871849946875635;
data->test_mat_extr_triu_P->i = c_malloc(21 * sizeof(c_int));
data->test_mat_extr_triu_P->i[0] = 0;
data->test_mat_extr_triu_P->i[1] = 1;
data->test_mat_extr_triu_P->i[2] = 3;
data->test_mat_extr_triu_P->i[3] = 4;
data->test_mat_extr_triu_P->i[4] = 0;
data->test_mat_extr_triu_P->i[5] = 1;
data->test_mat_extr_triu_P->i[6] = 2;
data->test_mat_extr_triu_P->i[7] = 3;
data->test_mat_extr_triu_P->i[8] = 4;
data->test_mat_extr_triu_P->i[9] = 1;
data->test_mat_extr_triu_P->i[10] = 2;
data->test_mat_extr_triu_P->i[11] = 3;
data->test_mat_extr_triu_P->i[12] = 4;
data->test_mat_extr_triu_P->i[13] = 0;
data->test_mat_extr_triu_P->i[14] = 1;
data->test_mat_extr_triu_P->i[15] = 2;
data->test_mat_extr_triu_P->i[16] = 4;
data->test_mat_extr_triu_P->i[17] = 0;
data->test_mat_extr_triu_P->i[18] = 1;
data->test_mat_extr_triu_P->i[19] = 2;
data->test_mat_extr_triu_P->i[20] = 3;
data->test_mat_extr_triu_P->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_mat_extr_triu_P->p[0] = 0;
data->test_mat_extr_triu_P->p[1] = 4;
data->test_mat_extr_triu_P->p[2] = 9;
data->test_mat_extr_triu_P->p[3] = 13;
data->test_mat_extr_triu_P->p[4] = 17;
data->test_mat_extr_triu_P->p[5] = 21;

data->test_sp_matrix_Adns = c_malloc(30 * sizeof(c_float));
data->test_sp_matrix_Adns[0] = 0.21939329715813443711;
data->test_sp_matrix_Adns[1] = 1.47093864883172265046;
data->test_sp_matrix_Adns[2] = -1.54385053983811215694;
data->test_sp_matrix_Adns[3] = 2.37528570346344025666;
data->test_sp_matrix_Adns[4] = 0.23499136827079761258;
data->test_sp_matrix_Adns[5] = 1.18283703252498639813;
data->test_sp_matrix_Adns[6] = 1.02383258482997518968;
data->test_sp_matrix_Adns[7] = 0.72535889498978922685;
data->test_sp_matrix_Adns[8] = -1.40697818529832008316;
data->test_sp_matrix_Adns[9] = 0.79070254268935802333;
data->test_sp_matrix_Adns[10] = 0.96801525093549589673;
data->test_sp_matrix_Adns[11] = -0.04630915204284631603;
data->test_sp_matrix_Adns[12] = 0.18655932252031626195;
data->test_sp_matrix_Adns[13] = 0.19757156162017830425;
data->test_sp_matrix_Adns[14] = -1.51290421496221183517;
data->test_sp_matrix_Adns[15] = -0.55177765203161199814;
data->test_sp_matrix_Adns[16] = 0.40686332710025829851;
data->test_sp_matrix_Adns[17] = -0.27907618539435108396;
data->test_sp_matrix_Adns[18] = 0.26253367053101106521;
data->test_sp_matrix_Adns[19] = -0.70854787905188398867;
data->test_sp_matrix_Adns[20] = 0.45701183335689427034;
data->test_sp_matrix_Adns[21] = -0.95502227401915618099;
data->test_sp_matrix_Adns[22] = 0.78979253303003360553;
data->test_sp_matrix_Adns[23] = 0.76374498320029493570;
data->test_sp_matrix_Adns[24] = 1.64419578755487827415;
data->test_sp_matrix_Adns[25] = 0.81134690438235879117;
data->test_sp_matrix_Adns[26] = -1.24815792991564933345;
data->test_sp_matrix_Adns[27] = 0.32751625294310976466;
data->test_sp_matrix_Adns[28] = 1.55807760642326798006;
data->test_sp_matrix_Adns[29] = -0.33144627698270023863;
data->test_qpform_n = 4;
data->test_mat_vec_ATy = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_ATy[0] = 1.48716482459808019279;
data->test_mat_vec_ATy[1] = 0.43457514285773413132;
data->test_mat_vec_ATy[2] = 1.99076935290626799713;
data->test_mat_vec_ATy[3] = -0.04223672079278881220;
data->test_vec_ops_v1 = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_v1[0] = -0.74882404643439681724;
data->test_vec_ops_v1[1] = 0.18718172130828716893;
data->test_vec_ops_v1[2] = 0.02477097430763398889;
data->test_vec_ops_v1[3] = 1.44504424490168115192;
data->test_vec_ops_v1[4] = -0.88310601436914071272;
data->test_vec_ops_v1[5] = 0.79863625459747400459;
data->test_vec_ops_v1[6] = 0.10770306454515483907;
data->test_vec_ops_v1[7] = 0.63836233129009267717;
data->test_vec_ops_v1[8] = 0.03287568946825598221;
data->test_vec_ops_v1[9] = -0.42984706101604547301;
data->test_mat_ops_n = 2;
data->test_qpform_value = 1.46070734772099797283;
data->test_vec_ops_sc = -1.72650260325466708977;
data->test_vec_ops_add_scaled = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_add_scaled[0] = -1.94532379543138089772;
data->test_vec_ops_add_scaled[1] = -4.07946712626524821133;
data->test_vec_ops_add_scaled[2] = -2.10180332628329002986;
data->test_vec_ops_add_scaled[3] = 0.54764273834101595551;
data->test_vec_ops_add_scaled[4] = -0.98619113551432779907;
data->test_vec_ops_add_scaled[5] = 0.17933894776045666841;
data->test_vec_ops_add_scaled[6] = -1.80495074398539490090;
data->test_vec_ops_add_scaled[7] = 0.49329538864706923285;
data->test_vec_ops_add_scaled[8] = -3.38586988172831171440;
data->test_vec_ops_add_scaled[9] = -1.09408189343051898135;

// Matrix test_mat_ops_A
//----------------------
data->test_mat_ops_A = c_malloc(sizeof(csc));
data->test_mat_ops_A->m = 2;
data->test_mat_ops_A->n = 2;
data->test_mat_ops_A->nz = -1;
data->test_mat_ops_A->nzmax = 3;
data->test_mat_ops_A->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_A->x[0] = 0.62766513772076149014;
data->test_mat_ops_A->x[1] = 0.04834707402576599033;
data->test_mat_ops_A->x[2] = 0.78541755270078594808;
data->test_mat_ops_A->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_A->i[0] = 1;
data->test_mat_ops_A->i[1] = 0;
data->test_mat_ops_A->i[2] = 1;
data->test_mat_ops_A->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_A->p[0] = 0;
data->test_mat_ops_A->p[1] = 1;
data->test_mat_ops_A->p[2] = 3;

data->test_mat_vec_Px = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_Px[0] = -2.45695603945416829461;
data->test_mat_vec_Px[1] = -2.93706917090434060924;
data->test_mat_vec_Px[2] = -5.61684588113308436164;
data->test_mat_vec_Px[3] = -3.09036686615258204824;
data->test_mat_extr_triu_n = 5;

// Matrix test_mat_extr_triu_Pu
//-----------------------------
data->test_mat_extr_triu_Pu = c_malloc(sizeof(csc));
data->test_mat_extr_triu_Pu->m = 5;
data->test_mat_extr_triu_Pu->n = 5;
data->test_mat_extr_triu_Pu->nz = -1;
data->test_mat_extr_triu_Pu->nzmax = 12;
data->test_mat_extr_triu_Pu->x = c_malloc(12 * sizeof(c_float));
data->test_mat_extr_triu_Pu->x[0] = 1.14917142469609556699;
data->test_mat_extr_triu_Pu->x[1] = 0.31173688668596044593;
data->test_mat_extr_triu_Pu->x[2] = 0.23639581052096092506;
data->test_mat_extr_triu_Pu->x[3] = 1.48399997612500200717;
data->test_mat_extr_triu_Pu->x[4] = 1.88334741755529133656;
data->test_mat_extr_triu_Pu->x[5] = 0.73625464657738093344;
data->test_mat_extr_triu_Pu->x[6] = 1.30534889243361718059;
data->test_mat_extr_triu_Pu->x[7] = 1.04838550837143995587;
data->test_mat_extr_triu_Pu->x[8] = 0.89049293620335590038;
data->test_mat_extr_triu_Pu->x[9] = 0.13138286738822435584;
data->test_mat_extr_triu_Pu->x[10] = 1.30603140872840350895;
data->test_mat_extr_triu_Pu->x[11] = 1.06649871849946875635;
data->test_mat_extr_triu_Pu->i = c_malloc(12 * sizeof(c_int));
data->test_mat_extr_triu_Pu->i[0] = 0;
data->test_mat_extr_triu_Pu->i[1] = 0;
data->test_mat_extr_triu_Pu->i[2] = 1;
data->test_mat_extr_triu_Pu->i[3] = 1;
data->test_mat_extr_triu_Pu->i[4] = 2;
data->test_mat_extr_triu_Pu->i[5] = 0;
data->test_mat_extr_triu_Pu->i[6] = 1;
data->test_mat_extr_triu_Pu->i[7] = 2;
data->test_mat_extr_triu_Pu->i[8] = 0;
data->test_mat_extr_triu_Pu->i[9] = 1;
data->test_mat_extr_triu_Pu->i[10] = 2;
data->test_mat_extr_triu_Pu->i[11] = 3;
data->test_mat_extr_triu_Pu->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_mat_extr_triu_Pu->p[0] = 0;
data->test_mat_extr_triu_Pu->p[1] = 1;
data->test_mat_extr_triu_Pu->p[2] = 3;
data->test_mat_extr_triu_Pu->p[3] = 5;
data->test_mat_extr_triu_Pu->p[4] = 8;
data->test_mat_extr_triu_Pu->p[5] = 12;


// Matrix test_mat_ops_postm_diag
//-------------------------------
data->test_mat_ops_postm_diag = c_malloc(sizeof(csc));
data->test_mat_ops_postm_diag->m = 2;
data->test_mat_ops_postm_diag->n = 2;
data->test_mat_ops_postm_diag->nz = -1;
data->test_mat_ops_postm_diag->nzmax = 3;
data->test_mat_ops_postm_diag->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_postm_diag->x[0] = -0.32675369525124747794;
data->test_mat_ops_postm_diag->x[1] = -0.01993272874126506000;
data->test_mat_ops_postm_diag->x[2] = -0.32381514997721688731;
data->test_mat_ops_postm_diag->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_postm_diag->i[0] = 1;
data->test_mat_ops_postm_diag->i[1] = 0;
data->test_mat_ops_postm_diag->i[2] = 1;
data->test_mat_ops_postm_diag->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_postm_diag->p[0] = 0;
data->test_mat_ops_postm_diag->p[1] = 1;
data->test_mat_ops_postm_diag->p[2] = 3;

data->test_mat_vec_m = 5;
data->test_mat_vec_ATy_cum = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_ATy_cum[0] = 0.54953927689709880511;
data->test_mat_vec_ATy_cum[1] = -0.86657549117783638781;
data->test_mat_vec_ATy_cum[2] = 0.38566742825790689686;
data->test_mat_vec_ATy_cum[3] = -1.73288240849173202918;
data->test_mat_ops_d = c_malloc(2 * sizeof(c_float));
data->test_mat_ops_d[0] = -0.52058601890458211514;
data->test_mat_ops_d[1] = -0.41228407598445671045;

// Matrix test_mat_vec_Pu
//-----------------------
data->test_mat_vec_Pu = c_malloc(sizeof(csc));
data->test_mat_vec_Pu->m = 4;
data->test_mat_vec_Pu->n = 4;
data->test_mat_vec_Pu->nz = -1;
data->test_mat_vec_Pu->nzmax = 7;
data->test_mat_vec_Pu->x = c_malloc(7 * sizeof(c_float));
data->test_mat_vec_Pu->x[0] = 0.13057610828464238928;
data->test_mat_vec_Pu->x[1] = 0.34133213745747370549;
data->test_mat_vec_Pu->x[2] = 1.17774453342067375594;
data->test_mat_vec_Pu->x[3] = 0.62054002622175363779;
data->test_mat_vec_Pu->x[4] = 1.09906719411137832054;
data->test_mat_vec_Pu->x[5] = 0.95880376353085006169;
data->test_mat_vec_Pu->x[6] = 1.14810076109181791182;
data->test_mat_vec_Pu->i = c_malloc(7 * sizeof(c_int));
data->test_mat_vec_Pu->i[0] = 0;
data->test_mat_vec_Pu->i[1] = 0;
data->test_mat_vec_Pu->i[2] = 0;
data->test_mat_vec_Pu->i[3] = 1;
data->test_mat_vec_Pu->i[4] = 2;
data->test_mat_vec_Pu->i[5] = 1;
data->test_mat_vec_Pu->i[6] = 2;
data->test_mat_vec_Pu->p = c_malloc((4 + 1) * sizeof(c_int));
data->test_mat_vec_Pu->p[0] = 0;
data->test_mat_vec_Pu->p[1] = 1;
data->test_mat_vec_Pu->p[2] = 2;
data->test_mat_vec_Pu->p[3] = 5;
data->test_mat_vec_Pu->p[4] = 7;


// Matrix test_mat_vec_A
//----------------------
data->test_mat_vec_A = c_malloc(sizeof(csc));
data->test_mat_vec_A->m = 5;
data->test_mat_vec_A->n = 4;
data->test_mat_vec_A->nz = -1;
data->test_mat_vec_A->nzmax = 20;
data->test_mat_vec_A->x = c_malloc(20 * sizeof(c_float));
data->test_mat_vec_A->x[0] = 0.47991677274053390345;
data->test_mat_vec_A->x[1] = 0.95172309731193538429;
data->test_mat_vec_A->x[2] = 0.67045193362064059439;
data->test_mat_vec_A->x[3] = 0.73795793219687655373;
data->test_mat_vec_A->x[4] = 0.91655406371288450362;
data->test_mat_vec_A->x[5] = 0.72350160375287853043;
data->test_mat_vec_A->x[6] = 0.45051941282925100918;
data->test_mat_vec_A->x[7] = 0.71283444411253127360;
data->test_mat_vec_A->x[8] = 0.00977309358929856486;
data->test_mat_vec_A->x[9] = 0.74501556017321768355;
data->test_mat_vec_A->x[10] = 0.12898127467347464403;
data->test_mat_vec_A->x[11] = 0.86346614914123231976;
data->test_mat_vec_A->x[12] = 0.45300067986513514739;
data->test_mat_vec_A->x[13] = 0.08075442713084135526;
data->test_mat_vec_A->x[14] = 0.89957516272438764471;
data->test_mat_vec_A->x[15] = 0.26880071125487003947;
data->test_mat_vec_A->x[16] = 0.22491065065949067403;
data->test_mat_vec_A->x[17] = 0.18020616115917809097;
data->test_mat_vec_A->x[18] = 0.71492827420812754546;
data->test_mat_vec_A->x[19] = 0.35065894015425513874;
data->test_mat_vec_A->i = c_malloc(20 * sizeof(c_int));
data->test_mat_vec_A->i[0] = 0;
data->test_mat_vec_A->i[1] = 1;
data->test_mat_vec_A->i[2] = 2;
data->test_mat_vec_A->i[3] = 3;
data->test_mat_vec_A->i[4] = 4;
data->test_mat_vec_A->i[5] = 0;
data->test_mat_vec_A->i[6] = 1;
data->test_mat_vec_A->i[7] = 2;
data->test_mat_vec_A->i[8] = 3;
data->test_mat_vec_A->i[9] = 4;
data->test_mat_vec_A->i[10] = 0;
data->test_mat_vec_A->i[11] = 1;
data->test_mat_vec_A->i[12] = 2;
data->test_mat_vec_A->i[13] = 3;
data->test_mat_vec_A->i[14] = 4;
data->test_mat_vec_A->i[15] = 0;
data->test_mat_vec_A->i[16] = 1;
data->test_mat_vec_A->i[17] = 2;
data->test_mat_vec_A->i[18] = 3;
data->test_mat_vec_A->i[19] = 4;
data->test_mat_vec_A->p = c_malloc((4 + 1) * sizeof(c_int));
data->test_mat_vec_A->p[0] = 0;
data->test_mat_vec_A->p[1] = 5;
data->test_mat_vec_A->p[2] = 10;
data->test_mat_vec_A->p[3] = 15;
data->test_mat_vec_A->p[4] = 20;

data->test_vec_ops_vec_prod = 1.03166334936002934697;

// Matrix test_qpform_Pu
//----------------------
data->test_qpform_Pu = c_malloc(sizeof(csc));
data->test_qpform_Pu->m = 4;
data->test_qpform_Pu->n = 4;
data->test_qpform_Pu->nz = -1;
data->test_qpform_Pu->nzmax = 10;
data->test_qpform_Pu->x = c_malloc(10 * sizeof(c_float));
data->test_qpform_Pu->x[0] = 1.18170351919800364904;
data->test_qpform_Pu->x[1] = 0.72060399243605166575;
data->test_qpform_Pu->x[2] = 1.74757957268396779682;
data->test_qpform_Pu->x[3] = 0.10391742243694246373;
data->test_qpform_Pu->x[4] = 0.43389537719408066696;
data->test_qpform_Pu->x[5] = 1.40099805663409782142;
data->test_qpform_Pu->x[6] = 1.04663463253468602687;
data->test_qpform_Pu->x[7] = 0.70831991110513614629;
data->test_qpform_Pu->x[8] = 0.06008036517666615062;
data->test_qpform_Pu->x[9] = 1.55842412596852875062;
data->test_qpform_Pu->i = c_malloc(10 * sizeof(c_int));
data->test_qpform_Pu->i[0] = 0;
data->test_qpform_Pu->i[1] = 0;
data->test_qpform_Pu->i[2] = 1;
data->test_qpform_Pu->i[3] = 0;
data->test_qpform_Pu->i[4] = 1;
data->test_qpform_Pu->i[5] = 2;
data->test_qpform_Pu->i[6] = 0;
data->test_qpform_Pu->i[7] = 1;
data->test_qpform_Pu->i[8] = 2;
data->test_qpform_Pu->i[9] = 3;
data->test_qpform_Pu->p = c_malloc((4 + 1) * sizeof(c_int));
data->test_qpform_Pu->p[0] = 0;
data->test_qpform_Pu->p[1] = 1;
data->test_qpform_Pu->p[2] = 3;
data->test_qpform_Pu->p[3] = 6;
data->test_qpform_Pu->p[4] = 10;


// Matrix test_mat_ops_ew_square
//------------------------------
data->test_mat_ops_ew_square = c_malloc(sizeof(csc));
data->test_mat_ops_ew_square->m = 2;
data->test_mat_ops_ew_square->n = 2;
data->test_mat_ops_ew_square->nz = -1;
data->test_mat_ops_ew_square->nzmax = 3;
data->test_mat_ops_ew_square->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_ew_square->x[0] = 0.39396352511002247221;
data->test_mat_ops_ew_square->x[1] = 0.00233743956685289640;
data->test_mat_ops_ew_square->x[2] = 0.61688073209049187895;
data->test_mat_ops_ew_square->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_ew_square->i[0] = 1;
data->test_mat_ops_ew_square->i[1] = 0;
data->test_mat_ops_ew_square->i[2] = 1;
data->test_mat_ops_ew_square->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_ew_square->p[0] = 0;
data->test_mat_ops_ew_square->p[1] = 1;
data->test_mat_ops_ew_square->p[2] = 3;

data->test_vec_ops_norm2_diff = 4.05637881037691094832;

return data;

}

/* function to clean data struct */
void clean_problem_lin_alg_sols_data(lin_alg_sols_data * data){

c_free(data->test_mat_ops_ew_abs->x);
c_free(data->test_mat_ops_ew_abs->i);
c_free(data->test_mat_ops_ew_abs->p);
c_free(data->test_mat_ops_ew_abs);
c_free(data->test_mat_vec_y);
c_free(data->test_vec_ops_v2);
c_free(data->test_sp_matrix_A->x);
c_free(data->test_sp_matrix_A->i);
c_free(data->test_sp_matrix_A->p);
c_free(data->test_sp_matrix_A);
c_free(data->test_mat_vec_Px_cum);
c_free(data->test_mat_vec_Ax_cum);
c_free(data->test_mat_ops_prem_diag->x);
c_free(data->test_mat_ops_prem_diag->i);
c_free(data->test_mat_ops_prem_diag->p);
c_free(data->test_mat_ops_prem_diag);
c_free(data->test_qpform_x);
c_free(data->test_mat_vec_Ax);
c_free(data->test_mat_vec_x);
c_free(data->test_vec_ops_ew_reciprocal);
c_free(data->test_mat_extr_triu_P->x);
c_free(data->test_mat_extr_triu_P->i);
c_free(data->test_mat_extr_triu_P->p);
c_free(data->test_mat_extr_triu_P);
c_free(data->test_sp_matrix_Adns);
c_free(data->test_mat_vec_ATy);
c_free(data->test_vec_ops_v1);
c_free(data->test_vec_ops_add_scaled);
c_free(data->test_mat_ops_A->x);
c_free(data->test_mat_ops_A->i);
c_free(data->test_mat_ops_A->p);
c_free(data->test_mat_ops_A);
c_free(data->test_mat_vec_Px);
c_free(data->test_mat_extr_triu_Pu->x);
c_free(data->test_mat_extr_triu_Pu->i);
c_free(data->test_mat_extr_triu_Pu->p);
c_free(data->test_mat_extr_triu_Pu);
c_free(data->test_mat_ops_postm_diag->x);
c_free(data->test_mat_ops_postm_diag->i);
c_free(data->test_mat_ops_postm_diag->p);
c_free(data->test_mat_ops_postm_diag);
c_free(data->test_mat_vec_ATy_cum);
c_free(data->test_mat_ops_d);
c_free(data->test_mat_vec_Pu->x);
c_free(data->test_mat_vec_Pu->i);
c_free(data->test_mat_vec_Pu->p);
c_free(data->test_mat_vec_Pu);
c_free(data->test_mat_vec_A->x);
c_free(data->test_mat_vec_A->i);
c_free(data->test_mat_vec_A->p);
c_free(data->test_mat_vec_A);
c_free(data->test_qpform_Pu->x);
c_free(data->test_qpform_Pu->i);
c_free(data->test_qpform_Pu->p);
c_free(data->test_qpform_Pu);
c_free(data->test_mat_ops_ew_square->x);
c_free(data->test_mat_ops_ew_square->i);
c_free(data->test_mat_ops_ew_square->p);
c_free(data->test_mat_ops_ew_square);

c_free(data);

}

#endif
