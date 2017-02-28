#ifndef LIN_ALG_DATA_H
#define LIN_ALG_DATA_H
#include "osqp.h"


/* create data and solutions structure */
typedef struct {
c_float * test_mat_vec_ATy_cum;
c_float * test_mat_vec_ATy;
c_int test_mat_ops_n;
c_float * test_mat_vec_Px;
csc * test_sp_matrix_A;
c_float test_vec_ops_vec_prod;
c_float test_vec_ops_sc;
c_float test_vec_ops_norm2_diff;
c_float * test_mat_vec_x;
c_int test_mat_extr_triu_n;
c_float * test_vec_ops_add_scaled;
csc * test_mat_ops_A;
c_int test_qpform_n;
c_float * test_vec_ops_v2;
csc * test_mat_ops_postm_diag;
csc * test_mat_vec_A;
c_float test_vec_ops_norm2;
csc * test_mat_extr_triu_Pu;
csc * test_mat_ops_ew_square;
c_float * test_sp_matrix_Adns;
csc * test_qpform_Pu;
csc * test_mat_extr_triu_P;
c_int test_vec_ops_n;
c_float * test_mat_vec_Px_cum;
csc * test_mat_ops_prem_diag;
c_float test_qpform_value;
c_float * test_mat_vec_Ax_cum;
csc * test_mat_ops_ew_abs;
c_int test_mat_vec_m;
c_float * test_vec_ops_ew_reciprocal;
c_float * test_mat_vec_y;
c_int test_mat_vec_n;
c_float * test_mat_ops_d;
c_float * test_mat_vec_Ax;
c_float * test_qpform_x;
csc * test_mat_vec_Pu;
c_float * test_vec_ops_v1;
} lin_alg_sols_data;

/* function to define problem data */
lin_alg_sols_data *  generate_problem_lin_alg_sols_data(){

lin_alg_sols_data * data = (lin_alg_sols_data *)c_malloc(sizeof(lin_alg_sols_data));

data->test_mat_vec_ATy_cum = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_ATy_cum[0] = 0.40920559203774042878;
data->test_mat_vec_ATy_cum[1] = 2.46167313738098059162;
data->test_mat_vec_ATy_cum[2] = 0.28432994299150687878;
data->test_mat_vec_ATy_cum[3] = 1.72361740288636067220;
data->test_mat_vec_ATy = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_ATy[0] = 1.52419871225611713861;
data->test_mat_vec_ATy[1] = 1.62651582743140865617;
data->test_mat_vec_ATy[2] = 0.00115350602986293536;
data->test_mat_vec_ATy[3] = 0.65102585553477543279;
data->test_mat_ops_n = 2;
data->test_mat_vec_Px = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_Px[0] = 1.88746255466870271889;
data->test_mat_vec_Px[1] = 2.65606382997522683098;
data->test_mat_vec_Px[2] = 1.76390183543262302202;
data->test_mat_vec_Px[3] = 0.55093276066801877278;

// Matrix test_sp_matrix_A
//------------------------
data->test_sp_matrix_A = c_malloc(sizeof(csc));
data->test_sp_matrix_A->m = 5;
data->test_sp_matrix_A->n = 6;
data->test_sp_matrix_A->nz = -1;
data->test_sp_matrix_A->nzmax = 30;
data->test_sp_matrix_A->x = c_malloc(30 * sizeof(c_float));
data->test_sp_matrix_A->x[0] = -0.25836255150518888657;
data->test_sp_matrix_A->x[1] = 0.77900101455942494244;
data->test_sp_matrix_A->x[2] = -0.30730224323140520326;
data->test_sp_matrix_A->x[3] = 1.91871546580648599800;
data->test_sp_matrix_A->x[4] = 1.14778159050847849976;
data->test_sp_matrix_A->x[5] = -0.79655789359672068972;
data->test_sp_matrix_A->x[6] = 0.05226521782658790499;
data->test_sp_matrix_A->x[7] = -0.46273845336057306543;
data->test_sp_matrix_A->x[8] = 0.14121013715962507651;
data->test_sp_matrix_A->x[9] = 1.98689936929940791366;
data->test_sp_matrix_A->x[10] = -0.93424725958822263383;
data->test_sp_matrix_A->x[11] = 1.67484722471999614157;
data->test_sp_matrix_A->x[12] = -3.13672245549071382342;
data->test_sp_matrix_A->x[13] = -0.79579227798216578549;
data->test_sp_matrix_A->x[14] = 0.67729576566480820254;
data->test_sp_matrix_A->x[15] = -0.29060812439077221558;
data->test_sp_matrix_A->x[16] = -0.39845476872233054344;
data->test_sp_matrix_A->x[17] = -0.76876309514717844351;
data->test_sp_matrix_A->x[18] = -1.10873801410918182420;
data->test_sp_matrix_A->x[19] = -0.27749848572072766117;
data->test_sp_matrix_A->x[20] = 2.18913085116145644804;
data->test_sp_matrix_A->x[21] = 0.52027733930351449665;
data->test_sp_matrix_A->x[22] = -2.07538208857404882224;
data->test_sp_matrix_A->x[23] = 1.52975869323554358736;
data->test_sp_matrix_A->x[24] = 0.43635310352648060128;
data->test_sp_matrix_A->x[25] = 0.31527211510471464528;
data->test_sp_matrix_A->x[26] = -0.86197378845442573780;
data->test_sp_matrix_A->x[27] = -0.89128331409421834852;
data->test_sp_matrix_A->x[28] = 1.24763191202336276575;
data->test_sp_matrix_A->x[29] = 0.76568427027179453148;
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

data->test_vec_ops_vec_prod = 1.70676172408828130678;
data->test_vec_ops_sc = 0.22517117201734765386;
data->test_vec_ops_norm2_diff = 3.69681571023939126164;
data->test_mat_vec_x = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_x[0] = -1.11499312021837670983;
data->test_mat_vec_x[1] = 0.83515730994957182443;
data->test_mat_vec_x[2] = 0.28317643696164396250;
data->test_mat_vec_x[3] = 1.07259154735158523941;
data->test_mat_extr_triu_n = 5;
data->test_vec_ops_add_scaled = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_add_scaled[0] = -0.19368401940930776717;
data->test_vec_ops_add_scaled[1] = 0.11275901948058758562;
data->test_vec_ops_add_scaled[2] = 1.07360874122991778457;
data->test_vec_ops_add_scaled[3] = -0.56310521129860879874;
data->test_vec_ops_add_scaled[4] = -0.35754964253493276560;
data->test_vec_ops_add_scaled[5] = -0.34433790870268665696;
data->test_vec_ops_add_scaled[6] = -2.22442161083628153762;
data->test_vec_ops_add_scaled[7] = 1.11722923117461081510;
data->test_vec_ops_add_scaled[8] = 1.51552146565384782129;
data->test_vec_ops_add_scaled[9] = -1.57904779210518730892;

// Matrix test_mat_ops_A
//----------------------
data->test_mat_ops_A = c_malloc(sizeof(csc));
data->test_mat_ops_A->m = 2;
data->test_mat_ops_A->n = 2;
data->test_mat_ops_A->nz = -1;
data->test_mat_ops_A->nzmax = 3;
data->test_mat_ops_A->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_A->x[0] = 0.47190919362939987014;
data->test_mat_ops_A->x[1] = 0.00066525048367838124;
data->test_mat_ops_A->x[2] = 0.71669298035588258067;
data->test_mat_ops_A->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_A->i[0] = 0;
data->test_mat_ops_A->i[1] = 0;
data->test_mat_ops_A->i[2] = 1;
data->test_mat_ops_A->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_A->p[0] = 0;
data->test_mat_ops_A->p[1] = 1;
data->test_mat_ops_A->p[2] = 3;

data->test_qpform_n = 4;
data->test_vec_ops_v2 = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_v2[0] = 0.66170083411171243259;
data->test_vec_ops_v2[1] = -1.43096684310265653828;
data->test_vec_ops_v2[2] = 0.22245368401374787659;
data->test_vec_ops_v2[3] = -0.54627256172087146346;
data->test_vec_ops_v2[4] = 0.16509127038663051756;
data->test_vec_ops_v2[5] = -0.45409762853261631532;
data->test_vec_ops_v2[6] = -0.79858693183077311684;
data->test_vec_ops_v2[7] = 1.20358234478205283757;
data->test_vec_ops_v2[8] = 0.10163707640722317860;
data->test_vec_ops_v2[9] = 0.45210250621456449238;

// Matrix test_mat_ops_postm_diag
//-------------------------------
data->test_mat_ops_postm_diag = c_malloc(sizeof(csc));
data->test_mat_ops_postm_diag->m = 2;
data->test_mat_ops_postm_diag->n = 2;
data->test_mat_ops_postm_diag->nz = -1;
data->test_mat_ops_postm_diag->nzmax = 3;
data->test_mat_ops_postm_diag->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_postm_diag->x[0] = -0.10108795342337016654;
data->test_mat_ops_postm_diag->x[1] = 0.00076265308988789245;
data->test_mat_ops_postm_diag->x[2] = 0.82162753636362217957;
data->test_mat_ops_postm_diag->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_postm_diag->i[0] = 0;
data->test_mat_ops_postm_diag->i[1] = 0;
data->test_mat_ops_postm_diag->i[2] = 1;
data->test_mat_ops_postm_diag->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_postm_diag->p[0] = 0;
data->test_mat_ops_postm_diag->p[1] = 1;
data->test_mat_ops_postm_diag->p[2] = 3;


// Matrix test_mat_vec_A
//----------------------
data->test_mat_vec_A = c_malloc(sizeof(csc));
data->test_mat_vec_A->m = 5;
data->test_mat_vec_A->n = 4;
data->test_mat_vec_A->nz = -1;
data->test_mat_vec_A->nzmax = 20;
data->test_mat_vec_A->x = c_malloc(20 * sizeof(c_float));
data->test_mat_vec_A->x[0] = 0.39063664776589213101;
data->test_mat_vec_A->x[1] = 0.55619808748767063378;
data->test_mat_vec_A->x[2] = 0.97996889124542985172;
data->test_mat_vec_A->x[3] = 0.13265385009168328967;
data->test_mat_vec_A->x[4] = 0.01450261793377138897;
data->test_mat_vec_A->x[5] = 0.86478370459892850430;
data->test_mat_vec_A->x[6] = 0.94509424384775087002;
data->test_mat_vec_A->x[7] = 0.91850643967626555142;
data->test_mat_vec_A->x[8] = 0.62875131827179808752;
data->test_mat_vec_A->x[9] = 0.12683794318763152997;
data->test_mat_vec_A->x[10] = 0.47593636560965690840;
data->test_mat_vec_A->x[11] = 0.14777777712508910479;
data->test_mat_vec_A->x[12] = 0.33641652851586711925;
data->test_mat_vec_A->x[13] = 0.58307048222815716088;
data->test_mat_vec_A->x[14] = 0.23120718571097909066;
data->test_mat_vec_A->x[15] = 0.39769672045898218915;
data->test_mat_vec_A->x[16] = 0.16923820854714977102;
data->test_mat_vec_A->x[17] = 0.66190621979353292392;
data->test_mat_vec_A->x[18] = 0.22702084223604568347;
data->test_mat_vec_A->x[19] = 0.86058782248386023195;
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

data->test_vec_ops_norm2 = 3.42228261353745866202;

// Matrix test_mat_extr_triu_Pu
//-----------------------------
data->test_mat_extr_triu_Pu = c_malloc(sizeof(csc));
data->test_mat_extr_triu_Pu->m = 5;
data->test_mat_extr_triu_Pu->n = 5;
data->test_mat_extr_triu_Pu->nz = -1;
data->test_mat_extr_triu_Pu->nzmax = 14;
data->test_mat_extr_triu_Pu->x = c_malloc(14 * sizeof(c_float));
data->test_mat_extr_triu_Pu->x[0] = 0.98943163537754763581;
data->test_mat_extr_triu_Pu->x[1] = 0.29386444784767506988;
data->test_mat_extr_triu_Pu->x[2] = 0.86318107400112054073;
data->test_mat_extr_triu_Pu->x[3] = 1.50183149544987726287;
data->test_mat_extr_triu_Pu->x[4] = 0.95995572358299230409;
data->test_mat_extr_triu_Pu->x[5] = 0.74789665124558046827;
data->test_mat_extr_triu_Pu->x[6] = 0.03817340969884785995;
data->test_mat_extr_triu_Pu->x[7] = 1.14635171526916535001;
data->test_mat_extr_triu_Pu->x[8] = 0.06810670150504627429;
data->test_mat_extr_triu_Pu->x[9] = 0.49562741693233647311;
data->test_mat_extr_triu_Pu->x[10] = 0.98912220481432944208;
data->test_mat_extr_triu_Pu->x[11] = 0.83232965036087114274;
data->test_mat_extr_triu_Pu->x[12] = 0.70462311503843677585;
data->test_mat_extr_triu_Pu->x[13] = 0.84584756065446775608;
data->test_mat_extr_triu_Pu->i = c_malloc(14 * sizeof(c_int));
data->test_mat_extr_triu_Pu->i[0] = 0;
data->test_mat_extr_triu_Pu->i[1] = 0;
data->test_mat_extr_triu_Pu->i[2] = 1;
data->test_mat_extr_triu_Pu->i[3] = 0;
data->test_mat_extr_triu_Pu->i[4] = 1;
data->test_mat_extr_triu_Pu->i[5] = 0;
data->test_mat_extr_triu_Pu->i[6] = 1;
data->test_mat_extr_triu_Pu->i[7] = 2;
data->test_mat_extr_triu_Pu->i[8] = 3;
data->test_mat_extr_triu_Pu->i[9] = 0;
data->test_mat_extr_triu_Pu->i[10] = 1;
data->test_mat_extr_triu_Pu->i[11] = 2;
data->test_mat_extr_triu_Pu->i[12] = 3;
data->test_mat_extr_triu_Pu->i[13] = 4;
data->test_mat_extr_triu_Pu->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_mat_extr_triu_Pu->p[0] = 0;
data->test_mat_extr_triu_Pu->p[1] = 1;
data->test_mat_extr_triu_Pu->p[2] = 3;
data->test_mat_extr_triu_Pu->p[3] = 5;
data->test_mat_extr_triu_Pu->p[4] = 9;
data->test_mat_extr_triu_Pu->p[5] = 14;


// Matrix test_mat_ops_ew_square
//------------------------------
data->test_mat_ops_ew_square = c_malloc(sizeof(csc));
data->test_mat_ops_ew_square->m = 2;
data->test_mat_ops_ew_square->n = 2;
data->test_mat_ops_ew_square->nz = -1;
data->test_mat_ops_ew_square->nzmax = 3;
data->test_mat_ops_ew_square->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_ew_square->x[0] = 0.22269828703195040931;
data->test_mat_ops_ew_square->x[1] = 0.00000044255820603432;
data->test_mat_ops_ew_square->x[2] = 0.51364882809139744690;
data->test_mat_ops_ew_square->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_ew_square->i[0] = 0;
data->test_mat_ops_ew_square->i[1] = 0;
data->test_mat_ops_ew_square->i[2] = 1;
data->test_mat_ops_ew_square->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_ew_square->p[0] = 0;
data->test_mat_ops_ew_square->p[1] = 1;
data->test_mat_ops_ew_square->p[2] = 3;

data->test_sp_matrix_Adns = c_malloc(30 * sizeof(c_float));
data->test_sp_matrix_Adns[0] = -0.25836255150518888657;
data->test_sp_matrix_Adns[1] = 0.77900101455942494244;
data->test_sp_matrix_Adns[2] = -0.30730224323140520326;
data->test_sp_matrix_Adns[3] = 1.91871546580648599800;
data->test_sp_matrix_Adns[4] = 1.14778159050847849976;
data->test_sp_matrix_Adns[5] = -0.79655789359672068972;
data->test_sp_matrix_Adns[6] = 0.05226521782658790499;
data->test_sp_matrix_Adns[7] = -0.46273845336057306543;
data->test_sp_matrix_Adns[8] = 0.14121013715962507651;
data->test_sp_matrix_Adns[9] = 1.98689936929940791366;
data->test_sp_matrix_Adns[10] = -0.93424725958822263383;
data->test_sp_matrix_Adns[11] = 1.67484722471999614157;
data->test_sp_matrix_Adns[12] = -3.13672245549071382342;
data->test_sp_matrix_Adns[13] = -0.79579227798216578549;
data->test_sp_matrix_Adns[14] = 0.67729576566480820254;
data->test_sp_matrix_Adns[15] = -0.29060812439077221558;
data->test_sp_matrix_Adns[16] = -0.39845476872233054344;
data->test_sp_matrix_Adns[17] = -0.76876309514717844351;
data->test_sp_matrix_Adns[18] = -1.10873801410918182420;
data->test_sp_matrix_Adns[19] = -0.27749848572072766117;
data->test_sp_matrix_Adns[20] = 2.18913085116145644804;
data->test_sp_matrix_Adns[21] = 0.52027733930351449665;
data->test_sp_matrix_Adns[22] = -2.07538208857404882224;
data->test_sp_matrix_Adns[23] = 1.52975869323554358736;
data->test_sp_matrix_Adns[24] = 0.43635310352648060128;
data->test_sp_matrix_Adns[25] = 0.31527211510471464528;
data->test_sp_matrix_Adns[26] = -0.86197378845442573780;
data->test_sp_matrix_Adns[27] = -0.89128331409421834852;
data->test_sp_matrix_Adns[28] = 1.24763191202336276575;
data->test_sp_matrix_Adns[29] = 0.76568427027179453148;

// Matrix test_qpform_Pu
//----------------------
data->test_qpform_Pu = c_malloc(sizeof(csc));
data->test_qpform_Pu->m = 4;
data->test_qpform_Pu->n = 4;
data->test_qpform_Pu->nz = -1;
data->test_qpform_Pu->nzmax = 8;
data->test_qpform_Pu->x = c_malloc(8 * sizeof(c_float));
data->test_qpform_Pu->x[0] = 0.28253297965835200145;
data->test_qpform_Pu->x[1] = 0.64778583897013508608;
data->test_qpform_Pu->x[2] = 1.77356665347102637753;
data->test_qpform_Pu->x[3] = 1.18746889736384808600;
data->test_qpform_Pu->x[4] = 0.32214846325540058558;
data->test_qpform_Pu->x[5] = 0.67231060736685976931;
data->test_qpform_Pu->x[6] = 0.89366624179329179345;
data->test_qpform_Pu->x[7] = 1.10762076153187871697;
data->test_qpform_Pu->i = c_malloc(8 * sizeof(c_int));
data->test_qpform_Pu->i[0] = 0;
data->test_qpform_Pu->i[1] = 0;
data->test_qpform_Pu->i[2] = 1;
data->test_qpform_Pu->i[3] = 0;
data->test_qpform_Pu->i[4] = 1;
data->test_qpform_Pu->i[5] = 0;
data->test_qpform_Pu->i[6] = 1;
data->test_qpform_Pu->i[7] = 2;
data->test_qpform_Pu->p = c_malloc((4 + 1) * sizeof(c_int));
data->test_qpform_Pu->p[0] = 0;
data->test_qpform_Pu->p[1] = 1;
data->test_qpform_Pu->p[2] = 3;
data->test_qpform_Pu->p[3] = 5;
data->test_qpform_Pu->p[4] = 8;


// Matrix test_mat_extr_triu_P
//----------------------------
data->test_mat_extr_triu_P = c_malloc(sizeof(csc));
data->test_mat_extr_triu_P->m = 5;
data->test_mat_extr_triu_P->n = 5;
data->test_mat_extr_triu_P->nz = -1;
data->test_mat_extr_triu_P->nzmax = 24;
data->test_mat_extr_triu_P->x = c_malloc(24 * sizeof(c_float));
data->test_mat_extr_triu_P->x[0] = 0.98943163537754763581;
data->test_mat_extr_triu_P->x[1] = 0.29386444784767506988;
data->test_mat_extr_triu_P->x[2] = 1.50183149544987726287;
data->test_mat_extr_triu_P->x[3] = 0.74789665124558046827;
data->test_mat_extr_triu_P->x[4] = 0.49562741693233647311;
data->test_mat_extr_triu_P->x[5] = 0.29386444784767506988;
data->test_mat_extr_triu_P->x[6] = 0.86318107400112054073;
data->test_mat_extr_triu_P->x[7] = 0.95995572358299230409;
data->test_mat_extr_triu_P->x[8] = 0.03817340969884785995;
data->test_mat_extr_triu_P->x[9] = 0.98912220481432944208;
data->test_mat_extr_triu_P->x[10] = 1.50183149544987726287;
data->test_mat_extr_triu_P->x[11] = 0.95995572358299230409;
data->test_mat_extr_triu_P->x[12] = 1.14635171526916535001;
data->test_mat_extr_triu_P->x[13] = 0.83232965036087114274;
data->test_mat_extr_triu_P->x[14] = 0.74789665124558046827;
data->test_mat_extr_triu_P->x[15] = 0.03817340969884785995;
data->test_mat_extr_triu_P->x[16] = 1.14635171526916535001;
data->test_mat_extr_triu_P->x[17] = 0.06810670150504627429;
data->test_mat_extr_triu_P->x[18] = 0.70462311503843677585;
data->test_mat_extr_triu_P->x[19] = 0.49562741693233647311;
data->test_mat_extr_triu_P->x[20] = 0.98912220481432944208;
data->test_mat_extr_triu_P->x[21] = 0.83232965036087114274;
data->test_mat_extr_triu_P->x[22] = 0.70462311503843677585;
data->test_mat_extr_triu_P->x[23] = 0.84584756065446775608;
data->test_mat_extr_triu_P->i = c_malloc(24 * sizeof(c_int));
data->test_mat_extr_triu_P->i[0] = 0;
data->test_mat_extr_triu_P->i[1] = 1;
data->test_mat_extr_triu_P->i[2] = 2;
data->test_mat_extr_triu_P->i[3] = 3;
data->test_mat_extr_triu_P->i[4] = 4;
data->test_mat_extr_triu_P->i[5] = 0;
data->test_mat_extr_triu_P->i[6] = 1;
data->test_mat_extr_triu_P->i[7] = 2;
data->test_mat_extr_triu_P->i[8] = 3;
data->test_mat_extr_triu_P->i[9] = 4;
data->test_mat_extr_triu_P->i[10] = 0;
data->test_mat_extr_triu_P->i[11] = 1;
data->test_mat_extr_triu_P->i[12] = 3;
data->test_mat_extr_triu_P->i[13] = 4;
data->test_mat_extr_triu_P->i[14] = 0;
data->test_mat_extr_triu_P->i[15] = 1;
data->test_mat_extr_triu_P->i[16] = 2;
data->test_mat_extr_triu_P->i[17] = 3;
data->test_mat_extr_triu_P->i[18] = 4;
data->test_mat_extr_triu_P->i[19] = 0;
data->test_mat_extr_triu_P->i[20] = 1;
data->test_mat_extr_triu_P->i[21] = 2;
data->test_mat_extr_triu_P->i[22] = 3;
data->test_mat_extr_triu_P->i[23] = 4;
data->test_mat_extr_triu_P->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_mat_extr_triu_P->p[0] = 0;
data->test_mat_extr_triu_P->p[1] = 5;
data->test_mat_extr_triu_P->p[2] = 10;
data->test_mat_extr_triu_P->p[3] = 14;
data->test_mat_extr_triu_P->p[4] = 19;
data->test_mat_extr_triu_P->p[5] = 24;

data->test_vec_ops_n = 10;
data->test_mat_vec_Px_cum = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_Px_cum[0] = 0.77246943445032600906;
data->test_mat_vec_Px_cum[1] = 3.49122113992479876643;
data->test_mat_vec_Px_cum[2] = 2.04707827239426709554;
data->test_mat_vec_Px_cum[3] = 1.62352430801960401219;

// Matrix test_mat_ops_prem_diag
//------------------------------
data->test_mat_ops_prem_diag = c_malloc(sizeof(csc));
data->test_mat_ops_prem_diag->m = 2;
data->test_mat_ops_prem_diag->n = 2;
data->test_mat_ops_prem_diag->nz = -1;
data->test_mat_ops_prem_diag->nzmax = 3;
data->test_mat_ops_prem_diag->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_prem_diag->x[0] = -0.10108795342337016654;
data->test_mat_ops_prem_diag->x[1] = -0.00014250370795226884;
data->test_mat_ops_prem_diag->x[2] = 0.82162753636362217957;
data->test_mat_ops_prem_diag->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_prem_diag->i[0] = 0;
data->test_mat_ops_prem_diag->i[1] = 0;
data->test_mat_ops_prem_diag->i[2] = 1;
data->test_mat_ops_prem_diag->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_prem_diag->p[0] = 0;
data->test_mat_ops_prem_diag->p[1] = 1;
data->test_mat_ops_prem_diag->p[2] = 3;

data->test_qpform_value = -1.31201171314690112624;
data->test_mat_vec_Ax_cum = c_malloc(5 * sizeof(c_float));
data->test_mat_vec_Ax_cum[0] = 1.57006714973520367096;
data->test_mat_vec_Ax_cum[1] = 1.58516869729587095428;
data->test_mat_vec_Ax_cum[2] = 1.24852841736790254323;
data->test_mat_vec_Ax_cum[3] = -0.52473677271544616474;
data->test_mat_vec_Ax_cum[4] = 1.02091991363016654226;

// Matrix test_mat_ops_ew_abs
//---------------------------
data->test_mat_ops_ew_abs = c_malloc(sizeof(csc));
data->test_mat_ops_ew_abs->m = 2;
data->test_mat_ops_ew_abs->n = 2;
data->test_mat_ops_ew_abs->nz = -1;
data->test_mat_ops_ew_abs->nzmax = 3;
data->test_mat_ops_ew_abs->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_ew_abs->x[0] = 0.47190919362939987014;
data->test_mat_ops_ew_abs->x[1] = 0.00066525048367838124;
data->test_mat_ops_ew_abs->x[2] = 0.71669298035588258067;
data->test_mat_ops_ew_abs->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_ew_abs->i[0] = 0;
data->test_mat_ops_ew_abs->i[1] = 0;
data->test_mat_ops_ew_abs->i[2] = 1;
data->test_mat_ops_ew_abs->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_ew_abs->p[0] = 0;
data->test_mat_ops_ew_abs->p[1] = 1;
data->test_mat_ops_ew_abs->p[2] = 3;

data->test_mat_vec_m = 5;
data->test_vec_ops_ew_reciprocal = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_ew_reciprocal[0] = -2.91817463066192184229;
data->test_vec_ops_ew_reciprocal[1] = 2.29900119544087955248;
data->test_vec_ops_ew_reciprocal[2] = 0.97702182956170546824;
data->test_vec_ops_ew_reciprocal[3] = -2.27220890784785245486;
data->test_vec_ops_ew_reciprocal[4] = -2.53341936481736906828;
data->test_vec_ops_ew_reciprocal[5] = -4.13072567905244092401;
data->test_vec_ops_ew_reciprocal[6] = -0.48909253811331271367;
data->test_vec_ops_ew_reciprocal[7] = 1.18172972486324279195;
data->test_vec_ops_ew_reciprocal[8] = 0.66995582549388521532;
data->test_vec_ops_ew_reciprocal[9] = -0.59493770718754035443;
data->test_mat_vec_y = c_malloc(5 * sizeof(c_float));
data->test_mat_vec_y[0] = 0.72205378707070888566;
data->test_mat_vec_y[1] = 1.19265271561753793961;
data->test_mat_vec_y[2] = 0.76886937143308631271;
data->test_mat_vec_y[3] = -1.31054736019775863731;
data->test_mat_vec_y[4] = -0.05737105378010095430;
data->test_mat_vec_n = 4;
data->test_mat_ops_d = c_malloc(2 * sizeof(c_float));
data->test_mat_ops_d[0] = -0.21421060404844888270;
data->test_mat_ops_d[1] = 1.14641493482416012561;
data->test_mat_vec_Ax = c_malloc(5 * sizeof(c_float));
data->test_mat_vec_Ax[0] = 0.84801336266449478529;
data->test_mat_vec_Ax[1] = 0.39251598167833301467;
data->test_mat_vec_Ax[2] = 0.47965904593481623053;
data->test_mat_vec_Ax[3] = 0.78581058748231247257;
data->test_mat_vec_Ax[4] = 1.07829096741026742023;
data->test_qpform_x = c_malloc(4 * sizeof(c_float));
data->test_qpform_x[0] = 1.13116398125624906257;
data->test_qpform_x[1] = -1.35936907428097053518;
data->test_qpform_x[2] = -1.19548906977814639596;
data->test_qpform_x[3] = 0.59212394334220841419;

// Matrix test_mat_vec_Pu
//-----------------------
data->test_mat_vec_Pu = c_malloc(sizeof(csc));
data->test_mat_vec_Pu->m = 4;
data->test_mat_vec_Pu->n = 4;
data->test_mat_vec_Pu->nz = -1;
data->test_mat_vec_Pu->nzmax = 8;
data->test_mat_vec_Pu->x = c_malloc(8 * sizeof(c_float));
data->test_mat_vec_Pu->x[0] = 0.48199843608235404258;
data->test_mat_vec_Pu->x[1] = 1.62500402878125527195;
data->test_mat_vec_Pu->x[2] = 0.48938396862170119306;
data->test_mat_vec_Pu->x[3] = 0.83042227596686624125;
data->test_mat_vec_Pu->x[4] = 1.25521782469212839217;
data->test_mat_vec_Pu->x[5] = 1.49283178624767298714;
data->test_mat_vec_Pu->x[6] = 1.50665774260719409483;
data->test_mat_vec_Pu->x[7] = 0.25833939319903143073;
data->test_mat_vec_Pu->i = c_malloc(8 * sizeof(c_int));
data->test_mat_vec_Pu->i[0] = 0;
data->test_mat_vec_Pu->i[1] = 1;
data->test_mat_vec_Pu->i[2] = 0;
data->test_mat_vec_Pu->i[3] = 1;
data->test_mat_vec_Pu->i[4] = 0;
data->test_mat_vec_Pu->i[5] = 1;
data->test_mat_vec_Pu->i[6] = 2;
data->test_mat_vec_Pu->i[7] = 3;
data->test_mat_vec_Pu->p = c_malloc((4 + 1) * sizeof(c_int));
data->test_mat_vec_Pu->p[0] = 0;
data->test_mat_vec_Pu->p[1] = 0;
data->test_mat_vec_Pu->p[2] = 2;
data->test_mat_vec_Pu->p[3] = 4;
data->test_mat_vec_Pu->p[4] = 8;

data->test_vec_ops_v1 = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_v1[0] = -0.34267997175109859986;
data->test_vec_ops_v1[1] = 0.43497150065997680635;
data->test_vec_ops_v1[2] = 1.02351858448096555421;
data->test_vec_ops_v1[3] = -0.44010037833500131876;
data->test_vec_ops_v1[4] = -0.39472343737772319638;
data->test_vec_ops_v1[5] = -0.24208821347569925431;
data->test_vec_ops_v1[6] = -2.04460285543820852183;
data->test_vec_ops_v1[7] = 0.84621718398064860178;
data->test_vec_ops_v1[8] = 1.49263572603881655709;
data->test_vec_ops_v1[9] = -1.68084824330150106597;

return data;

}

/* function to clean data struct */
void clean_problem_lin_alg_sols_data(lin_alg_sols_data * data){

c_free(data->test_mat_vec_ATy_cum);
c_free(data->test_mat_vec_ATy);
c_free(data->test_mat_vec_Px);
c_free(data->test_sp_matrix_A->x);
c_free(data->test_sp_matrix_A->i);
c_free(data->test_sp_matrix_A->p);
c_free(data->test_sp_matrix_A);
c_free(data->test_mat_vec_x);
c_free(data->test_vec_ops_add_scaled);
c_free(data->test_mat_ops_A->x);
c_free(data->test_mat_ops_A->i);
c_free(data->test_mat_ops_A->p);
c_free(data->test_mat_ops_A);
c_free(data->test_vec_ops_v2);
c_free(data->test_mat_ops_postm_diag->x);
c_free(data->test_mat_ops_postm_diag->i);
c_free(data->test_mat_ops_postm_diag->p);
c_free(data->test_mat_ops_postm_diag);
c_free(data->test_mat_vec_A->x);
c_free(data->test_mat_vec_A->i);
c_free(data->test_mat_vec_A->p);
c_free(data->test_mat_vec_A);
c_free(data->test_mat_extr_triu_Pu->x);
c_free(data->test_mat_extr_triu_Pu->i);
c_free(data->test_mat_extr_triu_Pu->p);
c_free(data->test_mat_extr_triu_Pu);
c_free(data->test_mat_ops_ew_square->x);
c_free(data->test_mat_ops_ew_square->i);
c_free(data->test_mat_ops_ew_square->p);
c_free(data->test_mat_ops_ew_square);
c_free(data->test_sp_matrix_Adns);
c_free(data->test_qpform_Pu->x);
c_free(data->test_qpform_Pu->i);
c_free(data->test_qpform_Pu->p);
c_free(data->test_qpform_Pu);
c_free(data->test_mat_extr_triu_P->x);
c_free(data->test_mat_extr_triu_P->i);
c_free(data->test_mat_extr_triu_P->p);
c_free(data->test_mat_extr_triu_P);
c_free(data->test_mat_vec_Px_cum);
c_free(data->test_mat_ops_prem_diag->x);
c_free(data->test_mat_ops_prem_diag->i);
c_free(data->test_mat_ops_prem_diag->p);
c_free(data->test_mat_ops_prem_diag);
c_free(data->test_mat_vec_Ax_cum);
c_free(data->test_mat_ops_ew_abs->x);
c_free(data->test_mat_ops_ew_abs->i);
c_free(data->test_mat_ops_ew_abs->p);
c_free(data->test_mat_ops_ew_abs);
c_free(data->test_vec_ops_ew_reciprocal);
c_free(data->test_mat_vec_y);
c_free(data->test_mat_ops_d);
c_free(data->test_mat_vec_Ax);
c_free(data->test_qpform_x);
c_free(data->test_mat_vec_Pu->x);
c_free(data->test_mat_vec_Pu->i);
c_free(data->test_mat_vec_Pu->p);
c_free(data->test_mat_vec_Pu);
c_free(data->test_vec_ops_v1);

c_free(data);

}

#endif
