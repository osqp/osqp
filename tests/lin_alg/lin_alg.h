#ifndef LIN_ALG_DATA_H
#define LIN_ALG_DATA_H
#include "osqp.h"


/* create data and solutions structure */
typedef struct {
csc * test_mat_ops_ew_square;
c_float * test_vec_ops_v2;
csc * test_mat_vec_Pu;
csc * test_mat_ops_A;
csc * test_mat_ops_ew_abs;
c_float * test_mat_vec_ATy;
c_float test_vec_ops_norm2_diff;
csc * test_mat_vec_A;
c_float * test_mat_ops_d;
c_int test_qpform_n;
c_float * test_sp_matrix_Adns;
csc * test_mat_ops_prem_diag;
c_float * test_qpform_x;
csc * test_mat_extr_triu_P;
c_float * test_vec_ops_ew_reciprocal;
csc * test_mat_extr_triu_Pu;
c_float * test_mat_vec_ATy_cum;
c_int test_mat_vec_n;
c_float * test_mat_vec_Px;
c_float test_qpform_value;
c_int test_mat_vec_m;
c_float * test_mat_vec_y;
c_float * test_vec_ops_add_scaled;
c_float * test_mat_vec_x;
c_float test_vec_ops_norm2;
c_int test_mat_ops_n;
c_float * test_mat_vec_Ax_cum;
c_float * test_mat_vec_Ax;
c_float * test_mat_vec_Px_cum;
c_int test_vec_ops_n;
csc * test_sp_matrix_A;
csc * test_qpform_Pu;
c_float test_vec_ops_sc;
c_float test_vec_ops_vec_prod;
c_int test_mat_extr_triu_n;
csc * test_mat_ops_postm_diag;
c_float * test_vec_ops_v1;
} lin_alg_sols_data;

/* function to define problem data */
lin_alg_sols_data *  generate_problem_lin_alg_sols_data(){

lin_alg_sols_data * data = (lin_alg_sols_data *)c_malloc(sizeof(lin_alg_sols_data));


// Matrix test_mat_ops_ew_square
//------------------------------
data->test_mat_ops_ew_square = c_malloc(sizeof(csc));
data->test_mat_ops_ew_square->m = 2;
data->test_mat_ops_ew_square->n = 2;
data->test_mat_ops_ew_square->nz = -1;
data->test_mat_ops_ew_square->nzmax = 3;
data->test_mat_ops_ew_square->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_ew_square->x[0] = 0.00174704854580486676;
data->test_mat_ops_ew_square->x[1] = 0.09813090563394967492;
data->test_mat_ops_ew_square->x[2] = 0.54523420353140394923;
data->test_mat_ops_ew_square->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_ew_square->i[0] = 0;
data->test_mat_ops_ew_square->i[1] = 1;
data->test_mat_ops_ew_square->i[2] = 0;
data->test_mat_ops_ew_square->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_ew_square->p[0] = 0;
data->test_mat_ops_ew_square->p[1] = 2;
data->test_mat_ops_ew_square->p[2] = 3;

data->test_vec_ops_v2 = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_v2[0] = 0.52429643001034864636;
data->test_vec_ops_v2[1] = 0.73527957606520255585;
data->test_vec_ops_v2[2] = -0.65325026779203565486;
data->test_vec_ops_v2[3] = 0.84245628157134011538;
data->test_vec_ops_v2[4] = -0.38151648176508617949;
data->test_vec_ops_v2[5] = 0.06648900914614486179;
data->test_vec_ops_v2[6] = -1.09873894699605645364;
data->test_vec_ops_v2[7] = 1.58448705639567632986;
data->test_vec_ops_v2[8] = -2.65944945638348828609;
data->test_vec_ops_v2[9] = -0.09145262289065814176;

// Matrix test_mat_vec_Pu
//-----------------------
data->test_mat_vec_Pu = c_malloc(sizeof(csc));
data->test_mat_vec_Pu->m = 4;
data->test_mat_vec_Pu->n = 4;
data->test_mat_vec_Pu->nz = -1;
data->test_mat_vec_Pu->nzmax = 9;
data->test_mat_vec_Pu->x = c_malloc(9 * sizeof(c_float));
data->test_mat_vec_Pu->x[0] = 0.43857994482138584758;
data->test_mat_vec_Pu->x[1] = 0.83651764852853427445;
data->test_mat_vec_Pu->x[2] = 0.97447737597704364720;
data->test_mat_vec_Pu->x[3] = 1.00661727616937901608;
data->test_mat_vec_Pu->x[4] = 1.01082326817688716858;
data->test_mat_vec_Pu->x[5] = 0.01827003127010540240;
data->test_mat_vec_Pu->x[6] = 0.38658285361775701627;
data->test_mat_vec_Pu->x[7] = 0.54587270844450219709;
data->test_mat_vec_Pu->x[8] = 1.24654467104204758066;
data->test_mat_vec_Pu->i = c_malloc(9 * sizeof(c_int));
data->test_mat_vec_Pu->i[0] = 0;
data->test_mat_vec_Pu->i[1] = 0;
data->test_mat_vec_Pu->i[2] = 0;
data->test_mat_vec_Pu->i[3] = 1;
data->test_mat_vec_Pu->i[4] = 2;
data->test_mat_vec_Pu->i[5] = 0;
data->test_mat_vec_Pu->i[6] = 1;
data->test_mat_vec_Pu->i[7] = 2;
data->test_mat_vec_Pu->i[8] = 3;
data->test_mat_vec_Pu->p = c_malloc((4 + 1) * sizeof(c_int));
data->test_mat_vec_Pu->p[0] = 0;
data->test_mat_vec_Pu->p[1] = 1;
data->test_mat_vec_Pu->p[2] = 2;
data->test_mat_vec_Pu->p[3] = 5;
data->test_mat_vec_Pu->p[4] = 9;


// Matrix test_mat_ops_A
//----------------------
data->test_mat_ops_A = c_malloc(sizeof(csc));
data->test_mat_ops_A->m = 2;
data->test_mat_ops_A->n = 2;
data->test_mat_ops_A->nz = -1;
data->test_mat_ops_A->nzmax = 3;
data->test_mat_ops_A->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_A->x[0] = 0.04179770981531005791;
data->test_mat_ops_A->x[1] = 0.31325852842971357859;
data->test_mat_ops_A->x[2] = 0.73839975862090034830;
data->test_mat_ops_A->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_A->i[0] = 0;
data->test_mat_ops_A->i[1] = 1;
data->test_mat_ops_A->i[2] = 0;
data->test_mat_ops_A->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_A->p[0] = 0;
data->test_mat_ops_A->p[1] = 2;
data->test_mat_ops_A->p[2] = 3;


// Matrix test_mat_ops_ew_abs
//---------------------------
data->test_mat_ops_ew_abs = c_malloc(sizeof(csc));
data->test_mat_ops_ew_abs->m = 2;
data->test_mat_ops_ew_abs->n = 2;
data->test_mat_ops_ew_abs->nz = -1;
data->test_mat_ops_ew_abs->nzmax = 3;
data->test_mat_ops_ew_abs->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_ew_abs->x[0] = 0.04179770981531005791;
data->test_mat_ops_ew_abs->x[1] = 0.31325852842971357859;
data->test_mat_ops_ew_abs->x[2] = 0.73839975862090034830;
data->test_mat_ops_ew_abs->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_ew_abs->i[0] = 0;
data->test_mat_ops_ew_abs->i[1] = 1;
data->test_mat_ops_ew_abs->i[2] = 0;
data->test_mat_ops_ew_abs->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_ew_abs->p[0] = 0;
data->test_mat_ops_ew_abs->p[1] = 2;
data->test_mat_ops_ew_abs->p[2] = 3;

data->test_mat_vec_ATy = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_ATy[0] = 0.91238056102500841860;
data->test_mat_vec_ATy[1] = 0.27390726630491868399;
data->test_mat_vec_ATy[2] = -0.30628345735146567108;
data->test_mat_vec_ATy[3] = -0.24721820388368365151;
data->test_vec_ops_norm2_diff = 4.12428039060981355135;

// Matrix test_mat_vec_A
//----------------------
data->test_mat_vec_A = c_malloc(sizeof(csc));
data->test_mat_vec_A->m = 5;
data->test_mat_vec_A->n = 4;
data->test_mat_vec_A->nz = -1;
data->test_mat_vec_A->nzmax = 20;
data->test_mat_vec_A->x = c_malloc(20 * sizeof(c_float));
data->test_mat_vec_A->x[0] = 0.62935972282958219104;
data->test_mat_vec_A->x[1] = 0.54790778009037666152;
data->test_mat_vec_A->x[2] = 0.18662714558648019203;
data->test_mat_vec_A->x[3] = 0.48926616699485914186;
data->test_mat_vec_A->x[4] = 0.91391547716513199529;
data->test_mat_vec_A->x[5] = 0.24581116390413071393;
data->test_mat_vec_A->x[6] = 0.54019151566713086154;
data->test_mat_vec_A->x[7] = 0.60844215780504962154;
data->test_mat_vec_A->x[8] = 0.41973546130682071187;
data->test_mat_vec_A->x[9] = 0.82624982843415795131;
data->test_mat_vec_A->x[10] = 0.11058314779616107426;
data->test_mat_vec_A->x[11] = 0.27405925253605367686;
data->test_mat_vec_A->x[12] = 0.59125735265354584236;
data->test_mat_vec_A->x[13] = 0.34623790945514643091;
data->test_mat_vec_A->x[14] = 0.29517230521974247015;
data->test_mat_vec_A->x[15] = 0.69952061979054813712;
data->test_mat_vec_A->x[16] = 0.01025003939959090449;
data->test_mat_vec_A->x[17] = 0.17671216158442604183;
data->test_mat_vec_A->x[18] = 0.62356318456213077894;
data->test_mat_vec_A->x[19] = 0.26377852928001976895;
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

data->test_mat_ops_d = c_malloc(2 * sizeof(c_float));
data->test_mat_ops_d[0] = -2.03346654612261357187;
data->test_mat_ops_d[1] = -1.14533805578424119354;
data->test_qpform_n = 4;
data->test_sp_matrix_Adns = c_malloc(30 * sizeof(c_float));
data->test_sp_matrix_Adns[0] = -1.42121722730412680669;
data->test_sp_matrix_Adns[1] = 0.37044453666323423624;
data->test_sp_matrix_Adns[2] = -0.31350819699340887192;
data->test_sp_matrix_Adns[3] = 1.61134077957371735224;
data->test_sp_matrix_Adns[4] = -0.37566942307899364728;
data->test_sp_matrix_Adns[5] = -0.15349519567694913658;
data->test_sp_matrix_Adns[6] = 1.35963386267259611628;
data->test_sp_matrix_Adns[7] = 0.77101173806941147859;
data->test_sp_matrix_Adns[8] = 0.04797059186818837528;
data->test_sp_matrix_Adns[9] = -0.07447076289398098237;
data->test_sp_matrix_Adns[10] = -0.26905696021601349655;
data->test_sp_matrix_Adns[11] = 0.50185720678131329198;
data->test_sp_matrix_Adns[12] = -1.86809065456376344194;
data->test_sp_matrix_Adns[13] = -0.82913528901477873134;
data->test_sp_matrix_Adns[14] = 0.43349633007657084605;
data->test_sp_matrix_Adns[15] = 2.23136678888660444642;
data->test_sp_matrix_Adns[16] = -0.84421370382986182790;
data->test_sp_matrix_Adns[17] = 1.73118466598046794047;
data->test_sp_matrix_Adns[18] = 0.08771021840833141681;
data->test_sp_matrix_Adns[19] = 1.27837923027186817215;
data->test_sp_matrix_Adns[20] = -2.43476757652104414120;
data->test_sp_matrix_Adns[21] = 0.00000976147159609348;
data->test_sp_matrix_Adns[22] = 2.46767801057341262805;
data->test_sp_matrix_Adns[23] = 1.00036588655069502707;
data->test_sp_matrix_Adns[24] = -0.63467930517936232970;
data->test_sp_matrix_Adns[25] = 0.11272650481664892030;
data->test_sp_matrix_Adns[26] = 0.54235257214902909961;
data->test_sp_matrix_Adns[27] = -0.33567733852384745719;
data->test_sp_matrix_Adns[28] = -0.38109251751534994890;
data->test_sp_matrix_Adns[29] = 0.50839624268343175384;

// Matrix test_mat_ops_prem_diag
//------------------------------
data->test_mat_ops_prem_diag = c_malloc(sizeof(csc));
data->test_mat_ops_prem_diag->m = 2;
data->test_mat_ops_prem_diag->n = 2;
data->test_mat_ops_prem_diag->nz = -1;
data->test_mat_ops_prem_diag->nzmax = 3;
data->test_mat_ops_prem_diag->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_prem_diag->x[0] = -0.08499424461397381281;
data->test_mat_ops_prem_diag->x[1] = -0.35878691390952061058;
data->test_mat_ops_prem_diag->x[2] = -1.50151120682061378631;
data->test_mat_ops_prem_diag->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_prem_diag->i[0] = 0;
data->test_mat_ops_prem_diag->i[1] = 1;
data->test_mat_ops_prem_diag->i[2] = 0;
data->test_mat_ops_prem_diag->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_prem_diag->p[0] = 0;
data->test_mat_ops_prem_diag->p[1] = 2;
data->test_mat_ops_prem_diag->p[2] = 3;

data->test_qpform_x = c_malloc(4 * sizeof(c_float));
data->test_qpform_x[0] = 0.35913333179380452220;
data->test_qpform_x[1] = 0.62222041446793285857;
data->test_qpform_x[2] = 0.96078194484322809732;
data->test_qpform_x[3] = 0.75837034716676965385;

// Matrix test_mat_extr_triu_P
//----------------------------
data->test_mat_extr_triu_P = c_malloc(sizeof(csc));
data->test_mat_extr_triu_P->m = 5;
data->test_mat_extr_triu_P->n = 5;
data->test_mat_extr_triu_P->nz = -1;
data->test_mat_extr_triu_P->nzmax = 24;
data->test_mat_extr_triu_P->x = c_malloc(24 * sizeof(c_float));
data->test_mat_extr_triu_P->x[0] = 1.25636243709775197175;
data->test_mat_extr_triu_P->x[1] = 0.46301878153105946456;
data->test_mat_extr_triu_P->x[2] = 0.86736337930038920341;
data->test_mat_extr_triu_P->x[3] = 0.35417242096340495472;
data->test_mat_extr_triu_P->x[4] = 0.34609815153014733546;
data->test_mat_extr_triu_P->x[5] = 0.46301878153105946456;
data->test_mat_extr_triu_P->x[6] = 1.56060373790060746835;
data->test_mat_extr_triu_P->x[7] = 0.82745335206725068034;
data->test_mat_extr_triu_P->x[8] = 1.12964779320375763305;
data->test_mat_extr_triu_P->x[9] = 0.83747221011704786608;
data->test_mat_extr_triu_P->x[10] = 0.86736337930038920341;
data->test_mat_extr_triu_P->x[11] = 0.82745335206725068034;
data->test_mat_extr_triu_P->x[12] = 1.38118030249938827936;
data->test_mat_extr_triu_P->x[13] = 1.01660609302193694070;
data->test_mat_extr_triu_P->x[14] = 0.35417242096340495472;
data->test_mat_extr_triu_P->x[15] = 1.12964779320375763305;
data->test_mat_extr_triu_P->x[16] = 1.38118030249938827936;
data->test_mat_extr_triu_P->x[17] = 0.95107793820572439358;
data->test_mat_extr_triu_P->x[18] = 0.85846834258151383246;
data->test_mat_extr_triu_P->x[19] = 0.34609815153014733546;
data->test_mat_extr_triu_P->x[20] = 0.83747221011704786608;
data->test_mat_extr_triu_P->x[21] = 1.01660609302193694070;
data->test_mat_extr_triu_P->x[22] = 0.85846834258151383246;
data->test_mat_extr_triu_P->x[23] = 1.07939560863581207606;
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

data->test_vec_ops_ew_reciprocal = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_ew_reciprocal[0] = 4.62714454745154402104;
data->test_vec_ops_ew_reciprocal[1] = -0.53803579889279495863;
data->test_vec_ops_ew_reciprocal[2] = -2.38483351492933337568;
data->test_vec_ops_ew_reciprocal[3] = -7.55692831885867999375;
data->test_vec_ops_ew_reciprocal[4] = -25.27151737626683214444;
data->test_vec_ops_ew_reciprocal[5] = 3.06745235659207171608;
data->test_vec_ops_ew_reciprocal[6] = -0.49011846463386282702;
data->test_vec_ops_ew_reciprocal[7] = 21.61903989144494175889;
data->test_vec_ops_ew_reciprocal[8] = -1.47563234304949930653;
data->test_vec_ops_ew_reciprocal[9] = -0.69471508096159595436;

// Matrix test_mat_extr_triu_Pu
//-----------------------------
data->test_mat_extr_triu_Pu = c_malloc(sizeof(csc));
data->test_mat_extr_triu_Pu->m = 5;
data->test_mat_extr_triu_Pu->n = 5;
data->test_mat_extr_triu_Pu->nz = -1;
data->test_mat_extr_triu_Pu->nzmax = 14;
data->test_mat_extr_triu_Pu->x = c_malloc(14 * sizeof(c_float));
data->test_mat_extr_triu_Pu->x[0] = 1.25636243709775197175;
data->test_mat_extr_triu_Pu->x[1] = 0.46301878153105946456;
data->test_mat_extr_triu_Pu->x[2] = 1.56060373790060746835;
data->test_mat_extr_triu_Pu->x[3] = 0.86736337930038920341;
data->test_mat_extr_triu_Pu->x[4] = 0.82745335206725068034;
data->test_mat_extr_triu_Pu->x[5] = 0.35417242096340495472;
data->test_mat_extr_triu_Pu->x[6] = 1.12964779320375763305;
data->test_mat_extr_triu_Pu->x[7] = 1.38118030249938827936;
data->test_mat_extr_triu_Pu->x[8] = 0.95107793820572439358;
data->test_mat_extr_triu_Pu->x[9] = 0.34609815153014733546;
data->test_mat_extr_triu_Pu->x[10] = 0.83747221011704786608;
data->test_mat_extr_triu_Pu->x[11] = 1.01660609302193694070;
data->test_mat_extr_triu_Pu->x[12] = 0.85846834258151383246;
data->test_mat_extr_triu_Pu->x[13] = 1.07939560863581207606;
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

data->test_mat_vec_ATy_cum = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_ATy_cum[0] = 1.54457232845005698607;
data->test_mat_vec_ATy_cum[1] = 1.05310365633043190314;
data->test_mat_vec_ATy_cum[2] = -0.51389114827022919219;
data->test_mat_vec_ATy_cum[3] = -0.72215413479291001853;
data->test_mat_vec_n = 4;
data->test_mat_vec_Px = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_Px[0] = 0.71809207021169918495;
data->test_mat_vec_Px[1] = 0.13625599490454512464;
data->test_mat_vec_Px[2] = 0.93129987478695408498;
data->test_mat_vec_Px[3] = -0.39258209895274831513;
data->test_qpform_value = 2.42504111102000630140;
data->test_mat_vec_m = 5;
data->test_mat_vec_y = c_malloc(5 * sizeof(c_float));
data->test_mat_vec_y[0] = -0.05053769326586431826;
data->test_mat_vec_y[1] = 1.96231586074821984234;
data->test_mat_vec_y[2] = -1.49139808547822494411;
data->test_mat_vec_y[3] = -0.02272573072923615017;
data->test_mat_vec_y[4] = 0.17340019433320591480;
data->test_vec_ops_add_scaled = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_add_scaled[0] = 0.58056473362002314342;
data->test_vec_ops_add_scaled[1] = -1.34750513760993451129;
data->test_vec_ops_add_scaled[2] = -0.87340355029713001578;
data->test_vec_ops_add_scaled[3] = 0.45327897927848359583;
data->test_vec_ops_add_scaled[4] = -0.30476982581707007247;
data->test_vec_ops_add_scaled[5] = 0.37222124716461457705;
data->test_vec_ops_add_scaled[6] = -2.80407803161436142148;
data->test_vec_ops_add_scaled[7] = 1.14766353998552950166;
data->test_vec_ops_add_scaled[8] = -2.52631103309176818783;
data->test_vec_ops_add_scaled[9] = -1.50300953784288382487;
data->test_mat_vec_x = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_x[0] = 0.63219176742504867850;
data->test_mat_vec_x[1] = 0.77919639002551310814;
data->test_mat_vec_x[2] = -0.20760769091876352110;
data->test_mat_vec_x[3] = -0.47493593090922636701;
data->test_vec_ops_norm2 = 3.24015648363328434556;
data->test_mat_ops_n = 2;
data->test_mat_vec_Ax_cum = c_malloc(5 * sizeof(c_float));
data->test_mat_vec_Ax_cum[0] = 0.18368812507906395748;
data->test_mat_vec_Ax_cum[1] = 2.66784900696169557932;
data->test_mat_vec_Ax_cum[2] = -1.10599453624761268067;
data->test_mat_vec_Ax_cum[3] = 0.24560645390923083187;
data->test_mat_vec_Ax_cum[4] = 1.20842297662712083728;
data->test_mat_vec_Ax = c_malloc(5 * sizeof(c_float));
data->test_mat_vec_Ax[0] = 0.23422581834492828268;
data->test_mat_vec_Ax[1] = 0.70553314621347551494;
data->test_mat_vec_Ax[2] = 0.38540354923061226344;
data->test_mat_vec_Ax[3] = 0.26833218463846697510;
data->test_mat_vec_Ax[4] = 1.03502278229391486697;
data->test_mat_vec_Px_cum = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_Px_cum[0] = 1.35028383763674786344;
data->test_mat_vec_Px_cum[1] = 0.91545238493005820501;
data->test_mat_vec_Px_cum[2] = 0.72369218386819056388;
data->test_mat_vec_Px_cum[3] = -0.86751802986197468215;
data->test_vec_ops_n = 10;

// Matrix test_sp_matrix_A
//------------------------
data->test_sp_matrix_A = c_malloc(sizeof(csc));
data->test_sp_matrix_A->m = 5;
data->test_sp_matrix_A->n = 6;
data->test_sp_matrix_A->nz = -1;
data->test_sp_matrix_A->nzmax = 30;
data->test_sp_matrix_A->x = c_malloc(30 * sizeof(c_float));
data->test_sp_matrix_A->x[0] = -1.42121722730412680669;
data->test_sp_matrix_A->x[1] = 0.37044453666323423624;
data->test_sp_matrix_A->x[2] = -0.31350819699340887192;
data->test_sp_matrix_A->x[3] = 1.61134077957371735224;
data->test_sp_matrix_A->x[4] = -0.37566942307899364728;
data->test_sp_matrix_A->x[5] = -0.15349519567694913658;
data->test_sp_matrix_A->x[6] = 1.35963386267259611628;
data->test_sp_matrix_A->x[7] = 0.77101173806941147859;
data->test_sp_matrix_A->x[8] = 0.04797059186818837528;
data->test_sp_matrix_A->x[9] = -0.07447076289398098237;
data->test_sp_matrix_A->x[10] = -0.26905696021601349655;
data->test_sp_matrix_A->x[11] = 0.50185720678131329198;
data->test_sp_matrix_A->x[12] = -1.86809065456376344194;
data->test_sp_matrix_A->x[13] = -0.82913528901477873134;
data->test_sp_matrix_A->x[14] = 0.43349633007657084605;
data->test_sp_matrix_A->x[15] = 2.23136678888660444642;
data->test_sp_matrix_A->x[16] = -0.84421370382986182790;
data->test_sp_matrix_A->x[17] = 1.73118466598046794047;
data->test_sp_matrix_A->x[18] = 0.08771021840833141681;
data->test_sp_matrix_A->x[19] = 1.27837923027186817215;
data->test_sp_matrix_A->x[20] = -2.43476757652104414120;
data->test_sp_matrix_A->x[21] = 0.00000976147159609348;
data->test_sp_matrix_A->x[22] = 2.46767801057341262805;
data->test_sp_matrix_A->x[23] = 1.00036588655069502707;
data->test_sp_matrix_A->x[24] = -0.63467930517936232970;
data->test_sp_matrix_A->x[25] = 0.11272650481664892030;
data->test_sp_matrix_A->x[26] = 0.54235257214902909961;
data->test_sp_matrix_A->x[27] = -0.33567733852384745719;
data->test_sp_matrix_A->x[28] = -0.38109251751534994890;
data->test_sp_matrix_A->x[29] = 0.50839624268343175384;
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


// Matrix test_qpform_Pu
//----------------------
data->test_qpform_Pu = c_malloc(sizeof(csc));
data->test_qpform_Pu->m = 4;
data->test_qpform_Pu->n = 4;
data->test_qpform_Pu->nz = -1;
data->test_qpform_Pu->nzmax = 9;
data->test_qpform_Pu->x = c_malloc(9 * sizeof(c_float));
data->test_qpform_Pu->x[0] = 0.69800328136771883081;
data->test_qpform_Pu->x[1] = 0.79233021930625791018;
data->test_qpform_Pu->x[2] = 1.61026308051140309985;
data->test_qpform_Pu->x[3] = 1.02429538602019309934;
data->test_qpform_Pu->x[4] = 0.56914053205563541749;
data->test_qpform_Pu->x[5] = 0.41158735390014400402;
data->test_qpform_Pu->x[6] = 0.73068020052611559745;
data->test_qpform_Pu->x[7] = 0.87629028108733897362;
data->test_qpform_Pu->x[8] = 0.54227105409040754491;
data->test_qpform_Pu->i = c_malloc(9 * sizeof(c_int));
data->test_qpform_Pu->i[0] = 0;
data->test_qpform_Pu->i[1] = 0;
data->test_qpform_Pu->i[2] = 1;
data->test_qpform_Pu->i[3] = 0;
data->test_qpform_Pu->i[4] = 1;
data->test_qpform_Pu->i[5] = 2;
data->test_qpform_Pu->i[6] = 0;
data->test_qpform_Pu->i[7] = 1;
data->test_qpform_Pu->i[8] = 2;
data->test_qpform_Pu->p = c_malloc((4 + 1) * sizeof(c_int));
data->test_qpform_Pu->p[0] = 0;
data->test_qpform_Pu->p[1] = 1;
data->test_qpform_Pu->p[2] = 3;
data->test_qpform_Pu->p[3] = 6;
data->test_qpform_Pu->p[4] = 9;

data->test_vec_ops_sc = 0.69511960504699144003;
data->test_vec_ops_vec_prod = 3.19487685565542900434;
data->test_mat_extr_triu_n = 5;

// Matrix test_mat_ops_postm_diag
//-------------------------------
data->test_mat_ops_postm_diag = c_malloc(sizeof(csc));
data->test_mat_ops_postm_diag->m = 2;
data->test_mat_ops_postm_diag->n = 2;
data->test_mat_ops_postm_diag->nz = -1;
data->test_mat_ops_postm_diag->nzmax = 3;
data->test_mat_ops_postm_diag->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_postm_diag->x[0] = -0.08499424461397381281;
data->test_mat_ops_postm_diag->x[1] = -0.63700073784942223831;
data->test_mat_ops_postm_diag->x[2] = -0.84571734393041497757;
data->test_mat_ops_postm_diag->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_postm_diag->i[0] = 0;
data->test_mat_ops_postm_diag->i[1] = 1;
data->test_mat_ops_postm_diag->i[2] = 0;
data->test_mat_ops_postm_diag->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_postm_diag->p[0] = 0;
data->test_mat_ops_postm_diag->p[1] = 2;
data->test_mat_ops_postm_diag->p[2] = 3;

data->test_vec_ops_v1 = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_v1[0] = 0.21611600626368202005;
data->test_vec_ops_v1[1] = -1.85861238612349755073;
data->test_vec_ops_v1[2] = -0.41931648215268885194;
data->test_vec_ops_v1[3] = -0.13232889843674336405;
data->test_vec_ops_v1[4] = -0.03957023969360570076;
data->test_vec_ops_v1[5] = 0.32600343338698056783;
data->test_vec_ops_v1[6] = -2.04032304872871517176;
data->test_vec_ops_v1[7] = 0.04625552314169690399;
data->test_vec_ops_v1[8] = -0.67767557732804151183;
data->test_vec_ops_v1[9] = -1.43943902673861812147;

return data;

}

/* function to clean data struct */
void clean_problem_lin_alg_sols_data(lin_alg_sols_data * data){

c_free(data->test_mat_ops_ew_square->x);
c_free(data->test_mat_ops_ew_square->i);
c_free(data->test_mat_ops_ew_square->p);
c_free(data->test_mat_ops_ew_square);
c_free(data->test_vec_ops_v2);
c_free(data->test_mat_vec_Pu->x);
c_free(data->test_mat_vec_Pu->i);
c_free(data->test_mat_vec_Pu->p);
c_free(data->test_mat_vec_Pu);
c_free(data->test_mat_ops_A->x);
c_free(data->test_mat_ops_A->i);
c_free(data->test_mat_ops_A->p);
c_free(data->test_mat_ops_A);
c_free(data->test_mat_ops_ew_abs->x);
c_free(data->test_mat_ops_ew_abs->i);
c_free(data->test_mat_ops_ew_abs->p);
c_free(data->test_mat_ops_ew_abs);
c_free(data->test_mat_vec_ATy);
c_free(data->test_mat_vec_A->x);
c_free(data->test_mat_vec_A->i);
c_free(data->test_mat_vec_A->p);
c_free(data->test_mat_vec_A);
c_free(data->test_mat_ops_d);
c_free(data->test_sp_matrix_Adns);
c_free(data->test_mat_ops_prem_diag->x);
c_free(data->test_mat_ops_prem_diag->i);
c_free(data->test_mat_ops_prem_diag->p);
c_free(data->test_mat_ops_prem_diag);
c_free(data->test_qpform_x);
c_free(data->test_mat_extr_triu_P->x);
c_free(data->test_mat_extr_triu_P->i);
c_free(data->test_mat_extr_triu_P->p);
c_free(data->test_mat_extr_triu_P);
c_free(data->test_vec_ops_ew_reciprocal);
c_free(data->test_mat_extr_triu_Pu->x);
c_free(data->test_mat_extr_triu_Pu->i);
c_free(data->test_mat_extr_triu_Pu->p);
c_free(data->test_mat_extr_triu_Pu);
c_free(data->test_mat_vec_ATy_cum);
c_free(data->test_mat_vec_Px);
c_free(data->test_mat_vec_y);
c_free(data->test_vec_ops_add_scaled);
c_free(data->test_mat_vec_x);
c_free(data->test_mat_vec_Ax_cum);
c_free(data->test_mat_vec_Ax);
c_free(data->test_mat_vec_Px_cum);
c_free(data->test_sp_matrix_A->x);
c_free(data->test_sp_matrix_A->i);
c_free(data->test_sp_matrix_A->p);
c_free(data->test_sp_matrix_A);
c_free(data->test_qpform_Pu->x);
c_free(data->test_qpform_Pu->i);
c_free(data->test_qpform_Pu->p);
c_free(data->test_qpform_Pu);
c_free(data->test_mat_ops_postm_diag->x);
c_free(data->test_mat_ops_postm_diag->i);
c_free(data->test_mat_ops_postm_diag->p);
c_free(data->test_mat_ops_postm_diag);
c_free(data->test_vec_ops_v1);

c_free(data);

}

#endif
