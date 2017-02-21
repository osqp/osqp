#ifndef LIN_ALG_DATA_H
#define LIN_ALG_DATA_H
#include "osqp.h"


/* create data and solutions structure */
typedef struct {
c_float * test_mat_vec_Px;
c_float test_vec_ops_sc;
c_int test_mat_vec_n;
c_int test_mat_extr_triu_n;
csc * test_mat_extr_triu_P;
c_float * test_mat_vec_Ax_cum;
csc * test_mat_ops_ew_abs;
c_float * test_sp_matrix_Adns;
c_int test_vec_ops_n;
c_float * test_mat_vec_Px_cum;
csc * test_mat_vec_Pu;
csc * test_mat_ops_A;
csc * test_mat_ops_postm_diag;
c_float * test_mat_ops_d;
csc * test_qpform_Pu;
c_float test_qpform_value;
csc * test_sp_matrix_A;
c_float test_vec_ops_norm2;
c_float * test_vec_ops_v2;
c_int test_qpform_n;
c_int test_mat_ops_n;
c_float * test_qpform_x;
c_float * test_mat_vec_ATy;
c_float * test_mat_vec_x;
c_float * test_vec_ops_ew_reciprocal;
c_float * test_mat_vec_y;
csc * test_mat_ops_ew_square;
csc * test_mat_extr_triu_Pu;
c_float * test_vec_ops_add_scaled;
c_float * test_mat_vec_Ax;
c_float * test_vec_ops_v1;
c_int test_mat_vec_m;
c_float test_vec_ops_vec_prod;
csc * test_mat_vec_A;
c_float * test_mat_vec_ATy_cum;
c_float test_vec_ops_norm2_diff;
csc * test_mat_ops_prem_diag;
} lin_alg_sols_data;

/* function to define problem data */
lin_alg_sols_data *  generate_problem_lin_alg_sols_data(){

lin_alg_sols_data * data = (lin_alg_sols_data *)c_malloc(sizeof(lin_alg_sols_data));

data->test_mat_vec_Px = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_Px[0] = -3.10042630464168578186;
data->test_mat_vec_Px[1] = -2.10867822812648686437;
data->test_mat_vec_Px[2] = -1.64381886910298957716;
data->test_mat_vec_Px[3] = -0.59413844472753019055;
data->test_vec_ops_sc = 1.65688707250140510041;
data->test_mat_vec_n = 4;
data->test_mat_extr_triu_n = 5;

// Matrix test_mat_extr_triu_P
//----------------------------
data->test_mat_extr_triu_P = c_malloc(sizeof(csc));
data->test_mat_extr_triu_P->m = 5;
data->test_mat_extr_triu_P->n = 5;
data->test_mat_extr_triu_P->nz = -1;
data->test_mat_extr_triu_P->nzmax = 24;
data->test_mat_extr_triu_P->x = c_malloc(24 * sizeof(c_float));
data->test_mat_extr_triu_P->x[0] = 1.87659931296780935206;
data->test_mat_extr_triu_P->x[1] = 0.40790593020428600468;
data->test_mat_extr_triu_P->x[2] = 1.41499890728894972547;
data->test_mat_extr_triu_P->x[3] = 0.13090244115428451011;
data->test_mat_extr_triu_P->x[4] = 0.05807699788934983509;
data->test_mat_extr_triu_P->x[5] = 0.40790593020428600468;
data->test_mat_extr_triu_P->x[6] = 1.60526806568301250167;
data->test_mat_extr_triu_P->x[7] = 1.48799732264566264561;
data->test_mat_extr_triu_P->x[8] = 0.85058132485578830106;
data->test_mat_extr_triu_P->x[9] = 0.50610821132657124455;
data->test_mat_extr_triu_P->x[10] = 1.41499890728894972547;
data->test_mat_extr_triu_P->x[11] = 1.48799732264566264561;
data->test_mat_extr_triu_P->x[12] = 1.24439486216141781405;
data->test_mat_extr_triu_P->x[13] = 0.75769551394649214959;
data->test_mat_extr_triu_P->x[14] = 0.51792012092106431975;
data->test_mat_extr_triu_P->x[15] = 0.13090244115428451011;
data->test_mat_extr_triu_P->x[16] = 0.85058132485578830106;
data->test_mat_extr_triu_P->x[17] = 0.75769551394649214959;
data->test_mat_extr_triu_P->x[18] = 1.78993391688108216542;
data->test_mat_extr_triu_P->x[19] = 1.02021911375076301809;
data->test_mat_extr_triu_P->x[20] = 0.05807699788934983509;
data->test_mat_extr_triu_P->x[21] = 0.50610821132657124455;
data->test_mat_extr_triu_P->x[22] = 0.51792012092106431975;
data->test_mat_extr_triu_P->x[23] = 1.02021911375076301809;
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
data->test_mat_extr_triu_P->i[12] = 2;
data->test_mat_extr_triu_P->i[13] = 3;
data->test_mat_extr_triu_P->i[14] = 4;
data->test_mat_extr_triu_P->i[15] = 0;
data->test_mat_extr_triu_P->i[16] = 1;
data->test_mat_extr_triu_P->i[17] = 2;
data->test_mat_extr_triu_P->i[18] = 3;
data->test_mat_extr_triu_P->i[19] = 4;
data->test_mat_extr_triu_P->i[20] = 0;
data->test_mat_extr_triu_P->i[21] = 1;
data->test_mat_extr_triu_P->i[22] = 2;
data->test_mat_extr_triu_P->i[23] = 3;
data->test_mat_extr_triu_P->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_mat_extr_triu_P->p[0] = 0;
data->test_mat_extr_triu_P->p[1] = 5;
data->test_mat_extr_triu_P->p[2] = 10;
data->test_mat_extr_triu_P->p[3] = 15;
data->test_mat_extr_triu_P->p[4] = 20;
data->test_mat_extr_triu_P->p[5] = 24;

data->test_mat_vec_Ax_cum = c_malloc(5 * sizeof(c_float));
data->test_mat_vec_Ax_cum[0] = -3.14994220355042786608;
data->test_mat_vec_Ax_cum[1] = -2.24601296805998140727;
data->test_mat_vec_Ax_cum[2] = -1.42206954076089919781;
data->test_mat_vec_Ax_cum[3] = -1.60623012807783949185;
data->test_mat_vec_Ax_cum[4] = -1.61003841927201163386;

// Matrix test_mat_ops_ew_abs
//---------------------------
data->test_mat_ops_ew_abs = c_malloc(sizeof(csc));
data->test_mat_ops_ew_abs->m = 2;
data->test_mat_ops_ew_abs->n = 2;
data->test_mat_ops_ew_abs->nz = -1;
data->test_mat_ops_ew_abs->nzmax = 3;
data->test_mat_ops_ew_abs->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_ew_abs->x[0] = 0.83682568722636174741;
data->test_mat_ops_ew_abs->x[1] = 0.24947717041060202270;
data->test_mat_ops_ew_abs->x[2] = 0.68551650376192496683;
data->test_mat_ops_ew_abs->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_ew_abs->i[0] = 0;
data->test_mat_ops_ew_abs->i[1] = 0;
data->test_mat_ops_ew_abs->i[2] = 1;
data->test_mat_ops_ew_abs->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_ew_abs->p[0] = 0;
data->test_mat_ops_ew_abs->p[1] = 1;
data->test_mat_ops_ew_abs->p[2] = 3;

data->test_sp_matrix_Adns = c_malloc(30 * sizeof(c_float));
data->test_sp_matrix_Adns[0] = 0.77120204510700851586;
data->test_sp_matrix_Adns[1] = -0.41711624390860740919;
data->test_sp_matrix_Adns[2] = 0.24626350632941360064;
data->test_sp_matrix_Adns[3] = 1.00157057531537274642;
data->test_sp_matrix_Adns[4] = 0.53250173896975661059;
data->test_sp_matrix_Adns[5] = -0.82449301853388368144;
data->test_sp_matrix_Adns[6] = 0.13208613406015670688;
data->test_sp_matrix_Adns[7] = 1.27880275344261140624;
data->test_sp_matrix_Adns[8] = 1.73396591514021514158;
data->test_sp_matrix_Adns[9] = -0.39736854249668246419;
data->test_sp_matrix_Adns[10] = -1.67889198643108095155;
data->test_sp_matrix_Adns[11] = 0.81399219319461235678;
data->test_sp_matrix_Adns[12] = -1.79448553457930581878;
data->test_sp_matrix_Adns[13] = -0.03508021476940936800;
data->test_sp_matrix_Adns[14] = 0.22450635552698977593;
data->test_sp_matrix_Adns[15] = -1.63071607596873269230;
data->test_sp_matrix_Adns[16] = 0.02615749426483353693;
data->test_sp_matrix_Adns[17] = -1.17070934212600330504;
data->test_sp_matrix_Adns[18] = -1.50641256686994506886;
data->test_sp_matrix_Adns[19] = -0.17986390931374862667;
data->test_sp_matrix_Adns[20] = -1.43963580529228529770;
data->test_sp_matrix_Adns[21] = -0.80644626553919573908;
data->test_sp_matrix_Adns[22] = -0.62217692264299939708;
data->test_sp_matrix_Adns[23] = 0.22870264423729413927;
data->test_sp_matrix_Adns[24] = 2.21419332022375225932;
data->test_sp_matrix_Adns[25] = -1.03476106954961477591;
data->test_sp_matrix_Adns[26] = -1.60298899230416891371;
data->test_sp_matrix_Adns[27] = -0.01011828493912804959;
data->test_sp_matrix_Adns[28] = 0.35816112234650654411;
data->test_sp_matrix_Adns[29] = -0.01895322388238612765;
data->test_vec_ops_n = 10;
data->test_mat_vec_Px_cum = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_Px_cum[0] = -2.78876394603934452121;
data->test_mat_vec_Px_cum[1] = -3.93847939596677099061;
data->test_mat_vec_Px_cum[2] = -2.43599857769207783775;
data->test_mat_vec_Px_cum[3] = -2.42772822449501068931;

// Matrix test_mat_vec_Pu
//-----------------------
data->test_mat_vec_Pu = c_malloc(sizeof(csc));
data->test_mat_vec_Pu->m = 4;
data->test_mat_vec_Pu->n = 4;
data->test_mat_vec_Pu->nz = -1;
data->test_mat_vec_Pu->nzmax = 9;
data->test_mat_vec_Pu->x = c_malloc(9 * sizeof(c_float));
data->test_mat_vec_Pu->x[0] = 1.24389657174205070511;
data->test_mat_vec_Pu->x[1] = 0.81282737549599248794;
data->test_mat_vec_Pu->x[2] = 0.87692791653001056495;
data->test_mat_vec_Pu->x[3] = 1.84014575261686719188;
data->test_mat_vec_Pu->x[4] = 0.53194393859132971247;
data->test_mat_vec_Pu->x[5] = 0.54436619888931025990;
data->test_mat_vec_Pu->x[6] = 0.29617499128537205788;
data->test_mat_vec_Pu->x[7] = 0.18325091510673452433;
data->test_mat_vec_Pu->x[8] = 0.44324829176177416024;
data->test_mat_vec_Pu->i = c_malloc(9 * sizeof(c_int));
data->test_mat_vec_Pu->i[0] = 0;
data->test_mat_vec_Pu->i[1] = 0;
data->test_mat_vec_Pu->i[2] = 1;
data->test_mat_vec_Pu->i[3] = 0;
data->test_mat_vec_Pu->i[4] = 1;
data->test_mat_vec_Pu->i[5] = 2;
data->test_mat_vec_Pu->i[6] = 0;
data->test_mat_vec_Pu->i[7] = 1;
data->test_mat_vec_Pu->i[8] = 2;
data->test_mat_vec_Pu->p = c_malloc((4 + 1) * sizeof(c_int));
data->test_mat_vec_Pu->p[0] = 0;
data->test_mat_vec_Pu->p[1] = 1;
data->test_mat_vec_Pu->p[2] = 3;
data->test_mat_vec_Pu->p[3] = 6;
data->test_mat_vec_Pu->p[4] = 9;


// Matrix test_mat_ops_A
//----------------------
data->test_mat_ops_A = c_malloc(sizeof(csc));
data->test_mat_ops_A->m = 2;
data->test_mat_ops_A->n = 2;
data->test_mat_ops_A->nz = -1;
data->test_mat_ops_A->nzmax = 3;
data->test_mat_ops_A->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_A->x[0] = 0.83682568722636174741;
data->test_mat_ops_A->x[1] = 0.24947717041060202270;
data->test_mat_ops_A->x[2] = 0.68551650376192496683;
data->test_mat_ops_A->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_A->i[0] = 0;
data->test_mat_ops_A->i[1] = 0;
data->test_mat_ops_A->i[2] = 1;
data->test_mat_ops_A->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_A->p[0] = 0;
data->test_mat_ops_A->p[1] = 1;
data->test_mat_ops_A->p[2] = 3;


// Matrix test_mat_ops_postm_diag
//-------------------------------
data->test_mat_ops_postm_diag = c_malloc(sizeof(csc));
data->test_mat_ops_postm_diag->m = 2;
data->test_mat_ops_postm_diag->n = 2;
data->test_mat_ops_postm_diag->nz = -1;
data->test_mat_ops_postm_diag->nzmax = 3;
data->test_mat_ops_postm_diag->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_postm_diag->x[0] = 0.27450826733414390723;
data->test_mat_ops_postm_diag->x[1] = -0.25486650595331705738;
data->test_mat_ops_postm_diag->x[2] = -0.70032538768810281837;
data->test_mat_ops_postm_diag->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_postm_diag->i[0] = 0;
data->test_mat_ops_postm_diag->i[1] = 0;
data->test_mat_ops_postm_diag->i[2] = 1;
data->test_mat_ops_postm_diag->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_postm_diag->p[0] = 0;
data->test_mat_ops_postm_diag->p[1] = 1;
data->test_mat_ops_postm_diag->p[2] = 3;

data->test_mat_ops_d = c_malloc(2 * sizeof(c_float));
data->test_mat_ops_d[0] = 0.32803518286346450283;
data->test_mat_ops_d[1] = -1.02160251991733352916;

// Matrix test_qpform_Pu
//----------------------
data->test_qpform_Pu = c_malloc(sizeof(csc));
data->test_qpform_Pu->m = 4;
data->test_qpform_Pu->n = 4;
data->test_qpform_Pu->nz = -1;
data->test_qpform_Pu->nzmax = 9;
data->test_qpform_Pu->x = c_malloc(9 * sizeof(c_float));
data->test_qpform_Pu->x[0] = 1.83316970496872411189;
data->test_qpform_Pu->x[1] = 0.61488984459830564600;
data->test_qpform_Pu->x[2] = 0.96883386468614118847;
data->test_qpform_Pu->x[3] = 1.07371326975903125245;
data->test_qpform_Pu->x[4] = 0.91734373582336159458;
data->test_qpform_Pu->x[5] = 1.03055620629624988815;
data->test_qpform_Pu->x[6] = 0.75693311794270790038;
data->test_qpform_Pu->x[7] = 0.40683449490143952509;
data->test_qpform_Pu->x[8] = 0.92530405852323127647;
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

data->test_qpform_value = -0.47052856560683714582;

// Matrix test_sp_matrix_A
//------------------------
data->test_sp_matrix_A = c_malloc(sizeof(csc));
data->test_sp_matrix_A->m = 5;
data->test_sp_matrix_A->n = 6;
data->test_sp_matrix_A->nz = -1;
data->test_sp_matrix_A->nzmax = 30;
data->test_sp_matrix_A->x = c_malloc(30 * sizeof(c_float));
data->test_sp_matrix_A->x[0] = 0.77120204510700851586;
data->test_sp_matrix_A->x[1] = -0.41711624390860740919;
data->test_sp_matrix_A->x[2] = 0.24626350632941360064;
data->test_sp_matrix_A->x[3] = 1.00157057531537274642;
data->test_sp_matrix_A->x[4] = 0.53250173896975661059;
data->test_sp_matrix_A->x[5] = -0.82449301853388368144;
data->test_sp_matrix_A->x[6] = 0.13208613406015670688;
data->test_sp_matrix_A->x[7] = 1.27880275344261140624;
data->test_sp_matrix_A->x[8] = 1.73396591514021514158;
data->test_sp_matrix_A->x[9] = -0.39736854249668246419;
data->test_sp_matrix_A->x[10] = -1.67889198643108095155;
data->test_sp_matrix_A->x[11] = 0.81399219319461235678;
data->test_sp_matrix_A->x[12] = -1.79448553457930581878;
data->test_sp_matrix_A->x[13] = -0.03508021476940936800;
data->test_sp_matrix_A->x[14] = 0.22450635552698977593;
data->test_sp_matrix_A->x[15] = -1.63071607596873269230;
data->test_sp_matrix_A->x[16] = 0.02615749426483353693;
data->test_sp_matrix_A->x[17] = -1.17070934212600330504;
data->test_sp_matrix_A->x[18] = -1.50641256686994506886;
data->test_sp_matrix_A->x[19] = -0.17986390931374862667;
data->test_sp_matrix_A->x[20] = -1.43963580529228529770;
data->test_sp_matrix_A->x[21] = -0.80644626553919573908;
data->test_sp_matrix_A->x[22] = -0.62217692264299939708;
data->test_sp_matrix_A->x[23] = 0.22870264423729413927;
data->test_sp_matrix_A->x[24] = 2.21419332022375225932;
data->test_sp_matrix_A->x[25] = -1.03476106954961477591;
data->test_sp_matrix_A->x[26] = -1.60298899230416891371;
data->test_sp_matrix_A->x[27] = -0.01011828493912804959;
data->test_sp_matrix_A->x[28] = 0.35816112234650654411;
data->test_sp_matrix_A->x[29] = -0.01895322388238612765;
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

data->test_vec_ops_norm2 = 2.97328648513679683063;
data->test_vec_ops_v2 = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_v2[0] = -1.56271774213040015589;
data->test_vec_ops_v2[1] = 1.63494271003785551777;
data->test_vec_ops_v2[2] = 0.13924579082541474473;
data->test_vec_ops_v2[3] = -0.94154446727803520201;
data->test_vec_ops_v2[4] = 0.07376555808095372480;
data->test_vec_ops_v2[5] = -0.70122690607540771879;
data->test_vec_ops_v2[6] = 0.25357370592905548179;
data->test_vec_ops_v2[7] = -0.01578453700883215502;
data->test_vec_ops_v2[8] = -0.62108095431903997952;
data->test_vec_ops_v2[9] = 0.27810838277508698191;
data->test_qpform_n = 4;
data->test_mat_ops_n = 2;
data->test_qpform_x = c_malloc(4 * sizeof(c_float));
data->test_qpform_x[0] = 0.10246042610257902195;
data->test_qpform_x[1] = 1.11409031569996197497;
data->test_qpform_x[2] = -0.12363928482187427904;
data->test_qpform_x[3] = -2.44838533285356962921;
data->test_mat_vec_ATy = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_ATy[0] = -0.21562708593362819864;
data->test_mat_vec_ATy[1] = -0.45121432748573053750;
data->test_mat_vec_ATy[2] = 0.00121578628511676001;
data->test_mat_vec_ATy[3] = -0.59712623259610897453;
data->test_mat_vec_x = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_x[0] = 0.31166235860234114963;
data->test_mat_vec_x[1] = -1.82980116784028412624;
data->test_mat_vec_x[2] = -0.79217970858908837162;
data->test_mat_vec_x[3] = -1.83358977976748049876;
data->test_vec_ops_ew_reciprocal = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_ew_reciprocal[0] = -1.42965301722251747485;
data->test_vec_ops_ew_reciprocal[1] = -2.15293649837515488343;
data->test_vec_ops_ew_reciprocal[2] = 0.95078137573798893190;
data->test_vec_ops_ew_reciprocal[3] = -0.87367205937355929546;
data->test_vec_ops_ew_reciprocal[4] = -0.53239934324031290558;
data->test_vec_ops_ew_reciprocal[5] = -1.79501962396863690152;
data->test_vec_ops_ew_reciprocal[6] = 2.42362056954186577684;
data->test_vec_ops_ew_reciprocal[7] = -1.09336064478136418820;
data->test_vec_ops_ew_reciprocal[8] = -7.45617525090987687264;
data->test_vec_ops_ew_reciprocal[9] = -1.08081337209123895526;
data->test_mat_vec_y = c_malloc(5 * sizeof(c_float));
data->test_mat_vec_y[0] = -1.24518690417626731737;
data->test_mat_vec_y[1] = 0.03438406106254042471;
data->test_mat_vec_y[2] = 0.23044550440654865131;
data->test_mat_vec_y[3] = 0.11396283740912248328;
data->test_mat_vec_y[4] = -0.08932434171152027480;

// Matrix test_mat_ops_ew_square
//------------------------------
data->test_mat_ops_ew_square = c_malloc(sizeof(csc));
data->test_mat_ops_ew_square->m = 2;
data->test_mat_ops_ew_square->n = 2;
data->test_mat_ops_ew_square->nz = -1;
data->test_mat_ops_ew_square->nzmax = 3;
data->test_mat_ops_ew_square->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_ew_square->x[0] = 0.70027723080187265214;
data->test_mat_ops_ew_square->x[1] = 0.06223885855608056461;
data->test_mat_ops_ew_square->x[2] = 0.46993287692997326443;
data->test_mat_ops_ew_square->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_ew_square->i[0] = 0;
data->test_mat_ops_ew_square->i[1] = 0;
data->test_mat_ops_ew_square->i[2] = 1;
data->test_mat_ops_ew_square->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_ew_square->p[0] = 0;
data->test_mat_ops_ew_square->p[1] = 1;
data->test_mat_ops_ew_square->p[2] = 3;


// Matrix test_mat_extr_triu_Pu
//-----------------------------
data->test_mat_extr_triu_Pu = c_malloc(sizeof(csc));
data->test_mat_extr_triu_Pu->m = 5;
data->test_mat_extr_triu_Pu->n = 5;
data->test_mat_extr_triu_Pu->nz = -1;
data->test_mat_extr_triu_Pu->nzmax = 14;
data->test_mat_extr_triu_Pu->x = c_malloc(14 * sizeof(c_float));
data->test_mat_extr_triu_Pu->x[0] = 1.87659931296780935206;
data->test_mat_extr_triu_Pu->x[1] = 0.40790593020428600468;
data->test_mat_extr_triu_Pu->x[2] = 1.60526806568301250167;
data->test_mat_extr_triu_Pu->x[3] = 1.41499890728894972547;
data->test_mat_extr_triu_Pu->x[4] = 1.48799732264566264561;
data->test_mat_extr_triu_Pu->x[5] = 1.24439486216141781405;
data->test_mat_extr_triu_Pu->x[6] = 0.13090244115428451011;
data->test_mat_extr_triu_Pu->x[7] = 0.85058132485578830106;
data->test_mat_extr_triu_Pu->x[8] = 0.75769551394649214959;
data->test_mat_extr_triu_Pu->x[9] = 1.78993391688108216542;
data->test_mat_extr_triu_Pu->x[10] = 0.05807699788934983509;
data->test_mat_extr_triu_Pu->x[11] = 0.50610821132657124455;
data->test_mat_extr_triu_Pu->x[12] = 0.51792012092106431975;
data->test_mat_extr_triu_Pu->x[13] = 1.02021911375076301809;
data->test_mat_extr_triu_Pu->i = c_malloc(14 * sizeof(c_int));
data->test_mat_extr_triu_Pu->i[0] = 0;
data->test_mat_extr_triu_Pu->i[1] = 0;
data->test_mat_extr_triu_Pu->i[2] = 1;
data->test_mat_extr_triu_Pu->i[3] = 0;
data->test_mat_extr_triu_Pu->i[4] = 1;
data->test_mat_extr_triu_Pu->i[5] = 2;
data->test_mat_extr_triu_Pu->i[6] = 0;
data->test_mat_extr_triu_Pu->i[7] = 1;
data->test_mat_extr_triu_Pu->i[8] = 2;
data->test_mat_extr_triu_Pu->i[9] = 3;
data->test_mat_extr_triu_Pu->i[10] = 0;
data->test_mat_extr_triu_Pu->i[11] = 1;
data->test_mat_extr_triu_Pu->i[12] = 2;
data->test_mat_extr_triu_Pu->i[13] = 3;
data->test_mat_extr_triu_Pu->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_mat_extr_triu_Pu->p[0] = 0;
data->test_mat_extr_triu_Pu->p[1] = 1;
data->test_mat_extr_triu_Pu->p[2] = 3;
data->test_mat_extr_triu_Pu->p[3] = 6;
data->test_mat_extr_triu_Pu->p[4] = 10;
data->test_mat_extr_triu_Pu->p[5] = 14;

data->test_vec_ops_add_scaled = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_add_scaled[0] = -3.28871724741491266997;
data->test_vec_ops_add_scaled[1] = 2.24443355695908053349;
data->test_vec_ops_add_scaled[2] = 1.28248105090278730067;
data->test_vec_ops_add_scaled[3] = -2.70462708822589981139;
data->test_vec_ops_add_scaled[4] = -1.75606811969987908206;
data->test_vec_ops_add_scaled[5] = -1.71895076912998923113;
data->test_vec_ops_add_scaled[6] = 0.83274883489347195642;
data->test_vec_ops_add_scaled[7] = -0.94076449468214873306;
data->test_vec_ops_add_scaled[8] = -1.16317802348396748258;
data->test_vec_ops_add_scaled[9] = -0.46443493106611344334;
data->test_mat_vec_Ax = c_malloc(5 * sizeof(c_float));
data->test_mat_vec_Ax[0] = -1.90475529937416054871;
data->test_mat_vec_Ax[1] = -2.28039702912252195688;
data->test_mat_vec_Ax[2] = -1.65251504516744773809;
data->test_mat_vec_Ax[3] = -1.72019296548696187799;
data->test_mat_vec_Ax[4] = -1.52071407756049126192;
data->test_vec_ops_v1 = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_v1[0] = -0.69947042251046820382;
data->test_vec_ops_v1[1] = -0.46448188358305553258;
data->test_vec_ops_v1[2] = 1.05176650018392292552;
data->test_vec_ops_v1[3] = -1.14459423220770095675;
data->test_vec_ops_v1[4] = -1.87828931928006293006;
data->test_vec_ops_v1[5] = -0.55709697356348919506;
data->test_vec_ops_v1[6] = 0.41260583961334706959;
data->test_vec_ops_v1[7] = -0.91461129936679474994;
data->test_vec_ops_v1[8] = -0.13411701929591449134;
data->test_vec_ops_v1[9] = -0.92522911524042750209;
data->test_mat_vec_m = 5;
data->test_vec_ops_vec_prod = 1.75495858296186746372;

// Matrix test_mat_vec_A
//----------------------
data->test_mat_vec_A = c_malloc(sizeof(csc));
data->test_mat_vec_A->m = 5;
data->test_mat_vec_A->n = 4;
data->test_mat_vec_A->nz = -1;
data->test_mat_vec_A->nzmax = 20;
data->test_mat_vec_A->x = c_malloc(20 * sizeof(c_float));
data->test_mat_vec_A->x[0] = 0.25475386191298099448;
data->test_mat_vec_A->x[1] = 0.61608602698208558834;
data->test_mat_vec_A->x[2] = 0.32224198248235869091;
data->test_mat_vec_A->x[3] = 0.37167815695981099022;
data->test_mat_vec_A->x[4] = 0.40538968139254072387;
data->test_mat_vec_A->x[5] = 0.37809004752285391149;
data->test_mat_vec_A->x[6] = 0.35124270370842447520;
data->test_mat_vec_A->x[7] = 0.00106190194774313795;
data->test_mat_vec_A->x[8] = 0.18428201263529497833;
data->test_mat_vec_A->x[9] = 0.15387422502056735674;
data->test_mat_vec_A->x[10] = 0.08011597305011852743;
data->test_mat_vec_A->x[11] = 0.35232887319025896211;
data->test_mat_vec_A->x[12] = 0.41955184315454929767;
data->test_mat_vec_A->x[13] = 0.26176210538024347052;
data->test_mat_vec_A->x[14] = 0.42154622582923606533;
data->test_mat_vec_A->x[15] = 0.67019168829931374809;
data->test_mat_vec_A->x[16] = 0.84566121439545871574;
data->test_mat_vec_A->x[17] = 0.77369662006743700200;
data->test_mat_vec_A->x[18] = 0.70433910599021343213;
data->test_mat_vec_A->x[19] = 0.56258994799249384400;
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

data->test_mat_vec_ATy_cum = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_ATy_cum[0] = 0.09603527266871295098;
data->test_mat_vec_ATy_cum[1] = -2.28101549532601488579;
data->test_mat_vec_ATy_cum[2] = -0.79096392230397161160;
data->test_mat_vec_ATy_cum[3] = -2.43071601236358958431;
data->test_vec_ops_norm2_diff = 3.51800089102757596038;

// Matrix test_mat_ops_prem_diag
//------------------------------
data->test_mat_ops_prem_diag = c_malloc(sizeof(csc));
data->test_mat_ops_prem_diag->m = 2;
data->test_mat_ops_prem_diag->n = 2;
data->test_mat_ops_prem_diag->nz = -1;
data->test_mat_ops_prem_diag->nzmax = 3;
data->test_mat_ops_prem_diag->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_prem_diag->x[0] = 0.27450826733414390723;
data->test_mat_ops_prem_diag->x[1] = 0.08183728921590152638;
data->test_mat_ops_prem_diag->x[2] = -0.70032538768810281837;
data->test_mat_ops_prem_diag->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_prem_diag->i[0] = 0;
data->test_mat_ops_prem_diag->i[1] = 0;
data->test_mat_ops_prem_diag->i[2] = 1;
data->test_mat_ops_prem_diag->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_prem_diag->p[0] = 0;
data->test_mat_ops_prem_diag->p[1] = 1;
data->test_mat_ops_prem_diag->p[2] = 3;


return data;

}

/* function to clean data struct */
void clean_problem_lin_alg_sols_data(lin_alg_sols_data * data){

c_free(data->test_mat_vec_Px);
c_free(data->test_mat_extr_triu_P->x);
c_free(data->test_mat_extr_triu_P->i);
c_free(data->test_mat_extr_triu_P->p);
c_free(data->test_mat_extr_triu_P);
c_free(data->test_mat_vec_Ax_cum);
c_free(data->test_mat_ops_ew_abs->x);
c_free(data->test_mat_ops_ew_abs->i);
c_free(data->test_mat_ops_ew_abs->p);
c_free(data->test_mat_ops_ew_abs);
c_free(data->test_sp_matrix_Adns);
c_free(data->test_mat_vec_Px_cum);
c_free(data->test_mat_vec_Pu->x);
c_free(data->test_mat_vec_Pu->i);
c_free(data->test_mat_vec_Pu->p);
c_free(data->test_mat_vec_Pu);
c_free(data->test_mat_ops_A->x);
c_free(data->test_mat_ops_A->i);
c_free(data->test_mat_ops_A->p);
c_free(data->test_mat_ops_A);
c_free(data->test_mat_ops_postm_diag->x);
c_free(data->test_mat_ops_postm_diag->i);
c_free(data->test_mat_ops_postm_diag->p);
c_free(data->test_mat_ops_postm_diag);
c_free(data->test_mat_ops_d);
c_free(data->test_qpform_Pu->x);
c_free(data->test_qpform_Pu->i);
c_free(data->test_qpform_Pu->p);
c_free(data->test_qpform_Pu);
c_free(data->test_sp_matrix_A->x);
c_free(data->test_sp_matrix_A->i);
c_free(data->test_sp_matrix_A->p);
c_free(data->test_sp_matrix_A);
c_free(data->test_vec_ops_v2);
c_free(data->test_qpform_x);
c_free(data->test_mat_vec_ATy);
c_free(data->test_mat_vec_x);
c_free(data->test_vec_ops_ew_reciprocal);
c_free(data->test_mat_vec_y);
c_free(data->test_mat_ops_ew_square->x);
c_free(data->test_mat_ops_ew_square->i);
c_free(data->test_mat_ops_ew_square->p);
c_free(data->test_mat_ops_ew_square);
c_free(data->test_mat_extr_triu_Pu->x);
c_free(data->test_mat_extr_triu_Pu->i);
c_free(data->test_mat_extr_triu_Pu->p);
c_free(data->test_mat_extr_triu_Pu);
c_free(data->test_vec_ops_add_scaled);
c_free(data->test_mat_vec_Ax);
c_free(data->test_vec_ops_v1);
c_free(data->test_mat_vec_A->x);
c_free(data->test_mat_vec_A->i);
c_free(data->test_mat_vec_A->p);
c_free(data->test_mat_vec_A);
c_free(data->test_mat_vec_ATy_cum);
c_free(data->test_mat_ops_prem_diag->x);
c_free(data->test_mat_ops_prem_diag->i);
c_free(data->test_mat_ops_prem_diag->p);
c_free(data->test_mat_ops_prem_diag);

c_free(data);

}

#endif
