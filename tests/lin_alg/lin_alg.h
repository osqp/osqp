#ifndef LIN_ALG_DATA_H
#define LIN_ALG_DATA_H
#include "osqp.h"


/* create data and solutions structure */
typedef struct {
c_float test_qpform_value;
csc * test_mat_ops_postm_diag;
c_float test_vec_ops_sc;
csc * test_sp_matrix_A;
c_float * test_mat_vec_Px;
csc * test_mat_ops_A;
c_float test_vec_ops_norm2_diff;
c_float * test_mat_vec_ATy_cum;
c_float * test_mat_vec_Ax_cum;
c_float * test_mat_ops_d;
csc * test_mat_ops_ew_abs;
c_float * test_mat_vec_y;
c_float test_vec_ops_vec_prod;
c_float * test_mat_vec_ATy;
c_float * test_vec_ops_add_scaled;
c_int test_mat_extr_triu_n;
c_int test_mat_vec_m;
c_float * test_vec_ops_ew_reciprocal;
c_int test_vec_ops_n;
csc * test_mat_ops_prem_diag;
csc * test_mat_vec_Pu;
csc * test_qpform_Pu;
c_float * test_sp_matrix_Adns;
c_float * test_vec_ops_v2;
c_int test_mat_ops_n;
c_float * test_mat_vec_x;
c_float * test_mat_vec_Px_cum;
c_float * test_qpform_x;
c_float test_vec_ops_norm2;
csc * test_mat_ops_ew_square;
csc * test_mat_extr_triu_P;
csc * test_mat_vec_A;
c_float * test_mat_vec_Ax;
c_int test_mat_vec_n;
csc * test_mat_extr_triu_Pu;
c_int test_qpform_n;
c_float * test_vec_ops_v1;
} lin_alg_sols_data;

/* function to define problem data */
lin_alg_sols_data *  generate_problem_lin_alg_sols_data(){

lin_alg_sols_data * data = (lin_alg_sols_data *)c_malloc(sizeof(lin_alg_sols_data));

data->test_qpform_value = -1.44453048704165842864;

// Matrix test_mat_ops_postm_diag
//-------------------------------
data->test_mat_ops_postm_diag = c_malloc(sizeof(csc));
data->test_mat_ops_postm_diag->m = 2;
data->test_mat_ops_postm_diag->n = 2;
data->test_mat_ops_postm_diag->nz = -1;
data->test_mat_ops_postm_diag->nzmax = 3;
data->test_mat_ops_postm_diag->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_postm_diag->x[0] = -0.23110842805233117181;
data->test_mat_ops_postm_diag->x[1] = -0.00616632235602700227;
data->test_mat_ops_postm_diag->x[2] = 0.60406128202899889157;
data->test_mat_ops_postm_diag->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_postm_diag->i[0] = 0;
data->test_mat_ops_postm_diag->i[1] = 1;
data->test_mat_ops_postm_diag->i[2] = 1;
data->test_mat_ops_postm_diag->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_postm_diag->p[0] = 0;
data->test_mat_ops_postm_diag->p[1] = 2;
data->test_mat_ops_postm_diag->p[2] = 3;

data->test_vec_ops_sc = -0.68614714068225302057;

// Matrix test_sp_matrix_A
//------------------------
data->test_sp_matrix_A = c_malloc(sizeof(csc));
data->test_sp_matrix_A->m = 5;
data->test_sp_matrix_A->n = 6;
data->test_sp_matrix_A->nz = -1;
data->test_sp_matrix_A->nzmax = 30;
data->test_sp_matrix_A->x = c_malloc(30 * sizeof(c_float));
data->test_sp_matrix_A->x[0] = 0.89633564821013733681;
data->test_sp_matrix_A->x[1] = -0.86135203505779533995;
data->test_sp_matrix_A->x[2] = -1.10089144467338706512;
data->test_sp_matrix_A->x[3] = 1.37787757416684386591;
data->test_sp_matrix_A->x[4] = -0.44483357029666292792;
data->test_sp_matrix_A->x[5] = 0.20847086947025550430;
data->test_sp_matrix_A->x[6] = 0.76527742100100404343;
data->test_sp_matrix_A->x[7] = -0.23902998143290987709;
data->test_sp_matrix_A->x[8] = 1.97797682528532492441;
data->test_sp_matrix_A->x[9] = -0.78035854354643940933;
data->test_sp_matrix_A->x[10] = 0.20276779166673114529;
data->test_sp_matrix_A->x[11] = 1.36258847737912924813;
data->test_sp_matrix_A->x[12] = -1.37277544886201230412;
data->test_sp_matrix_A->x[13] = 0.94974490834839631059;
data->test_sp_matrix_A->x[14] = -1.72460560726366285422;
data->test_sp_matrix_A->x[15] = 0.90549088145200518074;
data->test_sp_matrix_A->x[16] = -0.82103366081837236834;
data->test_sp_matrix_A->x[17] = -0.12602258633573590330;
data->test_sp_matrix_A->x[18] = -0.53137601952194546406;
data->test_sp_matrix_A->x[19] = 1.58969905211242545917;
data->test_sp_matrix_A->x[20] = 1.77289964277917988511;
data->test_sp_matrix_A->x[21] = 0.50847677961895987320;
data->test_sp_matrix_A->x[22] = 0.72516819874470306306;
data->test_sp_matrix_A->x[23] = -1.04338677468111340474;
data->test_sp_matrix_A->x[24] = 0.36534560394744641787;
data->test_sp_matrix_A->x[25] = -0.64684787223965534153;
data->test_sp_matrix_A->x[26] = 1.73013142131965014237;
data->test_sp_matrix_A->x[27] = 1.28635457091712779309;
data->test_sp_matrix_A->x[28] = 0.48462218677153662272;
data->test_sp_matrix_A->x[29] = 0.47227886025304549600;
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

data->test_mat_vec_Px = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_Px[0] = 3.69375242514005552863;
data->test_mat_vec_Px[1] = 6.19153372859133188655;
data->test_mat_vec_Px[2] = 4.39918576372532488250;
data->test_mat_vec_Px[3] = 4.17078701428878861890;

// Matrix test_mat_ops_A
//----------------------
data->test_mat_ops_A = c_malloc(sizeof(csc));
data->test_mat_ops_A->m = 2;
data->test_mat_ops_A->n = 2;
data->test_mat_ops_A->nz = -1;
data->test_mat_ops_A->nzmax = 3;
data->test_mat_ops_A->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_A->x[0] = 0.62426073447292296237;
data->test_mat_ops_A->x[1] = 0.01665622043908576710;
data->test_mat_ops_A->x[2] = 0.60177928908376177031;
data->test_mat_ops_A->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_A->i[0] = 0;
data->test_mat_ops_A->i[1] = 1;
data->test_mat_ops_A->i[2] = 1;
data->test_mat_ops_A->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_A->p[0] = 0;
data->test_mat_ops_A->p[1] = 2;
data->test_mat_ops_A->p[2] = 3;

data->test_vec_ops_norm2_diff = 5.03739399078414074040;
data->test_mat_vec_ATy_cum = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_ATy_cum[0] = 1.13199923268230140394;
data->test_mat_vec_ATy_cum[1] = 3.40631315580704896462;
data->test_mat_vec_ATy_cum[2] = 0.54945972088149530599;
data->test_mat_vec_ATy_cum[3] = 0.85673175640007548370;
data->test_mat_vec_Ax_cum = c_malloc(5 * sizeof(c_float));
data->test_mat_vec_Ax_cum[0] = 3.93701800780493327281;
data->test_mat_vec_Ax_cum[1] = 3.58165935753707254108;
data->test_mat_vec_Ax_cum[2] = 2.35361012779303591103;
data->test_mat_vec_Ax_cum[3] = 3.02722710627426350172;
data->test_mat_vec_Ax_cum[4] = 3.88856104693935566985;
data->test_mat_ops_d = c_malloc(2 * sizeof(c_float));
data->test_mat_ops_d[0] = -0.37021138010139481578;
data->test_mat_ops_d[1] = 1.00379207624229072060;

// Matrix test_mat_ops_ew_abs
//---------------------------
data->test_mat_ops_ew_abs = c_malloc(sizeof(csc));
data->test_mat_ops_ew_abs->m = 2;
data->test_mat_ops_ew_abs->n = 2;
data->test_mat_ops_ew_abs->nz = -1;
data->test_mat_ops_ew_abs->nzmax = 3;
data->test_mat_ops_ew_abs->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_ew_abs->x[0] = 0.62426073447292296237;
data->test_mat_ops_ew_abs->x[1] = 0.01665622043908576710;
data->test_mat_ops_ew_abs->x[2] = 0.60177928908376177031;
data->test_mat_ops_ew_abs->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_ew_abs->i[0] = 0;
data->test_mat_ops_ew_abs->i[1] = 1;
data->test_mat_ops_ew_abs->i[2] = 1;
data->test_mat_ops_ew_abs->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_ew_abs->p[0] = 0;
data->test_mat_ops_ew_abs->p[1] = 2;
data->test_mat_ops_ew_abs->p[2] = 3;

data->test_mat_vec_y = c_malloc(5 * sizeof(c_float));
data->test_mat_vec_y[0] = 0.61489158851395997463;
data->test_mat_vec_y[1] = 1.91843709387737848537;
data->test_mat_vec_y[2] = -0.82834649130743720491;
data->test_mat_vec_y[3] = -0.13489036838571524801;
data->test_mat_vec_y[4] = -0.12999649583570069278;
data->test_vec_ops_vec_prod = -3.38083119169354340272;
data->test_mat_vec_ATy = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_ATy[0] = 0.11300407501150980250;
data->test_mat_vec_ATy[1] = 1.49650596236691990626;
data->test_mat_vec_ATy[2] = -0.62351626553278372000;
data->test_mat_vec_ATy[3] = -0.45281599284086981871;
data->test_vec_ops_add_scaled = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_add_scaled[0] = 0.10735988051847006330;
data->test_vec_ops_add_scaled[1] = -0.53668897351778854787;
data->test_vec_ops_add_scaled[2] = 0.39189177029140009889;
data->test_vec_ops_add_scaled[3] = 1.42966838537337292969;
data->test_vec_ops_add_scaled[4] = -3.14950655221140474183;
data->test_vec_ops_add_scaled[5] = 1.51012760033499415790;
data->test_vec_ops_add_scaled[6] = 1.51061298790958420390;
data->test_vec_ops_add_scaled[7] = -0.80152515076205677058;
data->test_vec_ops_add_scaled[8] = 0.62533907645048214530;
data->test_vec_ops_add_scaled[9] = -0.73519430916230499307;
data->test_mat_extr_triu_n = 5;
data->test_mat_vec_m = 5;
data->test_vec_ops_ew_reciprocal = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_ew_reciprocal[0] = -2.04095933998456535718;
data->test_vec_ops_ew_reciprocal[1] = -1.65689874551252502854;
data->test_vec_ops_ew_reciprocal[2] = 6.59383542449817028341;
data->test_vec_ops_ew_reciprocal[3] = 0.79454517245095168665;
data->test_vec_ops_ew_reciprocal[4] = -0.60465652480618325981;
data->test_vec_ops_ew_reciprocal[5] = 0.58353277518326651663;
data->test_vec_ops_ew_reciprocal[6] = 1.09281060459281098574;
data->test_vec_ops_ew_reciprocal[7] = -3.24150493004670892105;
data->test_vec_ops_ew_reciprocal[8] = 1.05495019272169088254;
data->test_vec_ops_ew_reciprocal[9] = 7.54424387905824378464;
data->test_vec_ops_n = 10;

// Matrix test_mat_ops_prem_diag
//------------------------------
data->test_mat_ops_prem_diag = c_malloc(sizeof(csc));
data->test_mat_ops_prem_diag->m = 2;
data->test_mat_ops_prem_diag->n = 2;
data->test_mat_ops_prem_diag->nz = -1;
data->test_mat_ops_prem_diag->nzmax = 3;
data->test_mat_ops_prem_diag->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_prem_diag->x[0] = -0.23110842805233117181;
data->test_mat_ops_prem_diag->x[1] = 0.01671938209689918103;
data->test_mat_ops_prem_diag->x[2] = 0.60406128202899889157;
data->test_mat_ops_prem_diag->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_prem_diag->i[0] = 0;
data->test_mat_ops_prem_diag->i[1] = 1;
data->test_mat_ops_prem_diag->i[2] = 1;
data->test_mat_ops_prem_diag->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_prem_diag->p[0] = 0;
data->test_mat_ops_prem_diag->p[1] = 2;
data->test_mat_ops_prem_diag->p[2] = 3;


// Matrix test_mat_vec_Pu
//-----------------------
data->test_mat_vec_Pu = c_malloc(sizeof(csc));
data->test_mat_vec_Pu->m = 4;
data->test_mat_vec_Pu->n = 4;
data->test_mat_vec_Pu->nz = -1;
data->test_mat_vec_Pu->nzmax = 7;
data->test_mat_vec_Pu->x = c_malloc(7 * sizeof(c_float));
data->test_mat_vec_Pu->x[0] = 0.45837695782432152924;
data->test_mat_vec_Pu->x[1] = 1.41414668212411442916;
data->test_mat_vec_Pu->x[2] = 1.31918583493948626817;
data->test_mat_vec_Pu->x[3] = 1.15869658375588824306;
data->test_mat_vec_Pu->x[4] = 0.97053926327983675026;
data->test_mat_vec_Pu->x[5] = 1.27110985816231480960;
data->test_mat_vec_Pu->x[6] = 0.64301184563195246113;
data->test_mat_vec_Pu->i = c_malloc(7 * sizeof(c_int));
data->test_mat_vec_Pu->i[0] = 0;
data->test_mat_vec_Pu->i[1] = 1;
data->test_mat_vec_Pu->i[2] = 0;
data->test_mat_vec_Pu->i[3] = 1;
data->test_mat_vec_Pu->i[4] = 0;
data->test_mat_vec_Pu->i[5] = 1;
data->test_mat_vec_Pu->i[6] = 2;
data->test_mat_vec_Pu->p = c_malloc((4 + 1) * sizeof(c_int));
data->test_mat_vec_Pu->p[0] = 0;
data->test_mat_vec_Pu->p[1] = 0;
data->test_mat_vec_Pu->p[2] = 2;
data->test_mat_vec_Pu->p[3] = 4;
data->test_mat_vec_Pu->p[4] = 7;


// Matrix test_qpform_Pu
//----------------------
data->test_qpform_Pu = c_malloc(sizeof(csc));
data->test_qpform_Pu->m = 4;
data->test_qpform_Pu->n = 4;
data->test_qpform_Pu->nz = -1;
data->test_qpform_Pu->nzmax = 9;
data->test_qpform_Pu->x = c_malloc(9 * sizeof(c_float));
data->test_qpform_Pu->x[0] = 0.36952717012944757968;
data->test_qpform_Pu->x[1] = 1.06319959908247918534;
data->test_qpform_Pu->x[2] = 1.03086744012863218423;
data->test_qpform_Pu->x[3] = 0.27698701297690897505;
data->test_qpform_Pu->x[4] = 0.04104920176016979561;
data->test_qpform_Pu->x[5] = 1.05644136058804738454;
data->test_qpform_Pu->x[6] = 0.54028342718959077029;
data->test_qpform_Pu->x[7] = 0.14595666543830509987;
data->test_qpform_Pu->x[8] = 0.54113037948828024426;
data->test_qpform_Pu->i = c_malloc(9 * sizeof(c_int));
data->test_qpform_Pu->i[0] = 0;
data->test_qpform_Pu->i[1] = 0;
data->test_qpform_Pu->i[2] = 0;
data->test_qpform_Pu->i[3] = 1;
data->test_qpform_Pu->i[4] = 2;
data->test_qpform_Pu->i[5] = 0;
data->test_qpform_Pu->i[6] = 1;
data->test_qpform_Pu->i[7] = 2;
data->test_qpform_Pu->i[8] = 3;
data->test_qpform_Pu->p = c_malloc((4 + 1) * sizeof(c_int));
data->test_qpform_Pu->p[0] = 0;
data->test_qpform_Pu->p[1] = 1;
data->test_qpform_Pu->p[2] = 2;
data->test_qpform_Pu->p[3] = 5;
data->test_qpform_Pu->p[4] = 9;

data->test_sp_matrix_Adns = c_malloc(30 * sizeof(c_float));
data->test_sp_matrix_Adns[0] = 0.89633564821013733681;
data->test_sp_matrix_Adns[1] = -0.86135203505779533995;
data->test_sp_matrix_Adns[2] = -1.10089144467338706512;
data->test_sp_matrix_Adns[3] = 1.37787757416684386591;
data->test_sp_matrix_Adns[4] = -0.44483357029666292792;
data->test_sp_matrix_Adns[5] = 0.20847086947025550430;
data->test_sp_matrix_Adns[6] = 0.76527742100100404343;
data->test_sp_matrix_Adns[7] = -0.23902998143290987709;
data->test_sp_matrix_Adns[8] = 1.97797682528532492441;
data->test_sp_matrix_Adns[9] = -0.78035854354643940933;
data->test_sp_matrix_Adns[10] = 0.20276779166673114529;
data->test_sp_matrix_Adns[11] = 1.36258847737912924813;
data->test_sp_matrix_Adns[12] = -1.37277544886201230412;
data->test_sp_matrix_Adns[13] = 0.94974490834839631059;
data->test_sp_matrix_Adns[14] = -1.72460560726366285422;
data->test_sp_matrix_Adns[15] = 0.90549088145200518074;
data->test_sp_matrix_Adns[16] = -0.82103366081837236834;
data->test_sp_matrix_Adns[17] = -0.12602258633573590330;
data->test_sp_matrix_Adns[18] = -0.53137601952194546406;
data->test_sp_matrix_Adns[19] = 1.58969905211242545917;
data->test_sp_matrix_Adns[20] = 1.77289964277917988511;
data->test_sp_matrix_Adns[21] = 0.50847677961895987320;
data->test_sp_matrix_Adns[22] = 0.72516819874470306306;
data->test_sp_matrix_Adns[23] = -1.04338677468111340474;
data->test_sp_matrix_Adns[24] = 0.36534560394744641787;
data->test_sp_matrix_Adns[25] = -0.64684787223965534153;
data->test_sp_matrix_Adns[26] = 1.73013142131965014237;
data->test_sp_matrix_Adns[27] = 1.28635457091712779309;
data->test_sp_matrix_Adns[28] = 0.48462218677153662272;
data->test_sp_matrix_Adns[29] = 0.47227886025304549600;
data->test_vec_ops_v2 = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_v2[0] = -0.87055022163369921717;
data->test_vec_ops_v2[1] = -0.09742547142458872489;
data->test_vec_ops_v2[2] = -0.35012164745106849617;
data->test_vec_ops_v2[3] = -0.24934404399022347398;
data->test_vec_ops_v2[4] = 2.17981679989343923864;
data->test_vec_ops_v2[5] = 0.29668884030961856224;
data->test_vec_ops_v2[6] = -0.86794989794447618969;
data->test_vec_ops_v2[7] = 0.71854335477495256956;
data->test_vec_ops_v2[8] = 0.47012215687329800273;
data->test_vec_ops_v2[9] = 1.26466416025621208874;
data->test_mat_ops_n = 2;
data->test_mat_vec_x = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_x[0] = 1.01899515767079162920;
data->test_mat_vec_x[1] = 1.90980719344012928040;
data->test_mat_vec_x[2] = 1.17297598641427902599;
data->test_mat_vec_x[3] = 1.30954774924094530242;
data->test_mat_vec_Px_cum = c_malloc(4 * sizeof(c_float));
data->test_mat_vec_Px_cum[0] = 4.71274758281084693579;
data->test_mat_vec_Px_cum[1] = 8.10134092203146138900;
data->test_mat_vec_Px_cum[2] = 5.57216175013960413054;
data->test_mat_vec_Px_cum[3] = 5.48033476352973369927;
data->test_qpform_x = c_malloc(4 * sizeof(c_float));
data->test_qpform_x[0] = -0.89998969901960557127;
data->test_qpform_x[1] = 1.87296215804183985298;
data->test_qpform_x[2] = -0.35931297150226731985;
data->test_qpform_x[3] = 0.40713313308628318321;
data->test_vec_ops_norm2 = 3.11960115978473817577;

// Matrix test_mat_ops_ew_square
//------------------------------
data->test_mat_ops_ew_square = c_malloc(sizeof(csc));
data->test_mat_ops_ew_square->m = 2;
data->test_mat_ops_ew_square->n = 2;
data->test_mat_ops_ew_square->nz = -1;
data->test_mat_ops_ew_square->nzmax = 3;
data->test_mat_ops_ew_square->x = c_malloc(3 * sizeof(c_float));
data->test_mat_ops_ew_square->x[0] = 0.38970146460467325333;
data->test_mat_ops_ew_square->x[1] = 0.00027742967931541847;
data->test_mat_ops_ew_square->x[2] = 0.36213831277015773313;
data->test_mat_ops_ew_square->i = c_malloc(3 * sizeof(c_int));
data->test_mat_ops_ew_square->i[0] = 0;
data->test_mat_ops_ew_square->i[1] = 1;
data->test_mat_ops_ew_square->i[2] = 1;
data->test_mat_ops_ew_square->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_mat_ops_ew_square->p[0] = 0;
data->test_mat_ops_ew_square->p[1] = 2;
data->test_mat_ops_ew_square->p[2] = 3;


// Matrix test_mat_extr_triu_P
//----------------------------
data->test_mat_extr_triu_P = c_malloc(sizeof(csc));
data->test_mat_extr_triu_P->m = 5;
data->test_mat_extr_triu_P->n = 5;
data->test_mat_extr_triu_P->nz = -1;
data->test_mat_extr_triu_P->nzmax = 22;
data->test_mat_extr_triu_P->x = c_malloc(22 * sizeof(c_float));
data->test_mat_extr_triu_P->x[0] = 0.68383978902610864647;
data->test_mat_extr_triu_P->x[1] = 0.95537784067153719292;
data->test_mat_extr_triu_P->x[2] = 0.44285664799195045838;
data->test_mat_extr_triu_P->x[3] = 0.96875476692783191179;
data->test_mat_extr_triu_P->x[4] = 0.68383978902610864647;
data->test_mat_extr_triu_P->x[5] = 0.39644573158425067128;
data->test_mat_extr_triu_P->x[6] = 1.08739232294260412814;
data->test_mat_extr_triu_P->x[7] = 1.34488394490581142371;
data->test_mat_extr_triu_P->x[8] = 0.35007311044698630198;
data->test_mat_extr_triu_P->x[9] = 0.95537784067153719292;
data->test_mat_extr_triu_P->x[10] = 1.08739232294260412814;
data->test_mat_extr_triu_P->x[11] = 0.93675794185869942776;
data->test_mat_extr_triu_P->x[12] = 0.17313530313010705441;
data->test_mat_extr_triu_P->x[13] = 0.44285664799195045838;
data->test_mat_extr_triu_P->x[14] = 1.34488394490581142371;
data->test_mat_extr_triu_P->x[15] = 0.93675794185869942776;
data->test_mat_extr_triu_P->x[16] = 1.01546385911073788755;
data->test_mat_extr_triu_P->x[17] = 1.70715055043758034969;
data->test_mat_extr_triu_P->x[18] = 0.96875476692783191179;
data->test_mat_extr_triu_P->x[19] = 0.35007311044698630198;
data->test_mat_extr_triu_P->x[20] = 0.17313530313010705441;
data->test_mat_extr_triu_P->x[21] = 1.70715055043758034969;
data->test_mat_extr_triu_P->i = c_malloc(22 * sizeof(c_int));
data->test_mat_extr_triu_P->i[0] = 1;
data->test_mat_extr_triu_P->i[1] = 2;
data->test_mat_extr_triu_P->i[2] = 3;
data->test_mat_extr_triu_P->i[3] = 4;
data->test_mat_extr_triu_P->i[4] = 0;
data->test_mat_extr_triu_P->i[5] = 1;
data->test_mat_extr_triu_P->i[6] = 2;
data->test_mat_extr_triu_P->i[7] = 3;
data->test_mat_extr_triu_P->i[8] = 4;
data->test_mat_extr_triu_P->i[9] = 0;
data->test_mat_extr_triu_P->i[10] = 1;
data->test_mat_extr_triu_P->i[11] = 3;
data->test_mat_extr_triu_P->i[12] = 4;
data->test_mat_extr_triu_P->i[13] = 0;
data->test_mat_extr_triu_P->i[14] = 1;
data->test_mat_extr_triu_P->i[15] = 2;
data->test_mat_extr_triu_P->i[16] = 3;
data->test_mat_extr_triu_P->i[17] = 4;
data->test_mat_extr_triu_P->i[18] = 0;
data->test_mat_extr_triu_P->i[19] = 1;
data->test_mat_extr_triu_P->i[20] = 2;
data->test_mat_extr_triu_P->i[21] = 3;
data->test_mat_extr_triu_P->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_mat_extr_triu_P->p[0] = 0;
data->test_mat_extr_triu_P->p[1] = 4;
data->test_mat_extr_triu_P->p[2] = 9;
data->test_mat_extr_triu_P->p[3] = 13;
data->test_mat_extr_triu_P->p[4] = 18;
data->test_mat_extr_triu_P->p[5] = 22;


// Matrix test_mat_vec_A
//----------------------
data->test_mat_vec_A = c_malloc(sizeof(csc));
data->test_mat_vec_A->m = 5;
data->test_mat_vec_A->n = 4;
data->test_mat_vec_A->nz = -1;
data->test_mat_vec_A->nzmax = 20;
data->test_mat_vec_A->x = c_malloc(20 * sizeof(c_float));
data->test_mat_vec_A->x[0] = 0.94211551059861875501;
data->test_mat_vec_A->x[1] = 0.06328573200029030676;
data->test_mat_vec_A->x[2] = 0.54964878404281203306;
data->test_mat_vec_A->x[3] = 0.57998565512165334290;
data->test_mat_vec_A->x[4] = 0.41670664203184870633;
data->test_mat_vec_A->x[5] = 0.84183037471454147394;
data->test_mat_vec_A->x[6] = 0.77764602291246898158;
data->test_mat_vec_A->x[7] = 0.34147878560508826418;
data->test_mat_vec_A->x[8] = 0.77721715028795290703;
data->test_mat_vec_A->x[9] = 0.96380727670167731791;
data->test_mat_vec_A->x[10] = 0.06751824103729442417;
data->test_mat_vec_A->x[11] = 0.09527539233202808600;
data->test_mat_vec_A->x[12] = 0.80152368916016236700;
data->test_mat_vec_A->x[13] = 0.63585176235844198533;
data->test_mat_vec_A->x[14] = 0.75465800898584944889;
data->test_mat_vec_A->x[15] = 0.51558585454244398871;
data->test_mat_vec_A->x[16] = 0.00139337675376516312;
data->test_mat_vec_A->x[17] = 0.78618018262801936391;
data->test_mat_vec_A->x[18] = 0.26035029888714700252;
data->test_mat_vec_A->x[19] = 0.66286525271650553748;
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

data->test_mat_vec_Ax = c_malloc(5 * sizeof(c_float));
data->test_mat_vec_Ax[0] = 3.32212641929097340920;
data->test_mat_vec_Ax[1] = 1.66322226365969405570;
data->test_mat_vec_Ax[2] = 3.18195661910047311594;
data->test_mat_vec_Ax[3] = 3.16211747465997872197;
data->test_mat_vec_Ax[4] = 4.01855754277505639038;
data->test_mat_vec_n = 4;

// Matrix test_mat_extr_triu_Pu
//-----------------------------
data->test_mat_extr_triu_Pu = c_malloc(sizeof(csc));
data->test_mat_extr_triu_Pu->m = 5;
data->test_mat_extr_triu_Pu->n = 5;
data->test_mat_extr_triu_Pu->nz = -1;
data->test_mat_extr_triu_Pu->nzmax = 12;
data->test_mat_extr_triu_Pu->x = c_malloc(12 * sizeof(c_float));
data->test_mat_extr_triu_Pu->x[0] = 0.68383978902610864647;
data->test_mat_extr_triu_Pu->x[1] = 0.39644573158425067128;
data->test_mat_extr_triu_Pu->x[2] = 0.95537784067153719292;
data->test_mat_extr_triu_Pu->x[3] = 1.08739232294260412814;
data->test_mat_extr_triu_Pu->x[4] = 0.44285664799195045838;
data->test_mat_extr_triu_Pu->x[5] = 1.34488394490581142371;
data->test_mat_extr_triu_Pu->x[6] = 0.93675794185869942776;
data->test_mat_extr_triu_Pu->x[7] = 1.01546385911073788755;
data->test_mat_extr_triu_Pu->x[8] = 0.96875476692783191179;
data->test_mat_extr_triu_Pu->x[9] = 0.35007311044698630198;
data->test_mat_extr_triu_Pu->x[10] = 0.17313530313010705441;
data->test_mat_extr_triu_Pu->x[11] = 1.70715055043758034969;
data->test_mat_extr_triu_Pu->i = c_malloc(12 * sizeof(c_int));
data->test_mat_extr_triu_Pu->i[0] = 0;
data->test_mat_extr_triu_Pu->i[1] = 1;
data->test_mat_extr_triu_Pu->i[2] = 0;
data->test_mat_extr_triu_Pu->i[3] = 1;
data->test_mat_extr_triu_Pu->i[4] = 0;
data->test_mat_extr_triu_Pu->i[5] = 1;
data->test_mat_extr_triu_Pu->i[6] = 2;
data->test_mat_extr_triu_Pu->i[7] = 3;
data->test_mat_extr_triu_Pu->i[8] = 0;
data->test_mat_extr_triu_Pu->i[9] = 1;
data->test_mat_extr_triu_Pu->i[10] = 2;
data->test_mat_extr_triu_Pu->i[11] = 3;
data->test_mat_extr_triu_Pu->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_mat_extr_triu_Pu->p[0] = 0;
data->test_mat_extr_triu_Pu->p[1] = 0;
data->test_mat_extr_triu_Pu->p[2] = 2;
data->test_mat_extr_triu_Pu->p[3] = 4;
data->test_mat_extr_triu_Pu->p[4] = 8;
data->test_mat_extr_triu_Pu->p[5] = 12;

data->test_qpform_n = 4;
data->test_vec_ops_v1 = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_v1[0] = -0.48996566487579434890;
data->test_vec_ops_v1[1] = -0.60353718216539065100;
data->test_vec_ops_v1[2] = 0.15165680300188957852;
data->test_vec_ops_v1[3] = 1.25858168254333113723;
data->test_vec_ops_v1[4] = -1.65383148775338240100;
data->test_vec_ops_v1[5] = 1.71369979978577258173;
data->test_vec_ops_v1[6] = 0.91507164717952860222;
data->test_vec_ops_v1[7] = -0.30849868242698935683;
data->test_vec_ops_v1[8] = 0.94791205016046919951;
data->test_vec_ops_v1[9] = 0.13255138832081753142;

return data;

}

/* function to clean data struct */
void clean_problem_lin_alg_sols_data(lin_alg_sols_data * data){

c_free(data->test_mat_ops_postm_diag->x);
c_free(data->test_mat_ops_postm_diag->i);
c_free(data->test_mat_ops_postm_diag->p);
c_free(data->test_mat_ops_postm_diag);
c_free(data->test_sp_matrix_A->x);
c_free(data->test_sp_matrix_A->i);
c_free(data->test_sp_matrix_A->p);
c_free(data->test_sp_matrix_A);
c_free(data->test_mat_vec_Px);
c_free(data->test_mat_ops_A->x);
c_free(data->test_mat_ops_A->i);
c_free(data->test_mat_ops_A->p);
c_free(data->test_mat_ops_A);
c_free(data->test_mat_vec_ATy_cum);
c_free(data->test_mat_vec_Ax_cum);
c_free(data->test_mat_ops_d);
c_free(data->test_mat_ops_ew_abs->x);
c_free(data->test_mat_ops_ew_abs->i);
c_free(data->test_mat_ops_ew_abs->p);
c_free(data->test_mat_ops_ew_abs);
c_free(data->test_mat_vec_y);
c_free(data->test_mat_vec_ATy);
c_free(data->test_vec_ops_add_scaled);
c_free(data->test_vec_ops_ew_reciprocal);
c_free(data->test_mat_ops_prem_diag->x);
c_free(data->test_mat_ops_prem_diag->i);
c_free(data->test_mat_ops_prem_diag->p);
c_free(data->test_mat_ops_prem_diag);
c_free(data->test_mat_vec_Pu->x);
c_free(data->test_mat_vec_Pu->i);
c_free(data->test_mat_vec_Pu->p);
c_free(data->test_mat_vec_Pu);
c_free(data->test_qpform_Pu->x);
c_free(data->test_qpform_Pu->i);
c_free(data->test_qpform_Pu->p);
c_free(data->test_qpform_Pu);
c_free(data->test_sp_matrix_Adns);
c_free(data->test_vec_ops_v2);
c_free(data->test_mat_vec_x);
c_free(data->test_mat_vec_Px_cum);
c_free(data->test_qpform_x);
c_free(data->test_mat_ops_ew_square->x);
c_free(data->test_mat_ops_ew_square->i);
c_free(data->test_mat_ops_ew_square->p);
c_free(data->test_mat_ops_ew_square);
c_free(data->test_mat_extr_triu_P->x);
c_free(data->test_mat_extr_triu_P->i);
c_free(data->test_mat_extr_triu_P->p);
c_free(data->test_mat_extr_triu_P);
c_free(data->test_mat_vec_A->x);
c_free(data->test_mat_vec_A->i);
c_free(data->test_mat_vec_A->p);
c_free(data->test_mat_vec_A);
c_free(data->test_mat_vec_Ax);
c_free(data->test_mat_extr_triu_Pu->x);
c_free(data->test_mat_extr_triu_Pu->i);
c_free(data->test_mat_extr_triu_Pu->p);
c_free(data->test_mat_extr_triu_Pu);
c_free(data->test_vec_ops_v1);

c_free(data);

}

#endif
