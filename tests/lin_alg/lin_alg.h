#ifndef LIN_ALG_DATA_H
#define LIN_ALG_DATA_H
#include "osqp.h"


/* create data and solutions structure */
typedef struct {
csc * test_sp_matrix_A;
c_int test_vec_ops_n;
c_float test_vec_ops_norm2_diff;
c_float * test_vec_ops_v2;
c_float * test_vec_ops_ew_reciprocal;
c_float * test_vec_ops_v1;
c_float * test_vec_ops_add_scaled;
c_float test_vec_ops_vec_prod;
c_float test_vec_ops_sc;
c_float * test_sp_matrix_Adns;
c_float test_vec_ops_norm2;
} lin_alg_sols_data;

/* function to define problem data */
lin_alg_sols_data *  generate_problem_lin_alg_sols_data(){

lin_alg_sols_data * data = (lin_alg_sols_data *)c_malloc(sizeof(lin_alg_sols_data));

// Matrix test_sp_matrix_A
//------------------------
data->test_sp_matrix_A = c_malloc(sizeof(csc));
data->test_sp_matrix_A->m = 5;
data->test_sp_matrix_A->n = 6;
data->test_sp_matrix_A->nz = -1;
data->test_sp_matrix_A->nzmax = 30;
data->test_sp_matrix_A->x = c_malloc(30 * sizeof(c_float));
data->test_sp_matrix_A->x[0] = -0.14944216512611632752;
data->test_sp_matrix_A->x[1] = -0.91908293415428243822;
data->test_sp_matrix_A->x[2] = 1.29132174843526414953;
data->test_sp_matrix_A->x[3] = 0.41442391922839050267;
data->test_sp_matrix_A->x[4] = 0.45299840307371253001;
data->test_sp_matrix_A->x[5] = 0.05392541028115958868;
data->test_sp_matrix_A->x[6] = -0.52592937576661891175;
data->test_sp_matrix_A->x[7] = -0.39277688353490236572;
data->test_sp_matrix_A->x[8] = 1.05844197766046166365;
data->test_sp_matrix_A->x[9] = 0.48823077729138442127;
data->test_sp_matrix_A->x[10] = -0.72315285780301374885;
data->test_sp_matrix_A->x[11] = -0.85910022888624970605;
data->test_sp_matrix_A->x[12] = -1.30506233112006753139;
data->test_sp_matrix_A->x[13] = -3.06624403533876410677;
data->test_sp_matrix_A->x[14] = -0.96773764404391782179;
data->test_sp_matrix_A->x[15] = -0.53702717305104841206;
data->test_sp_matrix_A->x[16] = -1.17635129348889444501;
data->test_sp_matrix_A->x[17] = -0.52514085482714367714;
data->test_sp_matrix_A->x[18] = -1.11713897365436487519;
data->test_sp_matrix_A->x[19] = -0.68304108376472727482;
data->test_sp_matrix_A->x[20] = 1.46290389332237724673;
data->test_sp_matrix_A->x[21] = 0.93900941185027952951;
data->test_sp_matrix_A->x[22] = -0.78081995980376406319;
data->test_sp_matrix_A->x[23] = 0.33063887330076657811;
data->test_sp_matrix_A->x[24] = -1.50501391236453430089;
data->test_sp_matrix_A->x[25] = -0.94563336163526967582;
data->test_sp_matrix_A->x[26] = -1.42742990837310035346;
data->test_sp_matrix_A->x[27] = 1.34188026075148369998;
data->test_sp_matrix_A->x[28] = -0.31393321766185744437;
data->test_sp_matrix_A->x[29] = 1.62249107296228101127;
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

data->test_vec_ops_n = 10;
data->test_vec_ops_norm2_diff = 4.37758307973055327267;
data->test_vec_ops_v2 = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_v2[0] = -0.34832122274031035447;
data->test_vec_ops_v2[1] = -1.18116831081805218062;
data->test_vec_ops_v2[2] = 0.46076773974659607225;
data->test_vec_ops_v2[3] = 0.45027639585367346120;
data->test_vec_ops_v2[4] = -0.53835617654709821966;
data->test_vec_ops_v2[5] = 0.99317020974717529214;
data->test_vec_ops_v2[6] = -0.41340330898124866277;
data->test_vec_ops_v2[7] = -3.18998774326313006000;
data->test_vec_ops_v2[8] = 1.64998362934882436548;
data->test_vec_ops_v2[9] = 0.53973515912156089236;
data->test_vec_ops_ew_reciprocal = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_ew_reciprocal[0] = 0.57411036915597846164;
data->test_vec_ops_ew_reciprocal[1] = 4.72210203374159487311;
data->test_vec_ops_ew_reciprocal[2] = 0.82382936734021861014;
data->test_vec_ops_ew_reciprocal[3] = -2.52726278194800269006;
data->test_vec_ops_ew_reciprocal[4] = 1.21815904593232504638;
data->test_vec_ops_ew_reciprocal[5] = 1.85066387058499226548;
data->test_vec_ops_ew_reciprocal[6] = -0.91383313091800910666;
data->test_vec_ops_ew_reciprocal[7] = -0.63775699820491049685;
data->test_vec_ops_ew_reciprocal[8] = -1.20158544018761626049;
data->test_vec_ops_ew_reciprocal[9] = 37.09660808568848722189;
data->test_vec_ops_v1 = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_v1[0] = 1.74182535924257586579;
data->test_vec_ops_v1[1] = 0.21177009578669397927;
data->test_vec_ops_v1[2] = 1.21384359388468832108;
data->test_vec_ops_v1[3] = -0.39568501033723313398;
data->test_vec_ops_v1[4] = 0.82091086819836744937;
data->test_vec_ops_v1[5] = 0.54034663770893276435;
data->test_vec_ops_v1[6] = -1.09429168867562309764;
data->test_vec_ops_v1[7] = -1.56799533805931101682;
data->test_vec_ops_v1[8] = -0.83223378592525176778;
data->test_vec_ops_v1[9] = 0.02695664244262241035;
data->test_vec_ops_add_scaled = c_malloc(10 * sizeof(c_float));
data->test_vec_ops_add_scaled[0] = 1.70962031519809820601;
data->test_vec_ops_add_scaled[1] = 0.10256177028022045450;
data->test_vec_ops_add_scaled[2] = 1.25644520522513980509;
data->test_vec_ops_add_scaled[3] = -0.35405340647866628823;
data->test_vec_ops_add_scaled[4] = 0.77113559398189501781;
data->test_vec_ops_add_scaled[5] = 0.63217305614229157840;
data->test_vec_ops_add_scaled[6] = -1.13251408485674787130;
data->test_vec_ops_add_scaled[7] = -1.86293486245839079452;
data->test_vec_ops_add_scaled[8] = -0.67967978695326625171;
data->test_vec_ops_add_scaled[9] = 0.07685941447331855192;
data->test_vec_ops_vec_prod = 3.71464227090149279320;
data->test_vec_ops_sc = 0.09245788640472257791;
data->test_sp_matrix_Adns = c_malloc(30 * sizeof(c_float));
data->test_sp_matrix_Adns[0] = -0.14944216512611632752;
data->test_sp_matrix_Adns[1] = -0.91908293415428243822;
data->test_sp_matrix_Adns[2] = 1.29132174843526414953;
data->test_sp_matrix_Adns[3] = 0.41442391922839050267;
data->test_sp_matrix_Adns[4] = 0.45299840307371253001;
data->test_sp_matrix_Adns[5] = 0.05392541028115958868;
data->test_sp_matrix_Adns[6] = -0.52592937576661891175;
data->test_sp_matrix_Adns[7] = -0.39277688353490236572;
data->test_sp_matrix_Adns[8] = 1.05844197766046166365;
data->test_sp_matrix_Adns[9] = 0.48823077729138442127;
data->test_sp_matrix_Adns[10] = -0.72315285780301374885;
data->test_sp_matrix_Adns[11] = -0.85910022888624970605;
data->test_sp_matrix_Adns[12] = -1.30506233112006753139;
data->test_sp_matrix_Adns[13] = -3.06624403533876410677;
data->test_sp_matrix_Adns[14] = -0.96773764404391782179;
data->test_sp_matrix_Adns[15] = -0.53702717305104841206;
data->test_sp_matrix_Adns[16] = -1.17635129348889444501;
data->test_sp_matrix_Adns[17] = -0.52514085482714367714;
data->test_sp_matrix_Adns[18] = -1.11713897365436487519;
data->test_sp_matrix_Adns[19] = -0.68304108376472727482;
data->test_sp_matrix_Adns[20] = 1.46290389332237724673;
data->test_sp_matrix_Adns[21] = 0.93900941185027952951;
data->test_sp_matrix_Adns[22] = -0.78081995980376406319;
data->test_sp_matrix_Adns[23] = 0.33063887330076657811;
data->test_sp_matrix_Adns[24] = -1.50501391236453430089;
data->test_sp_matrix_Adns[25] = -0.94563336163526967582;
data->test_sp_matrix_Adns[26] = -1.42742990837310035346;
data->test_sp_matrix_Adns[27] = 1.34188026075148369998;
data->test_sp_matrix_Adns[28] = -0.31393321766185744437;
data->test_sp_matrix_Adns[29] = 1.62249107296228101127;
data->test_vec_ops_norm2 = 3.16608237588468188761;

return data;

}

/* function to clean data struct */
void clean_problem_lin_alg_sols_data(lin_alg_sols_data * data){

c_free(data->test_sp_matrix_A->x);
c_free(data->test_sp_matrix_A->i);
c_free(data->test_sp_matrix_A->p);
c_free(data->test_sp_matrix_A);
c_free(data->test_vec_ops_v2);
c_free(data->test_vec_ops_ew_reciprocal);
c_free(data->test_vec_ops_v1);
c_free(data->test_vec_ops_add_scaled);
c_free(data->test_sp_matrix_Adns);

c_free(data);

}

#endif
