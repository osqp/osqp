#include "lin_alg_data.h"


/* function to define problem data */
lin_alg_sols_data *  generate_problem_lin_alg_sols_data(){

lin_alg_sols_data * data = (lin_alg_sols_data *)c_malloc(sizeof(lin_alg_sols_data));


// Matrix test_sp_matrix_A
//------------------------
data->test_sp_matrix_A = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_sp_matrix_A->m = 5;
data->test_sp_matrix_A->n = 6;
data->test_sp_matrix_A->nz = -1;
data->test_sp_matrix_A->nzmax = 30;
data->test_sp_matrix_A->x = (OSQPFloat*) c_malloc(30 * sizeof(OSQPFloat));
data->test_sp_matrix_A->x[0] = -0.12258433521470017691;
data->test_sp_matrix_A->x[1] = 0.66306337237626167269;
data->test_sp_matrix_A->x[2] = -0.68322666178056223885;
data->test_sp_matrix_A->x[3] = -0.50629165831431477418;
data->test_sp_matrix_A->x[4] = 0.49855998153294767139;
data->test_sp_matrix_A->x[5] = 3.11783875505104823844;
data->test_sp_matrix_A->x[6] = -0.51400637168746288186;
data->test_sp_matrix_A->x[7] = -0.07204367972722743041;
data->test_sp_matrix_A->x[8] = 0.59374807178582278411;
data->test_sp_matrix_A->x[9] = 0.87916061828798530708;
data->test_sp_matrix_A->x[10] = -1.11202076269228133931;
data->test_sp_matrix_A->x[11] = -1.64807517085565269355;
data->test_sp_matrix_A->x[12] = -0.94475162306077742347;
data->test_sp_matrix_A->x[13] = 0.89116695428232839404;
data->test_sp_matrix_A->x[14] = -1.07178741687744416566;
data->test_sp_matrix_A->x[15] = 0.62239499287300192876;
data->test_sp_matrix_A->x[16] = 0.16746474422274112981;
data->test_sp_matrix_A->x[17] = -0.09826996785221726871;
data->test_sp_matrix_A->x[18] = 0.32084830456656371345;
data->test_sp_matrix_A->x[19] = 0.91446720312878115866;
data->test_sp_matrix_A->x[20] = 2.04277160749233033243;
data->test_sp_matrix_A->x[21] = 0.10901408782154753396;
data->test_sp_matrix_A->x[22] = 0.09548302746945433461;
data->test_sp_matrix_A->x[23] = -0.81823022739030704109;
data->test_sp_matrix_A->x[24] = -0.02006345461548042150;
data->test_sp_matrix_A->x[25] = 0.64670299620184690248;
data->test_sp_matrix_A->x[26] = -1.22735205424457416434;
data->test_sp_matrix_A->x[27] = 0.03558623705548571298;
data->test_sp_matrix_A->x[28] = 1.73165228378544089338;
data->test_sp_matrix_A->x[29] = -0.24874889033441549557;
data->test_sp_matrix_A->i = (OSQPInt*) c_malloc(30 * sizeof(OSQPInt));
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
data->test_sp_matrix_A->p = (OSQPInt*) c_malloc((6 + 1) * sizeof(OSQPInt));
data->test_sp_matrix_A->p[0] = 0;
data->test_sp_matrix_A->p[1] = 5;
data->test_sp_matrix_A->p[2] = 10;
data->test_sp_matrix_A->p[3] = 15;
data->test_sp_matrix_A->p[4] = 20;
data->test_sp_matrix_A->p[5] = 25;
data->test_sp_matrix_A->p[6] = 30;

data->test_sp_matrix_Adns = (OSQPFloat*) c_malloc(30 * sizeof(OSQPFloat));
data->test_sp_matrix_Adns[0] = -0.12258433521470017691;
data->test_sp_matrix_Adns[1] = 0.66306337237626167269;
data->test_sp_matrix_Adns[2] = -0.68322666178056223885;
data->test_sp_matrix_Adns[3] = -0.50629165831431477418;
data->test_sp_matrix_Adns[4] = 0.49855998153294767139;
data->test_sp_matrix_Adns[5] = 3.11783875505104823844;
data->test_sp_matrix_Adns[6] = -0.51400637168746288186;
data->test_sp_matrix_Adns[7] = -0.07204367972722743041;
data->test_sp_matrix_Adns[8] = 0.59374807178582278411;
data->test_sp_matrix_Adns[9] = 0.87916061828798530708;
data->test_sp_matrix_Adns[10] = -1.11202076269228133931;
data->test_sp_matrix_Adns[11] = -1.64807517085565269355;
data->test_sp_matrix_Adns[12] = -0.94475162306077742347;
data->test_sp_matrix_Adns[13] = 0.89116695428232839404;
data->test_sp_matrix_Adns[14] = -1.07178741687744416566;
data->test_sp_matrix_Adns[15] = 0.62239499287300192876;
data->test_sp_matrix_Adns[16] = 0.16746474422274112981;
data->test_sp_matrix_Adns[17] = -0.09826996785221726871;
data->test_sp_matrix_Adns[18] = 0.32084830456656371345;
data->test_sp_matrix_Adns[19] = 0.91446720312878115866;
data->test_sp_matrix_Adns[20] = 2.04277160749233033243;
data->test_sp_matrix_Adns[21] = 0.10901408782154753396;
data->test_sp_matrix_Adns[22] = 0.09548302746945433461;
data->test_sp_matrix_Adns[23] = -0.81823022739030704109;
data->test_sp_matrix_Adns[24] = -0.02006345461548042150;
data->test_sp_matrix_Adns[25] = 0.64670299620184690248;
data->test_sp_matrix_Adns[26] = -1.22735205424457416434;
data->test_sp_matrix_Adns[27] = 0.03558623705548571298;
data->test_sp_matrix_Adns[28] = 1.73165228378544089338;
data->test_sp_matrix_Adns[29] = -0.24874889033441549557;

data->test_vec_ops_n = 10;
data->test_vec_ops_vn = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_vn[0] = 10.00000000000000000000;
data->test_vec_ops_vn[1] = 10.00000000000000000000;
data->test_vec_ops_vn[2] = 10.00000000000000000000;
data->test_vec_ops_vn[3] = 10.00000000000000000000;
data->test_vec_ops_vn[4] = 10.00000000000000000000;
data->test_vec_ops_vn[5] = 10.00000000000000000000;
data->test_vec_ops_vn[6] = 10.00000000000000000000;
data->test_vec_ops_vn[7] = 10.00000000000000000000;
data->test_vec_ops_vn[8] = 10.00000000000000000000;
data->test_vec_ops_vn[9] = 10.00000000000000000000;

data->test_vec_ops_vn_neg = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_vn_neg[0] = -10.00000000000000000000;
data->test_vec_ops_vn_neg[1] = -10.00000000000000000000;
data->test_vec_ops_vn_neg[2] = -10.00000000000000000000;
data->test_vec_ops_vn_neg[3] = -10.00000000000000000000;
data->test_vec_ops_vn_neg[4] = -10.00000000000000000000;
data->test_vec_ops_vn_neg[5] = -10.00000000000000000000;
data->test_vec_ops_vn_neg[6] = -10.00000000000000000000;
data->test_vec_ops_vn_neg[7] = -10.00000000000000000000;
data->test_vec_ops_vn_neg[8] = -10.00000000000000000000;
data->test_vec_ops_vn_neg[9] = -10.00000000000000000000;

data->test_vec_ops_ones = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_ones[0] = 1.00000000000000000000;
data->test_vec_ops_ones[1] = 1.00000000000000000000;
data->test_vec_ops_ones[2] = 1.00000000000000000000;
data->test_vec_ops_ones[3] = 1.00000000000000000000;
data->test_vec_ops_ones[4] = 1.00000000000000000000;
data->test_vec_ops_ones[5] = 1.00000000000000000000;
data->test_vec_ops_ones[6] = 1.00000000000000000000;
data->test_vec_ops_ones[7] = 1.00000000000000000000;
data->test_vec_ops_ones[8] = 1.00000000000000000000;
data->test_vec_ops_ones[9] = 1.00000000000000000000;

data->test_vec_ops_zero = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_zero[0] = 0.00000000000000000000;
data->test_vec_ops_zero[1] = 0.00000000000000000000;
data->test_vec_ops_zero[2] = 0.00000000000000000000;
data->test_vec_ops_zero[3] = 0.00000000000000000000;
data->test_vec_ops_zero[4] = 0.00000000000000000000;
data->test_vec_ops_zero[5] = 0.00000000000000000000;
data->test_vec_ops_zero[6] = 0.00000000000000000000;
data->test_vec_ops_zero[7] = 0.00000000000000000000;
data->test_vec_ops_zero[8] = 0.00000000000000000000;
data->test_vec_ops_zero[9] = 0.00000000000000000000;

data->test_vec_ops_zero_int = (OSQPInt*) c_malloc(10 * sizeof(OSQPInt));
data->test_vec_ops_zero_int[0] = 0;
data->test_vec_ops_zero_int[1] = 0;
data->test_vec_ops_zero_int[2] = 0;
data->test_vec_ops_zero_int[3] = 0;
data->test_vec_ops_zero_int[4] = 0;
data->test_vec_ops_zero_int[5] = 0;
data->test_vec_ops_zero_int[6] = 0;
data->test_vec_ops_zero_int[7] = 0;
data->test_vec_ops_zero_int[8] = 0;
data->test_vec_ops_zero_int[9] = 0;

data->test_vec_ops_v1 = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_v1[0] = -0.31389947196684775399;
data->test_vec_ops_v1[1] = 0.05410227877154388798;
data->test_vec_ops_v1[2] = 0.27279133916445374997;
data->test_vec_ops_v1[3] = -0.98218812494097773591;
data->test_vec_ops_v1[4] = -1.10737304716519302517;
data->test_vec_ops_v1[5] = 0.19958453284708083109;
data->test_vec_ops_v1[6] = -0.46674961687980204283;
data->test_vec_ops_v1[7] = 0.23550561173022521722;
data->test_vec_ops_v1[8] = 0.75951952247837917209;
data->test_vec_ops_v1[9] = -1.64878736635094846896;

data->test_vec_ops_v2 = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_v2[0] = 0.25438811651761727983;
data->test_vec_ops_v2[1] = 1.22464696753573232257;
data->test_vec_ops_v2[2] = -0.29752684437047322019;
data->test_vec_ops_v2[3] = -0.81081458323756994133;
data->test_vec_ops_v2[4] = 0.75224382717959281663;
data->test_vec_ops_v2[5] = 0.25344651620814145909;
data->test_vec_ops_v2[6] = 0.89588307077756035302;
data->test_vec_ops_v2[7] = -0.34521571005127971166;
data->test_vec_ops_v2[8] = -1.48181827372221119887;
data->test_vec_ops_v2[9] = -0.11001076471125098566;

data->test_vec_ops_v3 = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_v3[0] = -0.44582815301123218665;
data->test_vec_ops_v3[1] = 0.77532382204757410715;
data->test_vec_ops_v3[2] = 0.19363284837715380449;
data->test_vec_ops_v3[3] = -1.63084923243510115931;
data->test_vec_ops_v3[4] = -1.19516308010319982635;
data->test_vec_ops_v3[5] = 0.88378903658725527226;
data->test_vec_ops_v3[6] = 0.67976501741784656208;
data->test_vec_ops_v3[7] = -0.64024336590848873740;
data->test_vec_ops_v3[8] = -0.00104879656728068104;
data->test_vec_ops_v3[9] = 0.44557355377618612646;

data->test_vec_ops_neg_v1 = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_neg_v1[0] = 0.31389947196684775399;
data->test_vec_ops_neg_v1[1] = -0.05410227877154388798;
data->test_vec_ops_neg_v1[2] = -0.27279133916445374997;
data->test_vec_ops_neg_v1[3] = 0.98218812494097773591;
data->test_vec_ops_neg_v1[4] = 1.10737304716519302517;
data->test_vec_ops_neg_v1[5] = -0.19958453284708083109;
data->test_vec_ops_neg_v1[6] = 0.46674961687980204283;
data->test_vec_ops_neg_v1[7] = -0.23550561173022521722;
data->test_vec_ops_neg_v1[8] = -0.75951952247837917209;
data->test_vec_ops_neg_v1[9] = 1.64878736635094846896;

data->test_vec_ops_neg_v2 = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_neg_v2[0] = -0.25438811651761727983;
data->test_vec_ops_neg_v2[1] = -1.22464696753573232257;
data->test_vec_ops_neg_v2[2] = 0.29752684437047322019;
data->test_vec_ops_neg_v2[3] = 0.81081458323756994133;
data->test_vec_ops_neg_v2[4] = -0.75224382717959281663;
data->test_vec_ops_neg_v2[5] = -0.25344651620814145909;
data->test_vec_ops_neg_v2[6] = -0.89588307077756035302;
data->test_vec_ops_neg_v2[7] = 0.34521571005127971166;
data->test_vec_ops_neg_v2[8] = 1.48181827372221119887;
data->test_vec_ops_neg_v2[9] = 0.11001076471125098566;

data->test_vec_ops_neg_v3 = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_neg_v3[0] = 0.44582815301123218665;
data->test_vec_ops_neg_v3[1] = -0.77532382204757410715;
data->test_vec_ops_neg_v3[2] = -0.19363284837715380449;
data->test_vec_ops_neg_v3[3] = 1.63084923243510115931;
data->test_vec_ops_neg_v3[4] = 1.19516308010319982635;
data->test_vec_ops_neg_v3[5] = -0.88378903658725527226;
data->test_vec_ops_neg_v3[6] = -0.67976501741784656208;
data->test_vec_ops_neg_v3[7] = 0.64024336590848873740;
data->test_vec_ops_neg_v3[8] = 0.00104879656728068104;
data->test_vec_ops_neg_v3[9] = -0.44557355377618612646;

data->test_vec_ops_shift_v1 = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_shift_v1[0] = 3.68610052803315202397;
data->test_vec_ops_shift_v1[1] = 4.05410227877154394349;
data->test_vec_ops_shift_v1[2] = 4.27279133916445363894;
data->test_vec_ops_shift_v1[3] = 3.01781187505902215307;
data->test_vec_ops_shift_v1[4] = 2.89262695283480697483;
data->test_vec_ops_shift_v1[5] = 4.19958453284708088660;
data->test_vec_ops_shift_v1[6] = 3.53325038312019801268;
data->test_vec_ops_shift_v1[7] = 4.23550561173022543926;
data->test_vec_ops_shift_v1[8] = 4.75951952247837883903;
data->test_vec_ops_shift_v1[9] = 2.35121263364905175308;

data->test_vec_ops_shift_v2 = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_shift_v2[0] = -4.25438811651761739085;
data->test_vec_ops_shift_v2[1] = -5.22464696753573232257;
data->test_vec_ops_shift_v2[2] = -3.70247315562952694634;
data->test_vec_ops_shift_v2[3] = -3.18918541676243005867;
data->test_vec_ops_shift_v2[4] = -4.75224382717959237254;
data->test_vec_ops_shift_v2[5] = -4.25344651620814140358;
data->test_vec_ops_shift_v2[6] = -4.89588307077756024199;
data->test_vec_ops_shift_v2[7] = -3.65478428994872039937;
data->test_vec_ops_shift_v2[8] = -2.51818172627778880113;
data->test_vec_ops_shift_v2[9] = -3.88998923528874884781;

data->test_vec_ops_sc1 = 0.46840433584727791949;
data->test_vec_ops_sc2 = 0.87624219611435005817;
data->test_vec_ops_sc3 = 0.25648562722156198479;
data->test_vec_ops_same = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_same[0] = 0.46840433584727791949;
data->test_vec_ops_same[1] = 0.46840433584727791949;
data->test_vec_ops_same[2] = 0.46840433584727791949;
data->test_vec_ops_same[3] = 0.46840433584727791949;
data->test_vec_ops_same[4] = 0.46840433584727791949;
data->test_vec_ops_same[5] = 0.46840433584727791949;
data->test_vec_ops_same[6] = 0.46840433584727791949;
data->test_vec_ops_same[7] = 0.46840433584727791949;
data->test_vec_ops_same[8] = 0.46840433584727791949;
data->test_vec_ops_same[9] = 0.46840433584727791949;

data->test_vec_ops_mean = -0.29974943423120858910;
data->test_vec_ops_norm_2 = 2.44445605327349113622;
data->test_vec_ops_norm_inf = 1.64878736635094846896;
data->test_vec_ops_norm_inf_scaled = 1.12546990765723009531;
data->test_vec_ops_norm_inf_diff = 2.24133779620059048199;
data->test_vec_ops_sub = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_sub[0] = -0.56828758848446503382;
data->test_vec_ops_sub[1] = -1.17054468876418837908;
data->test_vec_ops_sub[2] = 0.57031818353492691465;
data->test_vec_ops_sub[3] = -0.17137354170340779458;
data->test_vec_ops_sub[4] = -1.85961687434478584180;
data->test_vec_ops_sub[5] = -0.05386198336106062801;
data->test_vec_ops_sub[6] = -1.36263268765736245136;
data->test_vec_ops_sub[7] = 0.58072132178150492887;
data->test_vec_ops_sub[8] = 2.24133779620059048199;
data->test_vec_ops_sub[9] = -1.53877660163969753881;

data->test_vec_ops_add = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_add[0] = -0.05951135544923047416;
data->test_vec_ops_add[1] = 1.27874924630727626607;
data->test_vec_ops_add[2] = -0.02473550520601947023;
data->test_vec_ops_add[3] = -1.79300270817854778826;
data->test_vec_ops_add[4] = -0.35512921998560020853;
data->test_vec_ops_add[5] = 0.45303104905522229018;
data->test_vec_ops_add[6] = 0.42913345389775831018;
data->test_vec_ops_add[7] = -0.10971009832105449444;
data->test_vec_ops_add[8] = -0.72229875124383202678;
data->test_vec_ops_add[9] = -1.75879813106219939911;

data->test_vec_ops_add_scaled = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_add_scaled[0] = 0.07587372819334756158;
data->test_vec_ops_add_scaled[1] = 1.09842909025409851687;
data->test_vec_ops_add_scaled[2] = -0.13292892946794040987;
data->test_vec_ops_add_scaled[3] = -1.17053112739769171746;
data->test_vec_ops_add_scaled[4] = 0.14044944644872148221;
data->test_vec_ops_add_scaled[5] = 0.31556679251337921288;
data->test_vec_ops_add_scaled[6] = 0.56638300509824213158;
data->test_vec_ops_add_scaled[7] = -0.19218072225770499450;
data->test_vec_ops_add_scaled[8] = -0.94266946091919856521;
data->test_vec_ops_add_scaled[9] = -0.86869522535580412370;

data->test_vec_ops_add_scaled_inc = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_add_scaled_inc[0] = -0.09099387008405762645;
data->test_vec_ops_add_scaled_inc[1] = 1.12718962706983316657;
data->test_vec_ops_add_scaled_inc[2] = 0.01208576365029784272;
data->test_vec_ops_add_scaled_inc[3] = -1.69265807599860762522;
data->test_vec_ops_add_scaled_inc[4] = -0.44822526402388296329;
data->test_vec_ops_add_scaled_inc[5] = 0.42166506480683391134;
data->test_vec_ops_add_scaled_inc[6] = 0.31826093251999515443;
data->test_vec_ops_add_scaled_inc[7] = -0.06698696017828281724;
data->test_vec_ops_add_scaled_inc[8] = -0.53891217593034623778;
data->test_vec_ops_add_scaled_inc[9] = -1.74518344041775397280;

data->test_vec_ops_add_scaled3 = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_add_scaled3[0] = -0.03847478526476882932;
data->test_vec_ops_add_scaled3[1] = 1.29728850705178921920;
data->test_vec_ops_add_scaled3[2] = -0.08326488690122850478;
data->test_vec_ops_add_scaled3[3] = -1.58882051568261162267;
data->test_vec_ops_add_scaled3[4] = -0.16609270578360163517;
data->test_vec_ops_add_scaled3[5] = 0.54224597789400141856;
data->test_vec_ops_add_scaled3[6] = 0.74073296195393445451;
data->test_vec_ops_add_scaled3[7] = -0.35639394353718778508;
data->test_vec_ops_add_scaled3[8] = -0.94293846216458532261;
data->test_vec_ops_add_scaled3[9] = -0.75441201294217863360;

data->test_vec_ops_add_scaled3_inc = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_add_scaled3_inc[0] = -0.20534238354217401734;
data->test_vec_ops_add_scaled3_inc[1] = 1.32604904386752386891;
data->test_vec_ops_add_scaled3_inc[2] = 0.06174980621700974781;
data->test_vec_ops_add_scaled3_inc[3] = -2.11094746428352753043;
data->test_vec_ops_add_scaled3_inc[4] = -0.75476741625620613618;
data->test_vec_ops_add_scaled3_inc[5] = 0.64834425018745611702;
data->test_vec_ops_add_scaled3_inc[6] = 0.49261088937568753288;
data->test_vec_ops_add_scaled3_inc[7] = -0.23120018145776558005;
data->test_vec_ops_add_scaled3_inc[8] = -0.53918117717573299519;
data->test_vec_ops_add_scaled3_inc[9] = -1.63090022800412848270;

data->test_vec_ops_ew_sqrt = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_ew_sqrt[0] = 1.91992201092470216039;
data->test_vec_ops_ew_sqrt[1] = 2.01348014114158679888;
data->test_vec_ops_ew_sqrt[2] = 2.06707313348232890604;
data->test_vec_ops_ew_sqrt[3] = 1.73718504341334400998;
data->test_vec_ops_ew_sqrt[4] = 1.70077245768938967174;
data->test_vec_ops_ew_sqrt[5] = 2.04928878707884409849;
data->test_vec_ops_ew_sqrt[6] = 1.87969422596341395604;
data->test_vec_ops_ew_sqrt[7] = 2.05803440489468636443;
data->test_vec_ops_ew_sqrt[8] = 2.18163230689279519225;
data->test_vec_ops_ew_sqrt[9] = 1.53336643815138051750;

data->test_vec_ops_ew_reciprocal = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_ew_reciprocal[0] = -3.18573329777889613368;
data->test_vec_ops_ew_reciprocal[1] = 18.48350980228893547519;
data->test_vec_ops_ew_reciprocal[2] = 3.66580553130077380075;
data->test_vec_ops_ew_reciprocal[3] = -1.01813489148027791487;
data->test_vec_ops_ew_reciprocal[4] = -0.90303805258755265317;
data->test_vec_ops_ew_reciprocal[5] = 5.01040830035756101779;
data->test_vec_ops_ew_reciprocal[6] = -2.14247631671333804704;
data->test_vec_ops_ew_reciprocal[7] = 4.24618331874619236999;
data->test_vec_ops_ew_reciprocal[8] = 1.31662185158442257560;
data->test_vec_ops_ew_reciprocal[9] = -0.60650634545628090422;

data->test_vec_ops_ew_prod = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_ew_prod[0] = -0.07985229544952100744;
data->test_vec_ops_ew_prod[1] = 0.06625619163434404157;
data->test_vec_ops_ew_prod[2] = -0.08116274631319540800;
data->test_vec_ops_ew_prod[3] = 0.79637245518490917817;
data->test_vec_ops_ew_prod[4] = -0.83301453911507250538;
data->test_vec_ops_ew_prod[5] = 0.05058400453912201278;
data->test_vec_ops_ew_prod[6] = -0.41815308005452689333;
data->test_vec_ops_ew_prod[7] = -0.08130023697451069231;
data->test_vec_ops_ew_prod[8] = -1.12546990765723009531;
data->test_vec_ops_ew_prod[9] = 0.18138435901851737708;

data->test_vec_ops_sca_prod = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_sca_prod[0] = -0.14703187368944256597;
data->test_vec_ops_sca_prod[1] = 0.02534174195580929725;
data->test_vec_ops_sca_prod[2] = 0.12777664604621549738;
data->test_vec_ops_sca_prod[3] = -0.46006117634006188366;
data->test_vec_ops_sca_prod[4] = -0.51869833669258857967;
data->test_vec_ops_sca_prod[5] = 0.09348626055362611875;
data->test_vec_ops_sca_prod[6] = -0.21862754430155509344;
data->test_vec_ops_sca_prod[7] = 0.11031184965080305382;
data->test_vec_ops_sca_prod[8] = 0.35576223748952684467;
data->test_vec_ops_sca_prod[9] = -0.77229915128899850885;

data->test_vec_ops_vec_dot = -1.52435579518716401992;
data->test_vec_ops_vec_dot_v1 = 5.97536539638541341901;
data->test_vec_ops_vec_dot_pos = -1.21417971844565419914;
data->test_vec_ops_vec_dot_neg = -0.31017607674150970976;
data->test_vec_ops_vec_dot_pos_flip = -1.17109269477147015515;
data->test_vec_ops_vec_dot_neg_flip = -0.35326310041569386478;
data->test_vec_ops_vec_dot_pos_v1 = 0.74950895522541016724;
data->test_vec_ops_vec_dot_neg_v1 = 5.22585644116000302972;
data->test_vec_ops_ew_bound_vec = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_ew_bound_vec[0] = -0.31389947196684775399;
data->test_vec_ops_ew_bound_vec[1] = 0.77532382204757410715;
data->test_vec_ops_ew_bound_vec[2] = -0.29752684437047322019;
data->test_vec_ops_ew_bound_vec[3] = -0.98218812494097773591;
data->test_vec_ops_ew_bound_vec[4] = -1.10737304716519302517;
data->test_vec_ops_ew_bound_vec[5] = 0.25344651620814145909;
data->test_vec_ops_ew_bound_vec[6] = 0.67976501741784656208;
data->test_vec_ops_ew_bound_vec[7] = -0.34521571005127971166;
data->test_vec_ops_ew_bound_vec[8] = -1.48181827372221119887;
data->test_vec_ops_ew_bound_vec[9] = -0.11001076471125098566;

data->test_vec_ops_ew_max_vec = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_ew_max_vec[0] = 0.25438811651761727983;
data->test_vec_ops_ew_max_vec[1] = 1.22464696753573232257;
data->test_vec_ops_ew_max_vec[2] = 0.27279133916445374997;
data->test_vec_ops_ew_max_vec[3] = -0.81081458323756994133;
data->test_vec_ops_ew_max_vec[4] = 0.75224382717959281663;
data->test_vec_ops_ew_max_vec[5] = 0.25344651620814145909;
data->test_vec_ops_ew_max_vec[6] = 0.89588307077756035302;
data->test_vec_ops_ew_max_vec[7] = 0.23550561173022521722;
data->test_vec_ops_ew_max_vec[8] = 0.75951952247837917209;
data->test_vec_ops_ew_max_vec[9] = -0.11001076471125098566;

data->test_vec_ops_ew_min_vec = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_ew_min_vec[0] = -0.31389947196684775399;
data->test_vec_ops_ew_min_vec[1] = 0.05410227877154388798;
data->test_vec_ops_ew_min_vec[2] = -0.29752684437047322019;
data->test_vec_ops_ew_min_vec[3] = -0.98218812494097773591;
data->test_vec_ops_ew_min_vec[4] = -1.10737304716519302517;
data->test_vec_ops_ew_min_vec[5] = 0.19958453284708083109;
data->test_vec_ops_ew_min_vec[6] = -0.46674961687980204283;
data->test_vec_ops_ew_min_vec[7] = -0.34521571005127971166;
data->test_vec_ops_ew_min_vec[8] = -1.48181827372221119887;
data->test_vec_ops_ew_min_vec[9] = -1.64878736635094846896;

data->test_vec_subvec_ind0 = (OSQPInt*) c_malloc(10 * sizeof(OSQPInt));
data->test_vec_subvec_ind0[0] = 0;
data->test_vec_subvec_ind0[1] = 0;
data->test_vec_subvec_ind0[2] = 0;
data->test_vec_subvec_ind0[3] = 0;
data->test_vec_subvec_ind0[4] = 0;
data->test_vec_subvec_ind0[5] = 0;
data->test_vec_subvec_ind0[6] = 0;
data->test_vec_subvec_ind0[7] = 0;
data->test_vec_subvec_ind0[8] = 0;
data->test_vec_subvec_ind0[9] = 0;

data->test_vec_subvec_ind5 = (OSQPInt*) c_malloc(10 * sizeof(OSQPInt));
data->test_vec_subvec_ind5[0] = 1;
data->test_vec_subvec_ind5[1] = 0;
data->test_vec_subvec_ind5[2] = 1;
data->test_vec_subvec_ind5[3] = 0;
data->test_vec_subvec_ind5[4] = 1;
data->test_vec_subvec_ind5[5] = 0;
data->test_vec_subvec_ind5[6] = 1;
data->test_vec_subvec_ind5[7] = 0;
data->test_vec_subvec_ind5[8] = 1;
data->test_vec_subvec_ind5[9] = 0;

data->test_vec_subvec_ind10 = (OSQPInt*) c_malloc(10 * sizeof(OSQPInt));
data->test_vec_subvec_ind10[0] = 1;
data->test_vec_subvec_ind10[1] = 1;
data->test_vec_subvec_ind10[2] = 1;
data->test_vec_subvec_ind10[3] = 1;
data->test_vec_subvec_ind10[4] = 1;
data->test_vec_subvec_ind10[5] = 1;
data->test_vec_subvec_ind10[6] = 1;
data->test_vec_subvec_ind10[7] = 1;
data->test_vec_subvec_ind10[8] = 1;
data->test_vec_subvec_ind10[9] = 1;

data->test_vec_subvec_0 = (OSQPFloat*) c_malloc(0 * sizeof(OSQPFloat));

data->test_vec_subvec_5 = (OSQPFloat*) c_malloc(5 * sizeof(OSQPFloat));
data->test_vec_subvec_5[0] = -0.31389947196684775399;
data->test_vec_subvec_5[1] = 0.27279133916445374997;
data->test_vec_subvec_5[2] = -1.10737304716519302517;
data->test_vec_subvec_5[3] = -0.46674961687980204283;
data->test_vec_subvec_5[4] = 0.75951952247837917209;

data->test_vec_subvec_assign_5 = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_subvec_assign_5[0] = -0.31389947196684775399;
data->test_vec_subvec_assign_5[1] = 0.05410227877154388798;
data->test_vec_subvec_assign_5[2] = -0.31389947196684775399;
data->test_vec_subvec_assign_5[3] = 0.27279133916445374997;
data->test_vec_subvec_assign_5[4] = -1.10737304716519302517;
data->test_vec_subvec_assign_5[5] = -0.46674961687980204283;
data->test_vec_subvec_assign_5[6] = 0.75951952247837917209;
data->test_vec_subvec_assign_5[7] = 0.23550561173022521722;
data->test_vec_subvec_assign_5[8] = 0.75951952247837917209;
data->test_vec_subvec_assign_5[9] = -1.64878736635094846896;

data->test_vec_ops_sca_lt = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_sca_lt[0] = 0.87624219611435005817;
data->test_vec_ops_sca_lt[1] = 0.87624219611435005817;
data->test_vec_ops_sca_lt[2] = 0.87624219611435005817;
data->test_vec_ops_sca_lt[3] = 0.87624219611435005817;
data->test_vec_ops_sca_lt[4] = 0.87624219611435005817;
data->test_vec_ops_sca_lt[5] = 0.87624219611435005817;
data->test_vec_ops_sca_lt[6] = 0.87624219611435005817;
data->test_vec_ops_sca_lt[7] = 0.87624219611435005817;
data->test_vec_ops_sca_lt[8] = 0.75951952247837917209;
data->test_vec_ops_sca_lt[9] = 0.87624219611435005817;

data->test_vec_ops_sca_gt = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_sca_gt[0] = -0.31389947196684775399;
data->test_vec_ops_sca_gt[1] = 0.05410227877154388798;
data->test_vec_ops_sca_gt[2] = 0.27279133916445374997;
data->test_vec_ops_sca_gt[3] = -0.98218812494097773591;
data->test_vec_ops_sca_gt[4] = -1.10737304716519302517;
data->test_vec_ops_sca_gt[5] = 0.19958453284708083109;
data->test_vec_ops_sca_gt[6] = -0.46674961687980204283;
data->test_vec_ops_sca_gt[7] = 0.23550561173022521722;
data->test_vec_ops_sca_gt[8] = 0.87624219611435005817;
data->test_vec_ops_sca_gt[9] = -1.64878736635094846896;

data->test_vec_ops_sca_cond = (OSQPInt*) c_malloc(10 * sizeof(OSQPInt));
data->test_vec_ops_sca_cond[0] = -1;
data->test_vec_ops_sca_cond[1] = 1;
data->test_vec_ops_sca_cond[2] = 0;
data->test_vec_ops_sca_cond[3] = -1;
data->test_vec_ops_sca_cond[4] = -1;
data->test_vec_ops_sca_cond[5] = 1;
data->test_vec_ops_sca_cond[6] = 0;
data->test_vec_ops_sca_cond[7] = 1;
data->test_vec_ops_sca_cond[8] = 1;
data->test_vec_ops_sca_cond[9] = -1;

data->test_vec_ops_sca_cond_res = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_sca_cond_res[0] = 0.46840433584727791949;
data->test_vec_ops_sca_cond_res[1] = 0.25648562722156198479;
data->test_vec_ops_sca_cond_res[2] = 0.87624219611435005817;
data->test_vec_ops_sca_cond_res[3] = 0.46840433584727791949;
data->test_vec_ops_sca_cond_res[4] = 0.46840433584727791949;
data->test_vec_ops_sca_cond_res[5] = 0.25648562722156198479;
data->test_vec_ops_sca_cond_res[6] = 0.87624219611435005817;
data->test_vec_ops_sca_cond_res[7] = 0.25648562722156198479;
data->test_vec_ops_sca_cond_res[8] = 0.25648562722156198479;
data->test_vec_ops_sca_cond_res[9] = 0.46840433584727791949;

data->test_mat_ops_n = 2;

// Matrix test_mat_ops_A
//----------------------
data->test_mat_ops_A = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_mat_ops_A->m = 2;
data->test_mat_ops_A->n = 2;
data->test_mat_ops_A->nz = -1;
data->test_mat_ops_A->nzmax = 3;
data->test_mat_ops_A->x = (OSQPFloat*) c_malloc(3 * sizeof(OSQPFloat));
data->test_mat_ops_A->x[0] = 0.63315994603655778583;
data->test_mat_ops_A->x[1] = 0.38042426988653232911;
data->test_mat_ops_A->x[2] = 0.10592123670732445095;
data->test_mat_ops_A->i = (OSQPInt*) c_malloc(3 * sizeof(OSQPInt));
data->test_mat_ops_A->i[0] = 0;
data->test_mat_ops_A->i[1] = 1;
data->test_mat_ops_A->i[2] = 0;
data->test_mat_ops_A->p = (OSQPInt*) c_malloc((2 + 1) * sizeof(OSQPInt));
data->test_mat_ops_A->p[0] = 0;
data->test_mat_ops_A->p[1] = 2;
data->test_mat_ops_A->p[2] = 3;

data->test_mat_ops_d = (OSQPFloat*) c_malloc(2 * sizeof(OSQPFloat));
data->test_mat_ops_d[0] = -1.42534896087018769784;
data->test_mat_ops_d[1] = 0.33281361313804663782;


// Matrix test_mat_ops_prem_diag
//------------------------------
data->test_mat_ops_prem_diag = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_mat_ops_prem_diag->m = 2;
data->test_mat_ops_prem_diag->n = 2;
data->test_mat_ops_prem_diag->nz = -1;
data->test_mat_ops_prem_diag->nzmax = 3;
data->test_mat_ops_prem_diag->x = (OSQPFloat*) c_malloc(3 * sizeof(OSQPFloat));
data->test_mat_ops_prem_diag->x[0] = -0.90247387114783172990;
data->test_mat_ops_prem_diag->x[1] = 0.12661037578634021239;
data->test_mat_ops_prem_diag->x[2] = -0.15097472467487008108;
data->test_mat_ops_prem_diag->i = (OSQPInt*) c_malloc(3 * sizeof(OSQPInt));
data->test_mat_ops_prem_diag->i[0] = 0;
data->test_mat_ops_prem_diag->i[1] = 1;
data->test_mat_ops_prem_diag->i[2] = 0;
data->test_mat_ops_prem_diag->p = (OSQPInt*) c_malloc((2 + 1) * sizeof(OSQPInt));
data->test_mat_ops_prem_diag->p[0] = 0;
data->test_mat_ops_prem_diag->p[1] = 2;
data->test_mat_ops_prem_diag->p[2] = 3;


// Matrix test_mat_ops_postm_diag
//-------------------------------
data->test_mat_ops_postm_diag = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_mat_ops_postm_diag->m = 2;
data->test_mat_ops_postm_diag->n = 2;
data->test_mat_ops_postm_diag->nz = -1;
data->test_mat_ops_postm_diag->nzmax = 3;
data->test_mat_ops_postm_diag->x = (OSQPFloat*) c_malloc(3 * sizeof(OSQPFloat));
data->test_mat_ops_postm_diag->x[0] = -0.90247387114783172990;
data->test_mat_ops_postm_diag->x[1] = -0.54223733777256866162;
data->test_mat_ops_postm_diag->x[2] = 0.03525202949661494778;
data->test_mat_ops_postm_diag->i = (OSQPInt*) c_malloc(3 * sizeof(OSQPInt));
data->test_mat_ops_postm_diag->i[0] = 0;
data->test_mat_ops_postm_diag->i[1] = 1;
data->test_mat_ops_postm_diag->i[2] = 0;
data->test_mat_ops_postm_diag->p = (OSQPInt*) c_malloc((2 + 1) * sizeof(OSQPInt));
data->test_mat_ops_postm_diag->p[0] = 0;
data->test_mat_ops_postm_diag->p[1] = 2;
data->test_mat_ops_postm_diag->p[2] = 3;


// Matrix test_mat_ops_scaled
//---------------------------
data->test_mat_ops_scaled = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_mat_ops_scaled->m = 2;
data->test_mat_ops_scaled->n = 2;
data->test_mat_ops_scaled->nz = -1;
data->test_mat_ops_scaled->nzmax = 3;
data->test_mat_ops_scaled->x = (OSQPFloat*) c_malloc(3 * sizeof(OSQPFloat));
data->test_mat_ops_scaled->x[0] = 1.26631989207311557166;
data->test_mat_ops_scaled->x[1] = 0.76084853977306465822;
data->test_mat_ops_scaled->x[2] = 0.21184247341464890191;
data->test_mat_ops_scaled->i = (OSQPInt*) c_malloc(3 * sizeof(OSQPInt));
data->test_mat_ops_scaled->i[0] = 0;
data->test_mat_ops_scaled->i[1] = 1;
data->test_mat_ops_scaled->i[2] = 0;
data->test_mat_ops_scaled->p = (OSQPInt*) c_malloc((2 + 1) * sizeof(OSQPInt));
data->test_mat_ops_scaled->p[0] = 0;
data->test_mat_ops_scaled->p[1] = 2;
data->test_mat_ops_scaled->p[2] = 3;

data->test_mat_ops_inf_norm_cols = (OSQPFloat*) c_malloc(2 * sizeof(OSQPFloat));
data->test_mat_ops_inf_norm_cols[0] = 0.63315994603655778583;
data->test_mat_ops_inf_norm_cols[1] = 0.10592123670732445095;

data->test_mat_ops_inf_norm_rows = (OSQPFloat*) c_malloc(2 * sizeof(OSQPFloat));
data->test_mat_ops_inf_norm_rows[0] = 0.63315994603655778583;
data->test_mat_ops_inf_norm_rows[1] = 0.38042426988653232911;

data->test_mat_vec_n = 4;
data->test_mat_vec_m = 5;

// Matrix test_mat_vec_A
//----------------------
data->test_mat_vec_A = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_mat_vec_A->m = 5;
data->test_mat_vec_A->n = 4;
data->test_mat_vec_A->nz = -1;
data->test_mat_vec_A->nzmax = 20;
data->test_mat_vec_A->x = (OSQPFloat*) c_malloc(20 * sizeof(OSQPFloat));
data->test_mat_vec_A->x[0] = 0.36769956969000661129;
data->test_mat_vec_A->x[1] = 0.59698773052375642134;
data->test_mat_vec_A->x[2] = 0.50035643073687097182;
data->test_mat_vec_A->x[3] = 0.79113394817280524585;
data->test_mat_vec_A->x[4] = 0.68963015544708095028;
data->test_mat_vec_A->x[5] = 0.91769225717091273964;
data->test_mat_vec_A->x[6] = 0.44839630934448426736;
data->test_mat_vec_A->x[7] = 0.31304785881993768548;
data->test_mat_vec_A->x[8] = 0.20324154408739658617;
data->test_mat_vec_A->x[9] = 0.97168997561975467558;
data->test_mat_vec_A->x[10] = 0.31413389560230220443;
data->test_mat_vec_A->x[11] = 0.10973466400669673604;
data->test_mat_vec_A->x[12] = 0.77466413492373198402;
data->test_mat_vec_A->x[13] = 0.36500726350855894342;
data->test_mat_vec_A->x[14] = 0.75926850053957994913;
data->test_mat_vec_A->x[15] = 0.72647361031237045470;
data->test_mat_vec_A->x[16] = 0.28380648894413107453;
data->test_mat_vec_A->x[17] = 0.58651832682553139975;
data->test_mat_vec_A->x[18] = 0.83968460360894237038;
data->test_mat_vec_A->x[19] = 0.57669971625295202156;
data->test_mat_vec_A->i = (OSQPInt*) c_malloc(20 * sizeof(OSQPInt));
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
data->test_mat_vec_A->p = (OSQPInt*) c_malloc((4 + 1) * sizeof(OSQPInt));
data->test_mat_vec_A->p[0] = 0;
data->test_mat_vec_A->p[1] = 5;
data->test_mat_vec_A->p[2] = 10;
data->test_mat_vec_A->p[3] = 15;
data->test_mat_vec_A->p[4] = 20;


// Matrix test_mat_vec_Pu
//-----------------------
data->test_mat_vec_Pu = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_mat_vec_Pu->m = 4;
data->test_mat_vec_Pu->n = 4;
data->test_mat_vec_Pu->nz = -1;
data->test_mat_vec_Pu->nzmax = 10;
data->test_mat_vec_Pu->x = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_mat_vec_Pu->x[0] = 0.59046851132539157625;
data->test_mat_vec_Pu->x[1] = 1.12608223708628596427;
data->test_mat_vec_Pu->x[2] = 0.37564948513093665561;
data->test_mat_vec_Pu->x[3] = 0.12446033251547983234;
data->test_mat_vec_Pu->x[4] = 0.92552271316505363430;
data->test_mat_vec_Pu->x[5] = 1.94938562576457852238;
data->test_mat_vec_Pu->x[6] = 1.46648947710749366635;
data->test_mat_vec_Pu->x[7] = 0.39007455193986173558;
data->test_mat_vec_Pu->x[8] = 0.84699837063372962476;
data->test_mat_vec_Pu->x[9] = 1.04305024426483505806;
data->test_mat_vec_Pu->i = (OSQPInt*) c_malloc(10 * sizeof(OSQPInt));
data->test_mat_vec_Pu->i[0] = 0;
data->test_mat_vec_Pu->i[1] = 0;
data->test_mat_vec_Pu->i[2] = 1;
data->test_mat_vec_Pu->i[3] = 0;
data->test_mat_vec_Pu->i[4] = 1;
data->test_mat_vec_Pu->i[5] = 2;
data->test_mat_vec_Pu->i[6] = 0;
data->test_mat_vec_Pu->i[7] = 1;
data->test_mat_vec_Pu->i[8] = 2;
data->test_mat_vec_Pu->i[9] = 3;
data->test_mat_vec_Pu->p = (OSQPInt*) c_malloc((4 + 1) * sizeof(OSQPInt));
data->test_mat_vec_Pu->p[0] = 0;
data->test_mat_vec_Pu->p[1] = 1;
data->test_mat_vec_Pu->p[2] = 3;
data->test_mat_vec_Pu->p[3] = 6;
data->test_mat_vec_Pu->p[4] = 10;

data->test_mat_vec_x = (OSQPFloat*) c_malloc(4 * sizeof(OSQPFloat));
data->test_mat_vec_x[0] = -0.62935492388334191016;
data->test_mat_vec_x[1] = 0.23151106405853766335;
data->test_mat_vec_x[2] = 0.70015175115065042544;
data->test_mat_vec_x[3] = 0.66365757101740663337;

data->test_mat_vec_y = (OSQPFloat*) c_malloc(5 * sizeof(OSQPFloat));
data->test_mat_vec_y[0] = 1.97247385392979168728;
data->test_mat_vec_y[1] = 0.20916747233627677738;
data->test_mat_vec_y[2] = -0.59241009975293257295;
data->test_mat_vec_y[3] = -0.12597918998664273116;
data->test_mat_vec_y[4] = -0.07249854561981204648;

data->test_mat_vec_Ax = (OSQPFloat*) c_malloc(5 * sizeof(OSQPFloat));
data->test_mat_vec_Ax[0] = 0.68311348497160873094;
data->test_mat_vec_Ax[1] = -0.00672721874894752214;
data->test_mat_vec_Ax[2] = 0.68920203827438686339;
data->test_mat_vec_Ax[3] = 0.36197213957879881274;
data->test_mat_vec_Ax[4] = 0.70526914934286621950;

data->test_mat_vec_Ax_cum = (OSQPFloat*) c_malloc(5 * sizeof(OSQPFloat));
data->test_mat_vec_Ax_cum[0] = 2.65558733890140041822;
data->test_mat_vec_Ax_cum[1] = 0.20244025358732925524;
data->test_mat_vec_Ax_cum[2] = 0.09679193852145429044;
data->test_mat_vec_Ax_cum[3] = 0.23599294959215608158;
data->test_mat_vec_Ax_cum[4] = 0.63277060372305415914;

data->test_mat_vec_ATy = (OSQPFloat*) c_malloc(4 * sizeof(OSQPFloat));
data->test_mat_vec_ATy[0] = 0.40406840163212393024;
data->test_mat_vec_ATy[1] = 1.62241087746143164416;
data->test_mat_vec_ATy[2] = 0.08262577914131154222;
data->test_mat_vec_ATy[3] = 0.99726123043996506290;

data->test_mat_vec_ATy_cum = (OSQPFloat*) c_malloc(4 * sizeof(OSQPFloat));
data->test_mat_vec_ATy_cum[0] = -0.22528652225121797992;
data->test_mat_vec_ATy_cum[1] = 1.85392194151996925200;
data->test_mat_vec_ATy_cum[2] = 0.78277753029196195378;
data->test_mat_vec_ATy_cum[3] = 1.66091880145737169627;

data->test_mat_vec_Px = (OSQPFloat*) c_malloc(4 * sizeof(OSQPFloat));
data->test_mat_vec_Px[0] = 0.94947419598381388450;
data->test_mat_vec_Px[1] = 0.28514388941620749662;
data->test_mat_vec_Px[2] = 2.06292166589586933867;
data->test_mat_vec_Px[3] = 0.45261978531991275965;

data->test_mat_vec_Px_cum = (OSQPFloat*) c_malloc(4 * sizeof(OSQPFloat));
data->test_mat_vec_Px_cum[0] = 0.32011927210047197434;
data->test_mat_vec_Px_cum[1] = 0.51665495347474510446;
data->test_mat_vec_Px_cum[2] = 2.76307341704651987513;
data->test_mat_vec_Px_cum[3] = 1.11627735633731939302;

data->test_submat_A4_num = 4;
data->test_submat_A5_num = 5;
data->test_submat_A3_num = 3;
data->test_submat_A0_num = 0;
data->test_submat_A4_ind = (OSQPInt*) c_malloc(5 * sizeof(OSQPInt));
data->test_submat_A4_ind[0] = 1;
data->test_submat_A4_ind[1] = 1;
data->test_submat_A4_ind[2] = 0;
data->test_submat_A4_ind[3] = 1;
data->test_submat_A4_ind[4] = 1;

data->test_submat_A5_ind = (OSQPInt*) c_malloc(5 * sizeof(OSQPInt));
data->test_submat_A5_ind[0] = 1;
data->test_submat_A5_ind[1] = 1;
data->test_submat_A5_ind[2] = 1;
data->test_submat_A5_ind[3] = 1;
data->test_submat_A5_ind[4] = 1;

data->test_submat_A3_ind = (OSQPInt*) c_malloc(5 * sizeof(OSQPInt));
data->test_submat_A3_ind[0] = 1;
data->test_submat_A3_ind[1] = 0;
data->test_submat_A3_ind[2] = 1;
data->test_submat_A3_ind[3] = 0;
data->test_submat_A3_ind[4] = 1;

data->test_submat_A0_ind = (OSQPInt*) c_malloc(5 * sizeof(OSQPInt));
data->test_submat_A0_ind[0] = 0;
data->test_submat_A0_ind[1] = 0;
data->test_submat_A0_ind[2] = 0;
data->test_submat_A0_ind[3] = 0;
data->test_submat_A0_ind[4] = 0;


// Matrix test_submat_A4
//----------------------
data->test_submat_A4 = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_submat_A4->m = 4;
data->test_submat_A4->n = 4;
data->test_submat_A4->nz = -1;
data->test_submat_A4->nzmax = 16;
data->test_submat_A4->x = (OSQPFloat*) c_malloc(16 * sizeof(OSQPFloat));
data->test_submat_A4->x[0] = 0.36769956969000661129;
data->test_submat_A4->x[1] = 0.59698773052375642134;
data->test_submat_A4->x[2] = 0.79113394817280524585;
data->test_submat_A4->x[3] = 0.68963015544708095028;
data->test_submat_A4->x[4] = 0.91769225717091273964;
data->test_submat_A4->x[5] = 0.44839630934448426736;
data->test_submat_A4->x[6] = 0.20324154408739658617;
data->test_submat_A4->x[7] = 0.97168997561975467558;
data->test_submat_A4->x[8] = 0.31413389560230220443;
data->test_submat_A4->x[9] = 0.10973466400669673604;
data->test_submat_A4->x[10] = 0.36500726350855894342;
data->test_submat_A4->x[11] = 0.75926850053957994913;
data->test_submat_A4->x[12] = 0.72647361031237045470;
data->test_submat_A4->x[13] = 0.28380648894413107453;
data->test_submat_A4->x[14] = 0.83968460360894237038;
data->test_submat_A4->x[15] = 0.57669971625295202156;
data->test_submat_A4->i = (OSQPInt*) c_malloc(16 * sizeof(OSQPInt));
data->test_submat_A4->i[0] = 0;
data->test_submat_A4->i[1] = 1;
data->test_submat_A4->i[2] = 2;
data->test_submat_A4->i[3] = 3;
data->test_submat_A4->i[4] = 0;
data->test_submat_A4->i[5] = 1;
data->test_submat_A4->i[6] = 2;
data->test_submat_A4->i[7] = 3;
data->test_submat_A4->i[8] = 0;
data->test_submat_A4->i[9] = 1;
data->test_submat_A4->i[10] = 2;
data->test_submat_A4->i[11] = 3;
data->test_submat_A4->i[12] = 0;
data->test_submat_A4->i[13] = 1;
data->test_submat_A4->i[14] = 2;
data->test_submat_A4->i[15] = 3;
data->test_submat_A4->p = (OSQPInt*) c_malloc((4 + 1) * sizeof(OSQPInt));
data->test_submat_A4->p[0] = 0;
data->test_submat_A4->p[1] = 4;
data->test_submat_A4->p[2] = 8;
data->test_submat_A4->p[3] = 12;
data->test_submat_A4->p[4] = 16;


// Matrix test_submat_A5
//----------------------
data->test_submat_A5 = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_submat_A5->m = 5;
data->test_submat_A5->n = 4;
data->test_submat_A5->nz = -1;
data->test_submat_A5->nzmax = 20;
data->test_submat_A5->x = (OSQPFloat*) c_malloc(20 * sizeof(OSQPFloat));
data->test_submat_A5->x[0] = 0.36769956969000661129;
data->test_submat_A5->x[1] = 0.59698773052375642134;
data->test_submat_A5->x[2] = 0.50035643073687097182;
data->test_submat_A5->x[3] = 0.79113394817280524585;
data->test_submat_A5->x[4] = 0.68963015544708095028;
data->test_submat_A5->x[5] = 0.91769225717091273964;
data->test_submat_A5->x[6] = 0.44839630934448426736;
data->test_submat_A5->x[7] = 0.31304785881993768548;
data->test_submat_A5->x[8] = 0.20324154408739658617;
data->test_submat_A5->x[9] = 0.97168997561975467558;
data->test_submat_A5->x[10] = 0.31413389560230220443;
data->test_submat_A5->x[11] = 0.10973466400669673604;
data->test_submat_A5->x[12] = 0.77466413492373198402;
data->test_submat_A5->x[13] = 0.36500726350855894342;
data->test_submat_A5->x[14] = 0.75926850053957994913;
data->test_submat_A5->x[15] = 0.72647361031237045470;
data->test_submat_A5->x[16] = 0.28380648894413107453;
data->test_submat_A5->x[17] = 0.58651832682553139975;
data->test_submat_A5->x[18] = 0.83968460360894237038;
data->test_submat_A5->x[19] = 0.57669971625295202156;
data->test_submat_A5->i = (OSQPInt*) c_malloc(20 * sizeof(OSQPInt));
data->test_submat_A5->i[0] = 0;
data->test_submat_A5->i[1] = 1;
data->test_submat_A5->i[2] = 2;
data->test_submat_A5->i[3] = 3;
data->test_submat_A5->i[4] = 4;
data->test_submat_A5->i[5] = 0;
data->test_submat_A5->i[6] = 1;
data->test_submat_A5->i[7] = 2;
data->test_submat_A5->i[8] = 3;
data->test_submat_A5->i[9] = 4;
data->test_submat_A5->i[10] = 0;
data->test_submat_A5->i[11] = 1;
data->test_submat_A5->i[12] = 2;
data->test_submat_A5->i[13] = 3;
data->test_submat_A5->i[14] = 4;
data->test_submat_A5->i[15] = 0;
data->test_submat_A5->i[16] = 1;
data->test_submat_A5->i[17] = 2;
data->test_submat_A5->i[18] = 3;
data->test_submat_A5->i[19] = 4;
data->test_submat_A5->p = (OSQPInt*) c_malloc((4 + 1) * sizeof(OSQPInt));
data->test_submat_A5->p[0] = 0;
data->test_submat_A5->p[1] = 5;
data->test_submat_A5->p[2] = 10;
data->test_submat_A5->p[3] = 15;
data->test_submat_A5->p[4] = 20;


// Matrix test_submat_A3
//----------------------
data->test_submat_A3 = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_submat_A3->m = 3;
data->test_submat_A3->n = 4;
data->test_submat_A3->nz = -1;
data->test_submat_A3->nzmax = 12;
data->test_submat_A3->x = (OSQPFloat*) c_malloc(12 * sizeof(OSQPFloat));
data->test_submat_A3->x[0] = 0.36769956969000661129;
data->test_submat_A3->x[1] = 0.50035643073687097182;
data->test_submat_A3->x[2] = 0.68963015544708095028;
data->test_submat_A3->x[3] = 0.91769225717091273964;
data->test_submat_A3->x[4] = 0.31304785881993768548;
data->test_submat_A3->x[5] = 0.97168997561975467558;
data->test_submat_A3->x[6] = 0.31413389560230220443;
data->test_submat_A3->x[7] = 0.77466413492373198402;
data->test_submat_A3->x[8] = 0.75926850053957994913;
data->test_submat_A3->x[9] = 0.72647361031237045470;
data->test_submat_A3->x[10] = 0.58651832682553139975;
data->test_submat_A3->x[11] = 0.57669971625295202156;
data->test_submat_A3->i = (OSQPInt*) c_malloc(12 * sizeof(OSQPInt));
data->test_submat_A3->i[0] = 0;
data->test_submat_A3->i[1] = 1;
data->test_submat_A3->i[2] = 2;
data->test_submat_A3->i[3] = 0;
data->test_submat_A3->i[4] = 1;
data->test_submat_A3->i[5] = 2;
data->test_submat_A3->i[6] = 0;
data->test_submat_A3->i[7] = 1;
data->test_submat_A3->i[8] = 2;
data->test_submat_A3->i[9] = 0;
data->test_submat_A3->i[10] = 1;
data->test_submat_A3->i[11] = 2;
data->test_submat_A3->p = (OSQPInt*) c_malloc((4 + 1) * sizeof(OSQPInt));
data->test_submat_A3->p[0] = 0;
data->test_submat_A3->p[1] = 3;
data->test_submat_A3->p[2] = 6;
data->test_submat_A3->p[3] = 9;
data->test_submat_A3->p[4] = 12;


// Matrix test_submat_A0
//----------------------
data->test_submat_A0 = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_submat_A0->m = 0;
data->test_submat_A0->n = 4;
data->test_submat_A0->nz = -1;
data->test_submat_A0->nzmax = 0;
data->test_submat_A0->x = OSQP_NULL;
data->test_submat_A0->i = OSQP_NULL;
data->test_submat_A0->p = (OSQPInt*) c_malloc((4 + 1) * sizeof(OSQPInt));
data->test_submat_A0->p[0] = 0;
data->test_submat_A0->p[1] = 0;
data->test_submat_A0->p[2] = 0;
data->test_submat_A0->p[3] = 0;
data->test_submat_A0->p[4] = 0;

data->test_mat_extr_triu_n = 5;

// Matrix test_mat_extr_triu_P
//----------------------------
data->test_mat_extr_triu_P = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_mat_extr_triu_P->m = 5;
data->test_mat_extr_triu_P->n = 5;
data->test_mat_extr_triu_P->nz = -1;
data->test_mat_extr_triu_P->nzmax = 25;
data->test_mat_extr_triu_P->x = (OSQPFloat*) c_malloc(25 * sizeof(OSQPFloat));
data->test_mat_extr_triu_P->x[0] = 0.44501373189254489482;
data->test_mat_extr_triu_P->x[1] = 0.73236083732560453008;
data->test_mat_extr_triu_P->x[2] = 0.61522231691087359007;
data->test_mat_extr_triu_P->x[3] = 0.44681295173445190194;
data->test_mat_extr_triu_P->x[4] = 1.07889745339591458517;
data->test_mat_extr_triu_P->x[5] = 0.73236083732560453008;
data->test_mat_extr_triu_P->x[6] = 0.77553823131190791074;
data->test_mat_extr_triu_P->x[7] = 0.37185456987010601093;
data->test_mat_extr_triu_P->x[8] = 1.16718817937722385558;
data->test_mat_extr_triu_P->x[9] = 0.41523628869206530290;
data->test_mat_extr_triu_P->x[10] = 0.61522231691087359007;
data->test_mat_extr_triu_P->x[11] = 0.37185456987010601093;
data->test_mat_extr_triu_P->x[12] = 1.54778859509962263274;
data->test_mat_extr_triu_P->x[13] = 0.96286457844322781430;
data->test_mat_extr_triu_P->x[14] = 1.64402715271316246515;
data->test_mat_extr_triu_P->x[15] = 0.44681295173445190194;
data->test_mat_extr_triu_P->x[16] = 1.16718817937722385558;
data->test_mat_extr_triu_P->x[17] = 0.96286457844322781430;
data->test_mat_extr_triu_P->x[18] = 1.58331241118068111184;
data->test_mat_extr_triu_P->x[19] = 0.02648454890397233807;
data->test_mat_extr_triu_P->x[20] = 1.07889745339591458517;
data->test_mat_extr_triu_P->x[21] = 0.41523628869206530290;
data->test_mat_extr_triu_P->x[22] = 1.64402715271316246515;
data->test_mat_extr_triu_P->x[23] = 0.02648454890397233807;
data->test_mat_extr_triu_P->x[24] = 0.50253515634216361363;
data->test_mat_extr_triu_P->i = (OSQPInt*) c_malloc(25 * sizeof(OSQPInt));
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
data->test_mat_extr_triu_P->i[24] = 4;
data->test_mat_extr_triu_P->p = (OSQPInt*) c_malloc((5 + 1) * sizeof(OSQPInt));
data->test_mat_extr_triu_P->p[0] = 0;
data->test_mat_extr_triu_P->p[1] = 5;
data->test_mat_extr_triu_P->p[2] = 10;
data->test_mat_extr_triu_P->p[3] = 15;
data->test_mat_extr_triu_P->p[4] = 20;
data->test_mat_extr_triu_P->p[5] = 25;


// Matrix test_mat_extr_triu_Pu
//-----------------------------
data->test_mat_extr_triu_Pu = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_mat_extr_triu_Pu->m = 5;
data->test_mat_extr_triu_Pu->n = 5;
data->test_mat_extr_triu_Pu->nz = -1;
data->test_mat_extr_triu_Pu->nzmax = 15;
data->test_mat_extr_triu_Pu->x = (OSQPFloat*) c_malloc(15 * sizeof(OSQPFloat));
data->test_mat_extr_triu_Pu->x[0] = 0.44501373189254489482;
data->test_mat_extr_triu_Pu->x[1] = 0.73236083732560453008;
data->test_mat_extr_triu_Pu->x[2] = 0.77553823131190791074;
data->test_mat_extr_triu_Pu->x[3] = 0.61522231691087359007;
data->test_mat_extr_triu_Pu->x[4] = 0.37185456987010601093;
data->test_mat_extr_triu_Pu->x[5] = 1.54778859509962263274;
data->test_mat_extr_triu_Pu->x[6] = 0.44681295173445190194;
data->test_mat_extr_triu_Pu->x[7] = 1.16718817937722385558;
data->test_mat_extr_triu_Pu->x[8] = 0.96286457844322781430;
data->test_mat_extr_triu_Pu->x[9] = 1.58331241118068111184;
data->test_mat_extr_triu_Pu->x[10] = 1.07889745339591458517;
data->test_mat_extr_triu_Pu->x[11] = 0.41523628869206530290;
data->test_mat_extr_triu_Pu->x[12] = 1.64402715271316246515;
data->test_mat_extr_triu_Pu->x[13] = 0.02648454890397233807;
data->test_mat_extr_triu_Pu->x[14] = 0.50253515634216361363;
data->test_mat_extr_triu_Pu->i = (OSQPInt*) c_malloc(15 * sizeof(OSQPInt));
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
data->test_mat_extr_triu_Pu->i[14] = 4;
data->test_mat_extr_triu_Pu->p = (OSQPInt*) c_malloc((5 + 1) * sizeof(OSQPInt));
data->test_mat_extr_triu_Pu->p[0] = 0;
data->test_mat_extr_triu_Pu->p[1] = 1;
data->test_mat_extr_triu_Pu->p[2] = 3;
data->test_mat_extr_triu_Pu->p[3] = 6;
data->test_mat_extr_triu_Pu->p[4] = 10;
data->test_mat_extr_triu_Pu->p[5] = 15;

data->test_mat_extr_triu_P_inf_norm_cols = (OSQPFloat*) c_malloc(5 * sizeof(OSQPFloat));
data->test_mat_extr_triu_P_inf_norm_cols[0] = 1.07889745339591458517;
data->test_mat_extr_triu_P_inf_norm_cols[1] = 1.16718817937722385558;
data->test_mat_extr_triu_P_inf_norm_cols[2] = 1.64402715271316246515;
data->test_mat_extr_triu_P_inf_norm_cols[3] = 1.58331241118068111184;
data->test_mat_extr_triu_P_inf_norm_cols[4] = 1.64402715271316246515;

data->test_qpform_n = 4;

// Matrix test_qpform_Pu
//----------------------
data->test_qpform_Pu = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_qpform_Pu->m = 4;
data->test_qpform_Pu->n = 4;
data->test_qpform_Pu->nz = -1;
data->test_qpform_Pu->nzmax = 10;
data->test_qpform_Pu->x = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_qpform_Pu->x[0] = 1.69004618313669285889;
data->test_qpform_Pu->x[1] = 0.45103888617139120676;
data->test_qpform_Pu->x[2] = 0.09197436134103131877;
data->test_qpform_Pu->x[3] = 0.32593269184534967575;
data->test_qpform_Pu->x[4] = 0.81328509366694534677;
data->test_qpform_Pu->x[5] = 1.87750365695976917380;
data->test_qpform_Pu->x[6] = 1.46813374444913380401;
data->test_qpform_Pu->x[7] = 0.17483553789762362740;
data->test_qpform_Pu->x[8] = 0.95729436725353733717;
data->test_qpform_Pu->x[9] = 1.90830267373328488212;
data->test_qpform_Pu->i = (OSQPInt*) c_malloc(10 * sizeof(OSQPInt));
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
data->test_qpform_Pu->p = (OSQPInt*) c_malloc((4 + 1) * sizeof(OSQPInt));
data->test_qpform_Pu->p[0] = 0;
data->test_qpform_Pu->p[1] = 1;
data->test_qpform_Pu->p[2] = 3;
data->test_qpform_Pu->p[3] = 6;
data->test_qpform_Pu->p[4] = 10;

data->test_qpform_x = (OSQPFloat*) c_malloc(4 * sizeof(OSQPFloat));
data->test_qpform_x[0] = 0.25956345889205961752;
data->test_qpform_x[1] = -1.41238121547177541970;
data->test_qpform_x[2] = 0.77032208279449598809;
data->test_qpform_x[3] = -0.70109980043342623457;

data->test_qpform_value = -0.42135682247368677622;

// Matrix test_mat_no_entries
//---------------------------
data->test_mat_no_entries = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_mat_no_entries->m = 2;
data->test_mat_no_entries->n = 2;
data->test_mat_no_entries->nz = -1;
data->test_mat_no_entries->nzmax = 0;
data->test_mat_no_entries->x = OSQP_NULL;
data->test_mat_no_entries->i = OSQP_NULL;
data->test_mat_no_entries->p = (OSQPInt*) c_malloc((2 + 1) * sizeof(OSQPInt));
data->test_mat_no_entries->p[0] = 0;
data->test_mat_no_entries->p[1] = 0;
data->test_mat_no_entries->p[2] = 0;


// Matrix test_mat_no_rows
//------------------------
data->test_mat_no_rows = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_mat_no_rows->m = 0;
data->test_mat_no_rows->n = 2;
data->test_mat_no_rows->nz = -1;
data->test_mat_no_rows->nzmax = 0;
data->test_mat_no_rows->x = OSQP_NULL;
data->test_mat_no_rows->i = OSQP_NULL;
data->test_mat_no_rows->p = (OSQPInt*) c_malloc((2 + 1) * sizeof(OSQPInt));
data->test_mat_no_rows->p[0] = 0;
data->test_mat_no_rows->p[1] = 0;
data->test_mat_no_rows->p[2] = 0;


// Matrix test_mat_no_cols
//------------------------
data->test_mat_no_cols = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_mat_no_cols->m = 2;
data->test_mat_no_cols->n = 0;
data->test_mat_no_cols->nz = -1;
data->test_mat_no_cols->nzmax = 0;
data->test_mat_no_cols->x = OSQP_NULL;
data->test_mat_no_cols->i = OSQP_NULL;
data->test_mat_no_cols->p = (OSQPInt*) c_malloc((0 + 1) * sizeof(OSQPInt));
data->test_mat_no_cols->p[0] = 0;

data->test_vec_empty = (OSQPFloat*) c_malloc(0 * sizeof(OSQPFloat));

data->test_vec_mat_empty = (OSQPFloat*) c_malloc(2 * sizeof(OSQPFloat));
data->test_vec_mat_empty[0] = 1.00000000000000000000;
data->test_vec_mat_empty[1] = 2.00000000000000000000;

data->test_vec_zeros = (OSQPFloat*) c_malloc(2 * sizeof(OSQPFloat));
data->test_vec_zeros[0] = 0.00000000000000000000;
data->test_vec_zeros[1] = 0.00000000000000000000;


return data;

}

/* function to clean data struct */
void clean_problem_lin_alg_sols_data(lin_alg_sols_data * data){

c_free(data->test_sp_matrix_A->x);
c_free(data->test_sp_matrix_A->i);
c_free(data->test_sp_matrix_A->p);
c_free(data->test_sp_matrix_A);
c_free(data->test_sp_matrix_Adns);
c_free(data->test_vec_ops_vn);
c_free(data->test_vec_ops_vn_neg);
c_free(data->test_vec_ops_ones);
c_free(data->test_vec_ops_zero);
c_free(data->test_vec_ops_zero_int);
c_free(data->test_vec_ops_v1);
c_free(data->test_vec_ops_v2);
c_free(data->test_vec_ops_v3);
c_free(data->test_vec_ops_neg_v1);
c_free(data->test_vec_ops_neg_v2);
c_free(data->test_vec_ops_neg_v3);
c_free(data->test_vec_ops_shift_v1);
c_free(data->test_vec_ops_shift_v2);
c_free(data->test_vec_ops_same);
c_free(data->test_vec_ops_sub);
c_free(data->test_vec_ops_add);
c_free(data->test_vec_ops_add_scaled);
c_free(data->test_vec_ops_add_scaled_inc);
c_free(data->test_vec_ops_add_scaled3);
c_free(data->test_vec_ops_add_scaled3_inc);
c_free(data->test_vec_ops_ew_sqrt);
c_free(data->test_vec_ops_ew_reciprocal);
c_free(data->test_vec_ops_ew_prod);
c_free(data->test_vec_ops_sca_prod);
c_free(data->test_vec_ops_ew_bound_vec);
c_free(data->test_vec_ops_ew_max_vec);
c_free(data->test_vec_ops_ew_min_vec);
c_free(data->test_vec_subvec_ind0);
c_free(data->test_vec_subvec_ind5);
c_free(data->test_vec_subvec_ind10);
c_free(data->test_vec_subvec_0);
c_free(data->test_vec_subvec_5);
c_free(data->test_vec_subvec_assign_5);
c_free(data->test_vec_ops_sca_lt);
c_free(data->test_vec_ops_sca_gt);
c_free(data->test_vec_ops_sca_cond);
c_free(data->test_vec_ops_sca_cond_res);
c_free(data->test_mat_ops_A->x);
c_free(data->test_mat_ops_A->i);
c_free(data->test_mat_ops_A->p);
c_free(data->test_mat_ops_A);
c_free(data->test_mat_ops_d);
c_free(data->test_mat_ops_prem_diag->x);
c_free(data->test_mat_ops_prem_diag->i);
c_free(data->test_mat_ops_prem_diag->p);
c_free(data->test_mat_ops_prem_diag);
c_free(data->test_mat_ops_postm_diag->x);
c_free(data->test_mat_ops_postm_diag->i);
c_free(data->test_mat_ops_postm_diag->p);
c_free(data->test_mat_ops_postm_diag);
c_free(data->test_mat_ops_scaled->x);
c_free(data->test_mat_ops_scaled->i);
c_free(data->test_mat_ops_scaled->p);
c_free(data->test_mat_ops_scaled);
c_free(data->test_mat_ops_inf_norm_cols);
c_free(data->test_mat_ops_inf_norm_rows);
c_free(data->test_mat_vec_A->x);
c_free(data->test_mat_vec_A->i);
c_free(data->test_mat_vec_A->p);
c_free(data->test_mat_vec_A);
c_free(data->test_mat_vec_Pu->x);
c_free(data->test_mat_vec_Pu->i);
c_free(data->test_mat_vec_Pu->p);
c_free(data->test_mat_vec_Pu);
c_free(data->test_mat_vec_x);
c_free(data->test_mat_vec_y);
c_free(data->test_mat_vec_Ax);
c_free(data->test_mat_vec_Ax_cum);
c_free(data->test_mat_vec_ATy);
c_free(data->test_mat_vec_ATy_cum);
c_free(data->test_mat_vec_Px);
c_free(data->test_mat_vec_Px_cum);
c_free(data->test_submat_A4_ind);
c_free(data->test_submat_A5_ind);
c_free(data->test_submat_A3_ind);
c_free(data->test_submat_A0_ind);
c_free(data->test_submat_A4->x);
c_free(data->test_submat_A4->i);
c_free(data->test_submat_A4->p);
c_free(data->test_submat_A4);
c_free(data->test_submat_A5->x);
c_free(data->test_submat_A5->i);
c_free(data->test_submat_A5->p);
c_free(data->test_submat_A5);
c_free(data->test_submat_A3->x);
c_free(data->test_submat_A3->i);
c_free(data->test_submat_A3->p);
c_free(data->test_submat_A3);
c_free(data->test_submat_A0->x);
c_free(data->test_submat_A0->i);
c_free(data->test_submat_A0->p);
c_free(data->test_submat_A0);
c_free(data->test_mat_extr_triu_P->x);
c_free(data->test_mat_extr_triu_P->i);
c_free(data->test_mat_extr_triu_P->p);
c_free(data->test_mat_extr_triu_P);
c_free(data->test_mat_extr_triu_Pu->x);
c_free(data->test_mat_extr_triu_Pu->i);
c_free(data->test_mat_extr_triu_Pu->p);
c_free(data->test_mat_extr_triu_Pu);
c_free(data->test_mat_extr_triu_P_inf_norm_cols);
c_free(data->test_qpform_Pu->x);
c_free(data->test_qpform_Pu->i);
c_free(data->test_qpform_Pu->p);
c_free(data->test_qpform_Pu);
c_free(data->test_qpform_x);
c_free(data->test_mat_no_entries->x);
c_free(data->test_mat_no_entries->i);
c_free(data->test_mat_no_entries->p);
c_free(data->test_mat_no_entries);
c_free(data->test_mat_no_rows->x);
c_free(data->test_mat_no_rows->i);
c_free(data->test_mat_no_rows->p);
c_free(data->test_mat_no_rows);
c_free(data->test_mat_no_cols->x);
c_free(data->test_mat_no_cols->i);
c_free(data->test_mat_no_cols->p);
c_free(data->test_mat_no_cols);
c_free(data->test_vec_empty);
c_free(data->test_vec_mat_empty);
c_free(data->test_vec_zeros);

c_free(data);

}

