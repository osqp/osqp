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

data->test_vec_ops_sc1 = -0.44582815301123218665;
data->test_vec_ops_sc2 = 0.77532382204757410715;
data->test_vec_ops_norm_inf = 1.64878736635094846896;
data->test_vec_ops_norm_inf_diff = 2.24133779620059048199;
data->test_vec_ops_add_scaled = (OSQPFloat*) c_malloc(10 * sizeof(OSQPFloat));
data->test_vec_ops_add_scaled[0] = 0.33717838860010340696;
data->test_vec_ops_add_scaled[1] = 0.92537764851035919644;
data->test_vec_ops_add_scaled[2] = -0.35229770903621804301;
data->test_vec_ops_add_scaled[3] = -0.19075674399566217021;
data->test_vec_ops_add_scaled[4] = 1.07693063951265477485;
data->test_vec_ops_add_scaled[5] = 0.10752271798231502475;
data->test_vec_ops_add_scaled[6] = 0.90268960615519777679;
data->test_vec_ops_add_scaled[7] = -0.37264899564929204745;
data->test_vec_ops_add_scaled[8] = -1.48750419344475193206;
data->test_vec_ops_add_scaled[9] = 0.64978185968619373014;

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

data->test_vec_ops_vec_prod = -1.52435579518716401992;
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

data->test_mat_ops_n = 2;

// Matrix test_mat_ops_A
//----------------------
data->test_mat_ops_A = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_mat_ops_A->m = 2;
data->test_mat_ops_A->n = 2;
data->test_mat_ops_A->nz = -1;
data->test_mat_ops_A->nzmax = 3;
data->test_mat_ops_A->x = (OSQPFloat*) c_malloc(3 * sizeof(OSQPFloat));
data->test_mat_ops_A->x[0] = 0.16450726647410129910;
data->test_mat_ops_A->x[1] = 0.82548781339355581377;
data->test_mat_ops_A->x[2] = 0.06271792257076824750;
data->test_mat_ops_A->i = (OSQPInt*) c_malloc(3 * sizeof(OSQPInt));
data->test_mat_ops_A->i[0] = 1;
data->test_mat_ops_A->i[1] = 0;
data->test_mat_ops_A->i[2] = 1;
data->test_mat_ops_A->p = (OSQPInt*) c_malloc((2 + 1) * sizeof(OSQPInt));
data->test_mat_ops_A->p[0] = 0;
data->test_mat_ops_A->p[1] = 1;
data->test_mat_ops_A->p[2] = 3;

data->test_mat_ops_d = (OSQPFloat*) c_malloc(2 * sizeof(OSQPFloat));
data->test_mat_ops_d[0] = -0.00104879656728068104;
data->test_mat_ops_d[1] = 0.44557355377618612646;


// Matrix test_mat_ops_prem_diag
//------------------------------
data->test_mat_ops_prem_diag = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_mat_ops_prem_diag->m = 2;
data->test_mat_ops_prem_diag->n = 2;
data->test_mat_ops_prem_diag->nz = -1;
data->test_mat_ops_prem_diag->nzmax = 3;
data->test_mat_ops_prem_diag->x = (OSQPFloat*) c_malloc(3 * sizeof(OSQPFloat));
data->test_mat_ops_prem_diag->x[0] = 0.07330008734487135358;
data->test_mat_ops_prem_diag->x[1] = -0.00086576878501919672;
data->test_mat_ops_prem_diag->x[2] = 0.02794544764531688499;
data->test_mat_ops_prem_diag->i = (OSQPInt*) c_malloc(3 * sizeof(OSQPInt));
data->test_mat_ops_prem_diag->i[0] = 1;
data->test_mat_ops_prem_diag->i[1] = 0;
data->test_mat_ops_prem_diag->i[2] = 1;
data->test_mat_ops_prem_diag->p = (OSQPInt*) c_malloc((2 + 1) * sizeof(OSQPInt));
data->test_mat_ops_prem_diag->p[0] = 0;
data->test_mat_ops_prem_diag->p[1] = 1;
data->test_mat_ops_prem_diag->p[2] = 3;


// Matrix test_mat_ops_postm_diag
//-------------------------------
data->test_mat_ops_postm_diag = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_mat_ops_postm_diag->m = 2;
data->test_mat_ops_postm_diag->n = 2;
data->test_mat_ops_postm_diag->nz = -1;
data->test_mat_ops_postm_diag->nzmax = 3;
data->test_mat_ops_postm_diag->x = (OSQPFloat*) c_malloc(3 * sizeof(OSQPFloat));
data->test_mat_ops_postm_diag->x[0] = -0.00017253465637076572;
data->test_mat_ops_postm_diag->x[1] = 0.36781553861269983274;
data->test_mat_ops_postm_diag->x[2] = 0.02794544764531688499;
data->test_mat_ops_postm_diag->i = (OSQPInt*) c_malloc(3 * sizeof(OSQPInt));
data->test_mat_ops_postm_diag->i[0] = 1;
data->test_mat_ops_postm_diag->i[1] = 0;
data->test_mat_ops_postm_diag->i[2] = 1;
data->test_mat_ops_postm_diag->p = (OSQPInt*) c_malloc((2 + 1) * sizeof(OSQPInt));
data->test_mat_ops_postm_diag->p[0] = 0;
data->test_mat_ops_postm_diag->p[1] = 1;
data->test_mat_ops_postm_diag->p[2] = 3;


// Matrix test_mat_ops_scaled
//---------------------------
data->test_mat_ops_scaled = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_mat_ops_scaled->m = 2;
data->test_mat_ops_scaled->n = 2;
data->test_mat_ops_scaled->nz = -1;
data->test_mat_ops_scaled->nzmax = 3;
data->test_mat_ops_scaled->x = (OSQPFloat*) c_malloc(3 * sizeof(OSQPFloat));
data->test_mat_ops_scaled->x[0] = 0.32901453294820259821;
data->test_mat_ops_scaled->x[1] = 1.65097562678711162754;
data->test_mat_ops_scaled->x[2] = 0.12543584514153649501;
data->test_mat_ops_scaled->i = (OSQPInt*) c_malloc(3 * sizeof(OSQPInt));
data->test_mat_ops_scaled->i[0] = 1;
data->test_mat_ops_scaled->i[1] = 0;
data->test_mat_ops_scaled->i[2] = 1;
data->test_mat_ops_scaled->p = (OSQPInt*) c_malloc((2 + 1) * sizeof(OSQPInt));
data->test_mat_ops_scaled->p[0] = 0;
data->test_mat_ops_scaled->p[1] = 1;
data->test_mat_ops_scaled->p[2] = 3;

data->test_mat_ops_inf_norm_cols = (OSQPFloat*) c_malloc(2 * sizeof(OSQPFloat));
data->test_mat_ops_inf_norm_cols[0] = 0.16450726647410129910;
data->test_mat_ops_inf_norm_cols[1] = 0.82548781339355581377;

data->test_mat_ops_inf_norm_rows = (OSQPFloat*) c_malloc(2 * sizeof(OSQPFloat));
data->test_mat_ops_inf_norm_rows[0] = 0.82548781339355581377;
data->test_mat_ops_inf_norm_rows[1] = 0.16450726647410129910;

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
data->test_mat_vec_A->x[0] = 0.28380648894413107453;
data->test_mat_vec_A->x[1] = 0.25686746722710274149;
data->test_mat_vec_A->x[2] = 0.44839630934448426736;
data->test_mat_vec_A->x[3] = 0.58651832682553139975;
data->test_mat_vec_A->x[4] = 0.24321546430632712266;
data->test_mat_vec_A->x[5] = 0.69789357068308133236;
data->test_mat_vec_A->x[6] = 0.36500726350855894342;
data->test_mat_vec_A->x[7] = 0.42092139461746291840;
data->test_mat_vec_A->x[8] = 0.12867321231716943863;
data->test_mat_vec_A->x[9] = 0.10973466400669673604;
data->test_mat_vec_A->x[10] = 0.76312853254405321746;
data->test_mat_vec_A->x[11] = 0.36769956969000661129;
data->test_mat_vec_A->x[12] = 0.72647361031237045470;
data->test_mat_vec_A->x[13] = 0.20324154408739658617;
data->test_mat_vec_A->x[14] = 0.45592896304374885830;
data->test_mat_vec_A->x[15] = 0.37623850142809422969;
data->test_mat_vec_A->x[16] = 0.83968460360894237038;
data->test_mat_vec_A->x[17] = 0.07319007239096597672;
data->test_mat_vec_A->x[18] = 0.25780311899673657994;
data->test_mat_vec_A->x[19] = 0.66498424636196074022;
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
data->test_mat_vec_Pu->x[0] = 1.57017058519391805582;
data->test_mat_vec_Pu->x[1] = 0.50606492252937296250;
data->test_mat_vec_Pu->x[2] = 0.29809604674142509140;
data->test_mat_vec_Pu->x[3] = 1.25997418678525519020;
data->test_mat_vec_Pu->x[4] = 0.76877175990916646331;
data->test_mat_vec_Pu->x[5] = 0.80327244777703499246;
data->test_mat_vec_Pu->x[6] = 1.05982936597406407486;
data->test_mat_vec_Pu->x[7] = 0.52562952316225408644;
data->test_mat_vec_Pu->x[8] = 0.62114552463231109680;
data->test_mat_vec_Pu->x[9] = 0.59046851132539157625;
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
data->test_mat_vec_x[0] = -0.90047506999905169156;
data->test_mat_vec_x[1] = -0.39054534435532289871;
data->test_mat_vec_x[2] = 1.62730025462306726602;
data->test_mat_vec_x[3] = -1.17553590492078052776;

data->test_mat_vec_y = (OSQPFloat*) c_malloc(5 * sizeof(OSQPFloat));
data->test_mat_vec_y[0] = 0.16007589253434076348;
data->test_mat_vec_y[1] = -2.13782435809057735909;
data->test_mat_vec_y[2] = -0.00156693314519690589;
data->test_mat_vec_y[3] = 0.89956641747600707415;
data->test_mat_vec_y[4] = -0.23666332206145512806;

data->test_mat_vec_Ax = (OSQPFloat*) c_malloc(5 * sizeof(OSQPFloat));
data->test_mat_vec_Ax[0] = 0.27143763519281105534;
data->test_mat_vec_Ax[1] = -0.76257643492109372652;
data->test_mat_vec_Ax[2] = 0.52799454400723699887;
data->test_mat_vec_Ax[3] = -0.55071966175548414668;
data->test_mat_vec_Ax[4] = -0.30164536454234996965;

data->test_mat_vec_Ax_cum = (OSQPFloat*) c_malloc(5 * sizeof(OSQPFloat));
data->test_mat_vec_Ax_cum[0] = 0.43151352772715179107;
data->test_mat_vec_Ax_cum[1] = -2.90040079301167086356;
data->test_mat_vec_Ax_cum[2] = 0.52642761086204004073;
data->test_mat_vec_Ax_cum[3] = 0.34884675572052292747;
data->test_mat_vec_Ax_cum[4] = -0.53830868660380515323;

data->test_mat_vec_ATy = (OSQPFloat*) c_malloc(4 * sizeof(OSQPFloat));
data->test_mat_vec_ATy[0] = -0.03435754796664940158;
data->test_mat_vec_ATy[1] = -0.57948510777209627509;
data->test_mat_vec_ATy[2] = -0.59012934647739490046;
data->test_mat_vec_ATy[3] = -1.66045252142979160581;

data->test_mat_vec_ATy_cum = (OSQPFloat*) c_malloc(4 * sizeof(OSQPFloat));
data->test_mat_vec_ATy_cum[0] = -0.93483261796570105151;
data->test_mat_vec_ATy_cum[1] = -0.97003045212741922931;
data->test_mat_vec_ATy_cum[2] = 1.03717090814567236556;
data->test_mat_vec_ATy_cum[3] = -2.83598842635057213357;

data->test_mat_vec_Px = (OSQPFloat*) c_malloc(4 * sizeof(OSQPFloat));
data->test_mat_vec_Px[0] = -0.80705192486612986613;
data->test_mat_vec_Px[1] = 0.06100723371929939187;
data->test_mat_vec_Px[2] = -0.85782898333369717037;
data->test_mat_vec_Px[3] = -0.84295875103525430561;

data->test_mat_vec_Px_cum = (OSQPFloat*) c_malloc(4 * sizeof(OSQPFloat));
data->test_mat_vec_Px_cum[0] = -1.70752699486518144667;
data->test_mat_vec_Px_cum[1] = -0.32953811063602350684;
data->test_mat_vec_Px_cum[2] = 0.76947127128937009566;
data->test_mat_vec_Px_cum[3] = -2.01849465595603483337;

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
data->test_submat_A4->x[0] = 0.28380648894413107453;
data->test_submat_A4->x[1] = 0.25686746722710274149;
data->test_submat_A4->x[2] = 0.58651832682553139975;
data->test_submat_A4->x[3] = 0.24321546430632712266;
data->test_submat_A4->x[4] = 0.69789357068308133236;
data->test_submat_A4->x[5] = 0.36500726350855894342;
data->test_submat_A4->x[6] = 0.12867321231716943863;
data->test_submat_A4->x[7] = 0.10973466400669673604;
data->test_submat_A4->x[8] = 0.76312853254405321746;
data->test_submat_A4->x[9] = 0.36769956969000661129;
data->test_submat_A4->x[10] = 0.20324154408739658617;
data->test_submat_A4->x[11] = 0.45592896304374885830;
data->test_submat_A4->x[12] = 0.37623850142809422969;
data->test_submat_A4->x[13] = 0.83968460360894237038;
data->test_submat_A4->x[14] = 0.25780311899673657994;
data->test_submat_A4->x[15] = 0.66498424636196074022;
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
data->test_submat_A5->x[0] = 0.28380648894413107453;
data->test_submat_A5->x[1] = 0.25686746722710274149;
data->test_submat_A5->x[2] = 0.44839630934448426736;
data->test_submat_A5->x[3] = 0.58651832682553139975;
data->test_submat_A5->x[4] = 0.24321546430632712266;
data->test_submat_A5->x[5] = 0.69789357068308133236;
data->test_submat_A5->x[6] = 0.36500726350855894342;
data->test_submat_A5->x[7] = 0.42092139461746291840;
data->test_submat_A5->x[8] = 0.12867321231716943863;
data->test_submat_A5->x[9] = 0.10973466400669673604;
data->test_submat_A5->x[10] = 0.76312853254405321746;
data->test_submat_A5->x[11] = 0.36769956969000661129;
data->test_submat_A5->x[12] = 0.72647361031237045470;
data->test_submat_A5->x[13] = 0.20324154408739658617;
data->test_submat_A5->x[14] = 0.45592896304374885830;
data->test_submat_A5->x[15] = 0.37623850142809422969;
data->test_submat_A5->x[16] = 0.83968460360894237038;
data->test_submat_A5->x[17] = 0.07319007239096597672;
data->test_submat_A5->x[18] = 0.25780311899673657994;
data->test_submat_A5->x[19] = 0.66498424636196074022;
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
data->test_submat_A3->x[0] = 0.28380648894413107453;
data->test_submat_A3->x[1] = 0.44839630934448426736;
data->test_submat_A3->x[2] = 0.24321546430632712266;
data->test_submat_A3->x[3] = 0.69789357068308133236;
data->test_submat_A3->x[4] = 0.42092139461746291840;
data->test_submat_A3->x[5] = 0.10973466400669673604;
data->test_submat_A3->x[6] = 0.76312853254405321746;
data->test_submat_A3->x[7] = 0.72647361031237045470;
data->test_submat_A3->x[8] = 0.45592896304374885830;
data->test_submat_A3->x[9] = 0.37623850142809422969;
data->test_submat_A3->x[10] = 0.07319007239096597672;
data->test_submat_A3->x[11] = 0.66498424636196074022;
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
data->test_mat_extr_triu_P->nzmax = 22;
data->test_mat_extr_triu_P->x = (OSQPFloat*) c_malloc(22 * sizeof(OSQPFloat));
data->test_mat_extr_triu_P->x[0] = 0.25524137299213922603;
data->test_mat_extr_triu_P->x[1] = 0.26214679618998948385;
data->test_mat_extr_triu_P->x[2] = 0.63216769755967661126;
data->test_mat_extr_triu_P->x[3] = 1.42197396103381867860;
data->test_mat_extr_triu_P->x[4] = 0.26214679618998948385;
data->test_mat_extr_triu_P->x[5] = 1.61207141505424722538;
data->test_mat_extr_triu_P->x[6] = 0.76078878458348253577;
data->test_mat_extr_triu_P->x[7] = 1.41109611195898132507;
data->test_mat_extr_triu_P->x[8] = 1.36172273938178589603;
data->test_mat_extr_triu_P->x[9] = 0.63216769755967661126;
data->test_mat_extr_triu_P->x[10] = 0.76078878458348253577;
data->test_mat_extr_triu_P->x[11] = 0.44501373189254489482;
data->test_mat_extr_triu_P->x[12] = 0.92388695737180137613;
data->test_mat_extr_triu_P->x[13] = 0.99290570493346841374;
data->test_mat_extr_triu_P->x[14] = 1.42197396103381867860;
data->test_mat_extr_triu_P->x[15] = 1.41109611195898132507;
data->test_mat_extr_triu_P->x[16] = 0.92388695737180137613;
data->test_mat_extr_triu_P->x[17] = 0.60701785319902135107;
data->test_mat_extr_triu_P->x[18] = 1.23312125465709998551;
data->test_mat_extr_triu_P->x[19] = 1.36172273938178589603;
data->test_mat_extr_triu_P->x[20] = 0.99290570493346841374;
data->test_mat_extr_triu_P->x[21] = 1.23312125465709998551;
data->test_mat_extr_triu_P->i = (OSQPInt*) c_malloc(22 * sizeof(OSQPInt));
data->test_mat_extr_triu_P->i[0] = 0;
data->test_mat_extr_triu_P->i[1] = 1;
data->test_mat_extr_triu_P->i[2] = 2;
data->test_mat_extr_triu_P->i[3] = 3;
data->test_mat_extr_triu_P->i[4] = 0;
data->test_mat_extr_triu_P->i[5] = 1;
data->test_mat_extr_triu_P->i[6] = 2;
data->test_mat_extr_triu_P->i[7] = 3;
data->test_mat_extr_triu_P->i[8] = 4;
data->test_mat_extr_triu_P->i[9] = 0;
data->test_mat_extr_triu_P->i[10] = 1;
data->test_mat_extr_triu_P->i[11] = 2;
data->test_mat_extr_triu_P->i[12] = 3;
data->test_mat_extr_triu_P->i[13] = 4;
data->test_mat_extr_triu_P->i[14] = 0;
data->test_mat_extr_triu_P->i[15] = 1;
data->test_mat_extr_triu_P->i[16] = 2;
data->test_mat_extr_triu_P->i[17] = 3;
data->test_mat_extr_triu_P->i[18] = 4;
data->test_mat_extr_triu_P->i[19] = 1;
data->test_mat_extr_triu_P->i[20] = 2;
data->test_mat_extr_triu_P->i[21] = 3;
data->test_mat_extr_triu_P->p = (OSQPInt*) c_malloc((5 + 1) * sizeof(OSQPInt));
data->test_mat_extr_triu_P->p[0] = 0;
data->test_mat_extr_triu_P->p[1] = 4;
data->test_mat_extr_triu_P->p[2] = 9;
data->test_mat_extr_triu_P->p[3] = 14;
data->test_mat_extr_triu_P->p[4] = 19;
data->test_mat_extr_triu_P->p[5] = 22;


// Matrix test_mat_extr_triu_Pu
//-----------------------------
data->test_mat_extr_triu_Pu = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_mat_extr_triu_Pu->m = 5;
data->test_mat_extr_triu_Pu->n = 5;
data->test_mat_extr_triu_Pu->nz = -1;
data->test_mat_extr_triu_Pu->nzmax = 13;
data->test_mat_extr_triu_Pu->x = (OSQPFloat*) c_malloc(13 * sizeof(OSQPFloat));
data->test_mat_extr_triu_Pu->x[0] = 0.25524137299213922603;
data->test_mat_extr_triu_Pu->x[1] = 0.26214679618998948385;
data->test_mat_extr_triu_Pu->x[2] = 1.61207141505424722538;
data->test_mat_extr_triu_Pu->x[3] = 0.63216769755967661126;
data->test_mat_extr_triu_Pu->x[4] = 0.76078878458348253577;
data->test_mat_extr_triu_Pu->x[5] = 0.44501373189254489482;
data->test_mat_extr_triu_Pu->x[6] = 1.42197396103381867860;
data->test_mat_extr_triu_Pu->x[7] = 1.41109611195898132507;
data->test_mat_extr_triu_Pu->x[8] = 0.92388695737180137613;
data->test_mat_extr_triu_Pu->x[9] = 0.60701785319902135107;
data->test_mat_extr_triu_Pu->x[10] = 1.36172273938178589603;
data->test_mat_extr_triu_Pu->x[11] = 0.99290570493346841374;
data->test_mat_extr_triu_Pu->x[12] = 1.23312125465709998551;
data->test_mat_extr_triu_Pu->i = (OSQPInt*) c_malloc(13 * sizeof(OSQPInt));
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
data->test_mat_extr_triu_Pu->i[10] = 1;
data->test_mat_extr_triu_Pu->i[11] = 2;
data->test_mat_extr_triu_Pu->i[12] = 3;
data->test_mat_extr_triu_Pu->p = (OSQPInt*) c_malloc((5 + 1) * sizeof(OSQPInt));
data->test_mat_extr_triu_Pu->p[0] = 0;
data->test_mat_extr_triu_Pu->p[1] = 1;
data->test_mat_extr_triu_Pu->p[2] = 3;
data->test_mat_extr_triu_Pu->p[3] = 6;
data->test_mat_extr_triu_Pu->p[4] = 10;
data->test_mat_extr_triu_Pu->p[5] = 13;

data->test_mat_extr_triu_P_inf_norm_cols = (OSQPFloat*) c_malloc(5 * sizeof(OSQPFloat));
data->test_mat_extr_triu_P_inf_norm_cols[0] = 1.42197396103381867860;
data->test_mat_extr_triu_P_inf_norm_cols[1] = 1.61207141505424722538;
data->test_mat_extr_triu_P_inf_norm_cols[2] = 0.99290570493346841374;
data->test_mat_extr_triu_P_inf_norm_cols[3] = 1.42197396103381867860;
data->test_mat_extr_triu_P_inf_norm_cols[4] = 1.36172273938178589603;

data->test_qpform_n = 4;

// Matrix test_qpform_Pu
//----------------------
data->test_qpform_Pu = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_qpform_Pu->m = 4;
data->test_qpform_Pu->n = 4;
data->test_qpform_Pu->nz = -1;
data->test_qpform_Pu->nzmax = 9;
data->test_qpform_Pu->x = (OSQPFloat*) c_malloc(9 * sizeof(OSQPFloat));
data->test_qpform_Pu->x[0] = 1.02060649947737069887;
data->test_qpform_Pu->x[1] = 0.61655177862301391301;
data->test_qpform_Pu->x[2] = 0.31711006797342533581;
data->test_qpform_Pu->x[3] = 0.45114855199277636988;
data->test_qpform_Pu->x[4] = 0.27813685014104094773;
data->test_qpform_Pu->x[5] = 0.37358735870761705655;
data->test_qpform_Pu->x[6] = 0.67468939543477746135;
data->test_qpform_Pu->x[7] = 0.87172535865363898200;
data->test_qpform_Pu->x[8] = 0.95202926872618476306;
data->test_qpform_Pu->i = (OSQPInt*) c_malloc(9 * sizeof(OSQPInt));
data->test_qpform_Pu->i[0] = 0;
data->test_qpform_Pu->i[1] = 0;
data->test_qpform_Pu->i[2] = 1;
data->test_qpform_Pu->i[3] = 0;
data->test_qpform_Pu->i[4] = 1;
data->test_qpform_Pu->i[5] = 2;
data->test_qpform_Pu->i[6] = 0;
data->test_qpform_Pu->i[7] = 1;
data->test_qpform_Pu->i[8] = 2;
data->test_qpform_Pu->p = (OSQPInt*) c_malloc((4 + 1) * sizeof(OSQPInt));
data->test_qpform_Pu->p[0] = 0;
data->test_qpform_Pu->p[1] = 1;
data->test_qpform_Pu->p[2] = 3;
data->test_qpform_Pu->p[3] = 6;
data->test_qpform_Pu->p[4] = 9;

data->test_qpform_x = (OSQPFloat*) c_malloc(4 * sizeof(OSQPFloat));
data->test_qpform_x[0] = -1.26449279081971233119;
data->test_qpform_x[1] = 0.51878492087796412857;
data->test_qpform_x[2] = -1.14251801765249205722;
data->test_qpform_x[3] = -0.74585639818832494274;

data->test_qpform_value = 2.29520176806866293973;

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
c_free(data->test_vec_ops_v1);
c_free(data->test_vec_ops_v2);
c_free(data->test_vec_ops_add_scaled);
c_free(data->test_vec_ops_ew_reciprocal);
c_free(data->test_vec_ops_ew_max_vec);
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

