/*
 * This file was autogenerated by OSQP on Thu Sep 26 13:02:44 2024
 *
 * This file contains the workspace variables needed by OSQP.
 */

#include "hs21.h"
#include "types.h"


/* Define the settings structure */
OSQPSettings hs21_settings = {
  0,
  OSQP_DIRECT_SOLVER,
  1,
  0,
  0,
  1,
  10,
  0,
  (OSQPFloat)0.10000000000000000555,
  1,
  (OSQPFloat)0.00000100000000000000,
  (OSQPFloat)1.60000000000000008882,
  20,
  10,
  (OSQPFloat)0.14999999999999999445,
  1,
  2,
  0,
  (OSQPFloat)0.40000000000000002220,
  (OSQPFloat)5.00000000000000000000,
  1000000000,
  (OSQPFloat)0.00000100000000000000,
  (OSQPFloat)0.00000100000000000000,
  (OSQPFloat)0.00000000000000100000,
  (OSQPFloat)0.00000000000000100000,
  0,
  25,
  1,
  (OSQPFloat)1000.00000000000000000000,
  (OSQPFloat)0.00000100000000000000,
  3,

  // Restart settings
  1,                    // restart_enable
//  (OSQPFloat)0.2,       // restart_sufficient
  (OSQPFloat)0.39,       // restart_sufficient
//  (OSQPFloat)0.8,       // restart_necessary
  (OSQPFloat)0.82,       // restart_necessary
  (OSQPFloat)0.36       // restart_artificial
};


/* Define the data structure */
OSQPInt hs21_data_n = 2;
OSQPInt hs21_data_m = 3;

OSQPInt hs21_data_P_csc_p[3] = {
  0,
  1,
  2,
};
OSQPInt hs21_data_P_csc_i[2] = {
  0,
  1,
};
OSQPFloat hs21_data_P_csc_x[2] = {
  (OSQPFloat)0.00200000000000000004,
  (OSQPFloat)1.00000000000000022204,
};
OSQPCscMatrix hs21_data_P_csc = {
  2,
  2,
  hs21_data_P_csc_p,
  hs21_data_P_csc_i,
  hs21_data_P_csc_x,
  2,
  -1,
};

OSQPInt hs21_data_A_csc_p[3] = {
  0,
  2,
  4,
};
OSQPInt hs21_data_A_csc_i[4] = {
  0,
  1,
  0,
  2,
};
OSQPFloat hs21_data_A_csc_x[4] = {
  (OSQPFloat)1.00000000000000000000,
  (OSQPFloat)0.99775390799327379199,
  (OSQPFloat)-0.22360679774997901936,
  (OSQPFloat)0.99932332750265073784,
};
OSQPCscMatrix hs21_data_A_csc = {
  3,
  2,
  hs21_data_A_csc_p,
  hs21_data_A_csc_i,
  hs21_data_A_csc_x,
  4,
  -1,
};

OSQPFloat hs21_data_q_val[2] = {
  (OSQPFloat)0.00000000000000000000,
  (OSQPFloat)0.00000000000000000000,
};
OSQPFloat hs21_data_l_val[3] = {
  (OSQPFloat)3.16227766016837952279,
  (OSQPFloat)6.31034978718565131572,
  (OSQPFloat)-70.66283014750294455553,
};
OSQPFloat hs21_data_u_val[3] = {
  (OSQPFloat)31622776601683795968.00000000000000000000,
  (OSQPFloat)157.75874467964129621578,
  (OSQPFloat)70.66283014750294455553,
};