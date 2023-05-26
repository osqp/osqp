#ifndef TEST_LIN_ALG_H_
#define TEST_LIN_ALG_H_

#include <stdio.h>
#include <catch2/catch.hpp>

#include "osqp.h"

/* Main linar algebra includes */
#include "algebra_matrix.h"
#include "algebra_vector.h"
#include "lin_alg.h"
#include "util.h"

#ifndef OSQP_ALGEBRA_CUDA
#include "csc_utils.h"
#endif

//helper functions
#include "test_utils.h"
#include "osqp_tester.h"

#endif
