#include "catch.hpp"

#include "lin_alg/test_lin_alg.h"
#include "solve_linsys/test_solve_linsys.h"
#include "demo/test_demo.h"
#include "basic_qp/test_basic_qp.h"
#include "basic_qp2/test_basic_qp2.h"
#include "non_cvx/test_non_cvx.h"
#include "primal_dual_infeasibility/test_primal_dual_infeasibility.h"
#include "primal_infeasibility/test_primal_infeasibility.h"
#include "unconstrained/test_unconstrained.h"
#include "update_matrices/test_update_matrices.h"


TEST_CASE( "test_lin_alg", "[multi-file:2]" ) {
    SECTION( "test_constr_sparse_mat" ) {
        REQUIRE( test_constr_sparse_mat() == NULL );
    }
    SECTION( "test_vec_operations" ) {
        REQUIRE( test_vec_operations() == NULL );
    }
    SECTION( "test_mat_operations" ) {
        REQUIRE( test_mat_operations() == NULL );
    }
    SECTION( "test_mat_vec_multiplication" ) {
        REQUIRE( test_mat_vec_multiplication() == NULL );
    }
    SECTION( "test_extract_upper_triangular" ) {
        REQUIRE( test_extract_upper_triangular() == NULL );
    }
    SECTION( "test_quad_form_upper_triang" ) {
        REQUIRE( test_quad_form_upper_triang() == NULL );
    }
}


TEST_CASE( "test_solve_linsys", "[multi-file:3]" ) {
    SECTION( "test_solveKKT" ) {
        REQUIRE( test_solveKKT() == NULL );
    }
#ifdef ENABLE_MKL_PARDISO
    SECTION( "test_solveKKT_pardiso" ) {
        REQUIRE( test_solveKKT_pardiso() == NULL );
    }
#endif
}


TEST_CASE( "test_demo", "[multi-file:4]" ) {
    SECTION("test_demo_solve") {
        REQUIRE(test_demo_solve() == NULL);
    }
}


TEST_CASE( "test_basic_qp", "[multi-file:5]" ) {
    SECTION( "test_basic_qp_solve" ) {
        REQUIRE( test_basic_qp_solve() == NULL );
    }
#ifdef ENABLE_MKL_PARDISO
    SECTION( "test_basic_qp_solve_pardiso" ) {
        REQUIRE( test_basic_qp_solve_pardiso() == NULL );
    }
#endif
    SECTION( "test_basic_qp_update" ) {
        REQUIRE( test_basic_qp_update() == NULL );
    }
    SECTION( "test_basic_qp_check_termination" ) {
        REQUIRE( test_basic_qp_check_termination() == NULL );
    }
    SECTION( "test_basic_qp_update_rho" ) {
        REQUIRE( test_basic_qp_update_rho() == NULL );
    }
#ifdef PROFILING
    SECTION( "test_basic_qp_time_limit" ) {
        REQUIRE( test_basic_qp_time_limit() == NULL );
    }
#endif
    SECTION( "test_basic_qp_warm_start" ) {
        REQUIRE( test_basic_qp_warm_start() == NULL );
    }
}


TEST_CASE( "test_basic_qp2", "[multi-file:6]" ) {
    SECTION( "test_basic_qp2_solve" ) {
        REQUIRE( test_basic_qp2_solve() == NULL );
    }
#ifdef ENABLE_MKL_PARDISO
    SECTION( "test_basic_qp2_solve_pardiso" ) {
        REQUIRE( test_basic_qp2_solve_pardiso() == NULL );
    }
#endif
    SECTION( "test_basic_qp2_update" ) {
        REQUIRE( test_basic_qp2_update() == NULL );
    }
}


TEST_CASE( "test_non_cvx", "[multi-file:7]" ) {
    SECTION( "test_non_cvx_solve" ) {
        REQUIRE( test_non_cvx_solve() == NULL );
    }
}


TEST_CASE( "test_primal_infeasibility", "[multi-file:8]" ) {
    SECTION( "test_primal_infeasible_qp_solve" ) {
        REQUIRE( test_primal_infeasible_qp_solve() == NULL );
    }
}


TEST_CASE( "test_primal_dual_infeasibility", "[multi-file:9]" ) {
    SECTION( "test_optimal" ) {
        REQUIRE( test_optimal() == NULL );
    }
    SECTION( "test_prim_infeas" ) {
        REQUIRE( test_prim_infeas() == NULL );
    }
    SECTION( "test_dual_infeas" ) {
        REQUIRE( test_dual_infeas() == NULL );
    }
    SECTION( "test_primal_dual_infeas" ) {
        REQUIRE( test_primal_dual_infeas() == NULL );
    }
}


TEST_CASE( "test_unconstrained", "[multi-file:10]" ) {
    SECTION( "test_unconstrained_solve" ) {
        REQUIRE( test_unconstrained_solve() == NULL );
    }
}


TEST_CASE( "test_update_matrices", "[multi-file:11]" ) {
    SECTION( "test_form_KKT" ) {
        REQUIRE( test_form_KKT() == NULL );
    }
    SECTION( "test_update" ) {
        REQUIRE( test_update() == NULL );
    }
#ifdef ENABLE_MKL_PARDISO
    SECTION( "test_update_pardiso" ) {
        REQUIRE( test_update_pardiso() == NULL );
    }
#endif
}