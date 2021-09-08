/* OSQP TESTER MODULE */

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "osqp.h"
#include "osqp_tester.h"
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


TEST_CASE( "test_lin_alg", "[multi-file:1]" ) {
    SECTION( "test_constr_sparse_mat" ) {
        REQUIRE_NOTHROW( test_constr_sparse_mat() );
    }
    SECTION( "test_vec_operations" ) {
        REQUIRE_NOTHROW( test_vec_operations() );
    }
    SECTION( "test_mat_operations" ) {
        REQUIRE_NOTHROW( test_mat_operations() );
    }
    SECTION( "test_mat_vec_multiplication" ) {
        REQUIRE_NOTHROW( test_mat_vec_multiplication() );
    }
    SECTION( "test_extract_upper_triangular" ) {
        REQUIRE_NOTHROW( test_extract_upper_triangular() );
    }
    SECTION( "test_quad_form_upper_triang" ) {
        REQUIRE_NOTHROW( test_quad_form_upper_triang() );
    }
}


TEST_CASE( "test_solve_linsys", "[multi-file:2]" ) {
    SECTION( "test_solveKKT" ) {
        REQUIRE_NOTHROW( test_solveKKT() );
    }
#ifdef ENABLE_MKL_PARDISO
    SECTION( "test_solveKKT_pardiso" ) {
        REQUIRE_NOTHROW( test_solveKKT_pardiso() );
    }
#endif
}


TEST_CASE( "test_demo", "[multi-file:3]" ) {
    SECTION("test_demo_solve") {
        REQUIRE_NOTHROW( test_demo_solve() );
    }
}


TEST_CASE( "test_basic_qp", "[multi-file:4]" ) {
    SECTION( "test_basic_qp_solve" ) {
        REQUIRE_NOTHROW( test_basic_qp_solve() );
    }
#ifdef ENABLE_MKL_PARDISO
        SECTION( "test_basic_qp_solve_pardiso" ) {
        REQUIRE_NOTHROW( test_basic_qp_solve_pardiso() );
    }
#endif
    SECTION( "test_basic_qp_update" ) {
        REQUIRE_NOTHROW( test_basic_qp_update() );
    }
    SECTION( "test_basic_qp_check_termination" ) {
        REQUIRE_NOTHROW( test_basic_qp_check_termination() );
    }
    SECTION( "test_basic_qp_update_rho" ) {
        REQUIRE_NOTHROW( test_basic_qp_update_rho() );
    }
#ifdef PROFILING
    SECTION( "test_basic_qp_time_limit" ) {
        REQUIRE_NOTHROW( test_basic_qp_time_limit() );
    }
#endif
    SECTION( "test_basic_qp_warm_start" ) {
        REQUIRE_NOTHROW( test_basic_qp_warm_start() );
    }
}


TEST_CASE( "test_basic_qp2", "[multi-file:5]" ) {
    SECTION( "test_basic_qp2_solve" ) {
        REQUIRE_NOTHROW( test_basic_qp2_solve() );
    }
#ifdef ENABLE_MKL_PARDISO
        SECTION( "test_basic_qp2_solve_pardiso" ) {
        REQUIRE( test_basic_qp2_solve_pardiso() == 0 );
    }
#endif
    SECTION( "test_basic_qp2_update" ) {
        REQUIRE_NOTHROW( test_basic_qp2_update() );
    }
}


TEST_CASE( "test_non_cvx", "[multi-file:6]" ) {
    SECTION( "test_non_cvx_solve" ) {
        REQUIRE_NOTHROW( test_non_cvx_solve() );
    }
}


TEST_CASE( "test_primal_infeasibility", "[multi-file:7]" ) {
    SECTION( "test_primal_infeasible_qp_solve" ) {
        REQUIRE_NOTHROW( test_primal_infeasible_qp_solve() );
    }
}


TEST_CASE( "test_primal_dual_infeasibility", "[multi-file:8]" ) {
    SECTION( "test_optimal" ) {
        REQUIRE_NOTHROW( test_optimal() );
    }
    SECTION( "test_prim_infeas" ) {
        REQUIRE_NOTHROW( test_prim_infeas() );
    }
    SECTION( "test_dual_infeas" ) {
        REQUIRE_NOTHROW( test_dual_infeas() );
    }
    SECTION( "test_primal_dual_infeas" ) {
        REQUIRE_NOTHROW( test_primal_dual_infeas() );
    }
}


TEST_CASE( "test_unconstrained", "[multi-file:9]" ) {
    SECTION( "test_unconstrained_solve" ) {
        REQUIRE_NOTHROW( test_unconstrained_solve() );
    }
}


TEST_CASE( "test_update_matrices", "[multi-file:10]" ) {
    SECTION( "test_form_KKT" ) {
        REQUIRE_NOTHROW( test_form_KKT() );
    }
    SECTION( "test_update" ) {
        REQUIRE_NOTHROW( test_update() );
    }
#ifdef ENABLE_MKL_PARDISO
    SECTION( "test_update_pardiso" ) {
        REQUIRE_NOTHROW( test_update_pardiso() );
    }
#endif
}
