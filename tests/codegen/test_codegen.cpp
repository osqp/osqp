#include <catch2/catch.hpp>

#include "osqp_api.h"    /* OSQP API wrapper (public + some private) */
#include "osqp_tester.h" /* Tester helpers */
#include "test_utils.h"  /* Testing Helper functions */

#include "codegen_data.h"
#include "basic_lp_data.h"
#include "non_cvx_data.h"
#include "unconstrained_data.h"

#ifdef OSQP_CODEGEN
TEST_CASE("Basic codegen", "[codegen]")
{
  OSQPInt exitflag;

  // Problem settings
  OSQPSettings_ptr settings{(OSQPSettings *)c_malloc(sizeof(OSQPSettings))};

  // Codegen defines
  OSQPCodegenDefines_ptr defines{(OSQPCodegenDefines *)c_malloc(sizeof(OSQPCodegenDefines))};

  // Structures
  OSQPSolver *tmpSolver = nullptr;
  OSQPSolver_ptr solver{nullptr};   // Wrap solver inside memory management

  // Problem data
  codegen_problem_ptr   data{generate_problem_codegen()};
  codegen_sols_data_ptr sols_data{generate_problem_codegen_sols_data()};

  // Define Solver settings as default
  osqp_set_default_settings(settings.get());
  settings->max_iter      = 2000;
  settings->alpha         = 1.6;
  settings->polishing     = 1;
  settings->scaling       = 0;
  settings->verbose       = 1;
  settings->warm_starting = 0;

  // Define codegen settings
  defines->embedded_mode = 1;      // vector update
  defines->float_type = 1;         // floats
  defines->printing_enable = 0;    // no printing
  defines->profiling_enable = 0;   // no timing
  defines->interrupt_enable = 0;   // no interrupts
  defines->derivatives_enable = 0; // no derivatives

  // Setup solver
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Setup error!", exitflag == 0);

  // Vector update
  defines->embedded_mode = 1;

  exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "basic_vector_", defines.get());

  mu_assert("Codegen type 1 should have worked!",
            exitflag == OSQP_NO_ERROR);

  // matrix update
  defines->embedded_mode = 2;

  exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "basic_matrix_", defines.get());

  mu_assert("Codegen type 2 should have worked!",
            exitflag == OSQP_NO_ERROR);
}

TEST_CASE("Codegen: Data export", "[codegen],[unconstrained],[lp],[qp],[nonconvex]")
{
  OSQPInt exitflag;

  // Problem settings
  OSQPSettings_ptr settings{(OSQPSettings *)c_malloc(sizeof(OSQPSettings))};

  // Codegen defines
  OSQPCodegenDefines_ptr defines{(OSQPCodegenDefines *)c_malloc(sizeof(OSQPCodegenDefines))};

  // Structures
  OSQPSolver *tmpSolver = nullptr;
  OSQPSolver_ptr solver{nullptr};   // Wrap solver inside memory management

  // Define Solver settings as default
  osqp_set_default_settings(settings.get());
  settings->max_iter      = 2000;
  settings->alpha         = 1.6;
  settings->polishing     = 1;
  settings->scaling       = 0;
  settings->verbose       = 1;
  settings->warm_starting = 0;

  // Define codegen settings
  defines->embedded_mode = 1;      // vector update
  defines->float_type = 1;         // floats
  defines->printing_enable = 0;    // no printing
  defines->profiling_enable = 0;   // no timing
  defines->interrupt_enable = 0;   // no interrupts
  defines->derivatives_enable = 0; // no derivatives

  SECTION( "Unconstrained" ) {
    OSQPInt embedded;
    std::string dir;

    std::tie( embedded, dir ) =
      GENERATE( table<OSQPInt, std::string>(
          { /* first is embedded mode, second is output directory */
            std::make_tuple( 1, CODEGEN1_DIR ),
            std::make_tuple( 2, CODEGEN2_DIR ) } ) );

    char name[100];
    snprintf(name, 100, "data_unconstrained_embedded_%d_", embedded);

    CAPTURE(embedded);

    // Problem data
    unconstrained_problem_ptr   data{generate_problem_unconstrained()};
    unconstrained_sols_data_ptr sols_data{generate_problem_unconstrained_sols_data()};

    // Setup solver
    exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                          data->A, data->l, data->u,
                          data->m, data->n, settings.get());
    solver.reset(tmpSolver);

    // Setup correct
    mu_assert("Setup error!", exitflag == 0);

    defines->embedded_mode = embedded;

    exitflag = osqp_codegen(solver.get(), dir.c_str(), name, defines.get());

    // Codegen should work or error as appropriate
    mu_assert("Unconstrained should have worked!",
              exitflag == OSQP_NO_ERROR);
  }

  SECTION( "Linear program" ) {
    OSQPInt embedded;
    std::string dir;

    std::tie( embedded, dir ) =
      GENERATE( table<OSQPInt, std::string>(
          { /* first is embedded mode, second is output directory */
            std::make_tuple( 1, CODEGEN1_DIR ),
            std::make_tuple( 2, CODEGEN2_DIR ) } ) );

    char name[100];
    snprintf(name, 100, "data_lp_embedded_%d_", embedded);

    CAPTURE(embedded);

    // Problem data
    basic_lp_problem_ptr   data{generate_problem_basic_lp()};
    basic_lp_sols_data_ptr sols_data{generate_problem_basic_lp_sols_data()};

    // Setup solver
    exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                          data->A, data->l, data->u,
                          data->m, data->n, settings.get());
    solver.reset(tmpSolver);

    // Setup correct
    mu_assert("Setup error!", exitflag == 0);

    defines->embedded_mode = embedded;

    exitflag = osqp_codegen(solver.get(), dir.c_str(), name, defines.get());

    // Codegen should work or error as appropriate
    mu_assert("Linear program should have worked!",
              exitflag == OSQP_NO_ERROR);
  }

  SECTION( "Nonconvex" ) {
    OSQPInt embedded;
    std::string dir;

    std::tie( embedded, dir ) =
      GENERATE( table<OSQPInt, std::string>(
          { /* first is embedded mode, second is output directory */
            std::make_tuple( 1, CODEGEN1_DIR ),
            std::make_tuple( 2, CODEGEN2_DIR ) } ) );

    OSQPFloat sigma;
    OSQPInt   sigma_num;
    OSQPInt   expected_error;

    std::tie( sigma, sigma_num, expected_error ) =
      GENERATE( table<OSQPFloat, OSQPInt, OSQPInt>(
          { /* first is sigma value, second is the filename parameter, third is the expected return value */
            std::make_tuple( 1e-6, 1, OSQP_NONCVX_ERROR ),
            std::make_tuple(    5, 2, OSQP_NO_ERROR ) } ) );

    char name[100];
    snprintf(name, 100, "data_nonconvex_%d_embedded_%d_", sigma_num, embedded);

    CAPTURE(embedded, sigma);

    // Problem data
    non_cvx_problem_ptr   data{generate_problem_non_cvx()};
    non_cvx_sols_data_ptr sols_data{generate_problem_non_cvx_sols_data()};

    // Update solver settings
    settings->sigma = sigma;
    defines->embedded_mode = embedded;

    // Setup solver
    exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                          data->A, data->l, data->u,
                          data->m, data->n, settings.get());
    solver.reset(tmpSolver);

    // Setup correct
    mu_assert("Setup error!", exitflag == expected_error);

    exitflag = osqp_codegen(solver.get(), dir.c_str(), name, defines.get());

    // Codegen should work or error as appropriate
    mu_assert("Nonconvex codegen error!",
              exitflag == expected_error);
  }
}

TEST_CASE("Codegen: defines", "[codegen]")
{
  OSQPInt exitflag;

  // Problem settings
  OSQPSettings_ptr settings{(OSQPSettings *)c_malloc(sizeof(OSQPSettings))};

  // Codegen defines
  OSQPCodegenDefines_ptr defines{(OSQPCodegenDefines *)c_malloc(sizeof(OSQPCodegenDefines))};

  // Structures
  OSQPSolver *tmpSolver = nullptr;
  OSQPSolver_ptr solver{nullptr};   // Wrap solver inside memory management

  // Problem data
  codegen_problem_ptr   data{generate_problem_codegen()};
  codegen_sols_data_ptr sols_data{generate_problem_codegen_sols_data()};

  // Define Solver settings as default
  osqp_set_default_settings(settings.get());
  settings->max_iter      = 2000;
  settings->alpha         = 1.6;
  settings->polishing     = 1;
  settings->scaling       = 0;
  settings->verbose       = 1;
  settings->warm_starting = 0;

  // Define codegen settings
  defines->embedded_mode = 1;      // vector update
  defines->float_type = 1;         // floats
  defines->printing_enable = 0;    // no printing
  defines->profiling_enable = 0;   // no timing
  defines->interrupt_enable = 0;   // no interrupts
  defines->derivatives_enable = 0; // no derivatives

  // Setup solver
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Setup error!", exitflag == 0);

  SECTION( "embedded_mode" ) {
    OSQPInt test_input;
    OSQPInt expected_flag;
    std::tie( test_input, expected_flag ) =
        GENERATE( table<OSQPInt, OSQPInt>(
            { /* first is input, second is expected error */
              std::make_tuple( 0, OSQP_CODEGEN_DEFINES_ERROR ),
              std::make_tuple( 1, OSQP_NO_ERROR ),
              std::make_tuple( 2, OSQP_NO_ERROR ),
              std::make_tuple( 3, OSQP_CODEGEN_DEFINES_ERROR ) } ) );

    defines->embedded_mode = test_input;

    CAPTURE(defines->embedded_mode);

    exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "defines_embedded_", defines.get());

    // Codegen should work or error as appropriate
    mu_assert("embedded_mode define should have worked!",
              exitflag == expected_flag);
  }

  SECTION( "float_type" ) {
    OSQPInt test_input;
    OSQPInt expected_flag;
    std::tie( test_input, expected_flag ) =
        GENERATE( table<OSQPInt, OSQPInt>(
            { /* first is input, second is expected error */
              std::make_tuple( -1, OSQP_CODEGEN_DEFINES_ERROR ),
              std::make_tuple(  0, OSQP_NO_ERROR ),
              std::make_tuple(  1, OSQP_NO_ERROR ),
              std::make_tuple(  2, OSQP_CODEGEN_DEFINES_ERROR ),
              std::make_tuple(  3, OSQP_CODEGEN_DEFINES_ERROR ) } ) );

    defines->float_type = test_input;

    CAPTURE(defines->float_type);

    exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "defines_float_", defines.get());

    // Codegen should work or error as appropriate
    mu_assert("float_type define should have worked!",
              exitflag == expected_flag);
  }

  SECTION( "codegen define: printing" ) {
    OSQPInt test_input;
    OSQPInt expected_flag;
    std::tie( test_input, expected_flag ) =
        GENERATE( table<OSQPInt, OSQPInt>(
            { /* first is input, second is expected error */
              std::make_tuple( -1, OSQP_CODEGEN_DEFINES_ERROR ),
              std::make_tuple(  0, OSQP_NO_ERROR ),
              std::make_tuple(  1, OSQP_NO_ERROR ),
              std::make_tuple(  2, OSQP_CODEGEN_DEFINES_ERROR ),
              std::make_tuple(  3, OSQP_CODEGEN_DEFINES_ERROR ) } ) );

    defines->printing_enable = test_input;

    CAPTURE(defines->printing_enable);

    exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "defines_printing_", defines.get());

    // Codegen should work or error as appropriate
    mu_assert("Non Convex codegen: printing define should have worked!",
              exitflag == expected_flag);
  }

  SECTION( "profiling_enabled" ) {
    OSQPInt test_input;
    OSQPInt expected_flag;
    std::tie( test_input, expected_flag ) =
        GENERATE( table<OSQPInt, OSQPInt>(
            { /* first is input, second is expected error */
              std::make_tuple( -1, OSQP_CODEGEN_DEFINES_ERROR ),
              std::make_tuple(  0, OSQP_NO_ERROR ),
              std::make_tuple(  1, OSQP_NO_ERROR ),
              std::make_tuple(  2, OSQP_CODEGEN_DEFINES_ERROR ),
              std::make_tuple(  3, OSQP_CODEGEN_DEFINES_ERROR ) } ) );

    defines->profiling_enable = test_input;

    CAPTURE(defines->profiling_enable);

    exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "defines_profiling_", defines.get());

    // Codegen should work or error as appropriate
    mu_assert("profiling_enabled define should have worked!",
              exitflag == expected_flag);
  }

  SECTION( "interrupt_enable" ) {
    OSQPInt test_input;
    OSQPInt expected_flag;
    std::tie( test_input, expected_flag ) =
        GENERATE( table<OSQPInt, OSQPInt>(
            { /* first is input, second is expected error */
              std::make_tuple( -1, OSQP_CODEGEN_DEFINES_ERROR ),
              std::make_tuple(  0, OSQP_NO_ERROR ),
              std::make_tuple(  1, OSQP_NO_ERROR ),
              std::make_tuple(  2, OSQP_CODEGEN_DEFINES_ERROR ),
              std::make_tuple(  3, OSQP_CODEGEN_DEFINES_ERROR ) } ) );

    defines->interrupt_enable = test_input;

    CAPTURE(defines->interrupt_enable);

    exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "defines_interrupts_", defines.get());

    // Codegen should work or error as appropriate
    mu_assert("interrupt_enable define should have worked!",
              exitflag == expected_flag);
  }

  SECTION( "codegen define: derivatives" ) {
    OSQPInt test_input;
    OSQPInt expected_flag;
    std::tie( test_input, expected_flag ) =
        GENERATE( table<OSQPInt, OSQPInt>(
            { /* first is input, second is expected error */
              std::make_tuple( -1, OSQP_CODEGEN_DEFINES_ERROR ),
              std::make_tuple(  0, OSQP_NO_ERROR ),
              std::make_tuple(  1, OSQP_NO_ERROR ),
              std::make_tuple(  2, OSQP_CODEGEN_DEFINES_ERROR ),
              std::make_tuple(  3, OSQP_CODEGEN_DEFINES_ERROR ) } ) );

    defines->derivatives_enable = test_input;

    exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "defines_derivatives_", defines.get());

    // Codegen should work or error as appropriate
    mu_assert("Non Convex codegen: derivative define should have worked!",
              exitflag == expected_flag);
  }
}

TEST_CASE("Codegen: Error propgatation", "[codegen]")
{
  OSQPInt exitflag;

  // Problem settings
  OSQPSettings_ptr settings{(OSQPSettings *)c_malloc(sizeof(OSQPSettings))};

  // Codegen defines
  OSQPCodegenDefines_ptr defines{(OSQPCodegenDefines *)c_malloc(sizeof(OSQPCodegenDefines))};

  // Structures
  OSQPSolver *tmpSolver = nullptr;
  OSQPSolver_ptr solver{nullptr};   // Wrap solver inside memory management

  // Problem data
  codegen_problem_ptr   data{generate_problem_codegen()};
  codegen_sols_data_ptr sols_data{generate_problem_codegen_sols_data()};

  // Define Solver settings as default
  osqp_set_default_settings(settings.get());
  settings->max_iter      = 2000;
  settings->alpha         = 1.6;
  settings->polishing     = 1;
  settings->scaling       = 0;
  settings->verbose       = 1;
  settings->warm_starting = 0;

  // Define codegen settings
  defines->embedded_mode = 1;      // vector update
  defines->float_type = 1;         // floats
  defines->printing_enable = 0;    // no printing
  defines->profiling_enable = 0;   // no timing
  defines->interrupt_enable = 0;   // no interrupts
  defines->derivatives_enable = 0; // no derivatives

  // Setup solver
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Setup error!", exitflag == 0);

  SECTION( "Missing linear system solver" ) {
    // Artificially delete the linsys solver
    void *tmpVar = solver->work->linsys_solver;
    solver->work->linsys_solver = NULL;

    exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "error_", defines.get());

    solver->work->linsys_solver = (LinSysSolver *) tmpVar;

    // Codegen should work
    mu_assert("Data error not handled!",
              exitflag == OSQP_WORKSPACE_NOT_INIT_ERROR);
  }

  SECTION( "Missing data struct" ) {
    // Artificially delete all the data
    void *tmpVar = solver->work->data;
    solver->work->data = NULL;

    exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "error_", defines.get());

    solver->work->data = (OSQPData *) tmpVar;

    // Codegen should work
    mu_assert("Data error not handled!",
              exitflag == OSQP_WORKSPACE_NOT_INIT_ERROR);
  }

  SECTION( "Missing float vector" ) {
    // Artificially delete a vector
    void *tmpVar = solver->work->data->l;
    solver->work->data->l = NULL;

    exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "error_", defines.get());

    solver->work->data->l = (OSQPVectorf *) tmpVar;

    // Codegen should work
    mu_assert("Missing vector not handled!",
              exitflag == OSQP_DATA_NOT_INITIALIZED);
  }

  SECTION( "Missing integer vector" ) {
    defines->embedded_mode = 2;

    // Artificially delete a vector
    void *tmpVar = solver->work->constr_type;
    solver->work->constr_type = NULL;

    exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "error_", defines.get());

    solver->work->constr_type = (OSQPVectori *) tmpVar;

    // Codegen should work
    mu_assert("Missing vector not handled!",
              exitflag == OSQP_DATA_NOT_INITIALIZED);
  }

  SECTION( "Missing matrix" ) {
    // Artificially delete a matrix
    void *tmpVar = solver->work->data->A;
    solver->work->data->A = NULL;

    exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "error_", defines.get());

    solver->work->data->A = (OSQPMatrix *) tmpVar;

    // Codegen should work
    mu_assert("Missing matrix not handled!",
              exitflag == OSQP_DATA_NOT_INITIALIZED);
  }
}

TEST_CASE("Codegen: Settings", "[codegen],[settings]")
{
  OSQPInt exitflag;

  // Problem settings
  OSQPSettings_ptr settings{(OSQPSettings *)c_malloc(sizeof(OSQPSettings))};

  // Codegen defines
  OSQPCodegenDefines_ptr defines{(OSQPCodegenDefines *)c_malloc(sizeof(OSQPCodegenDefines))};

  // Structures
  OSQPSolver *tmpSolver = nullptr;
  OSQPSolver_ptr solver{nullptr};   // Wrap solver inside memory management

  // Problem data
  codegen_problem_ptr   data{generate_problem_codegen()};
  codegen_sols_data_ptr sols_data{generate_problem_codegen_sols_data()};

  // Define Solver settings as default
  osqp_set_default_settings(settings.get());
  settings->max_iter      = 2000;
  settings->alpha         = 1.6;
  settings->polishing     = 1;
  settings->scaling       = 0;
  settings->verbose       = 1;
  settings->warm_starting = 0;

  // Define codegen settings
  defines->embedded_mode = 1;      // vector update
  defines->float_type = 1;         // floats
  defines->printing_enable = 0;    // no printing
  defines->profiling_enable = 0;   // no timing
  defines->interrupt_enable = 0;   // no interrupts
  defines->derivatives_enable = 0; // no derivatives

  // scaling changes some allocations (some vectors become null)
  SECTION( "Scaling setting" ) {
    // Test with both scaling=0 and scaling=1
    OSQPInt scaling  = GENERATE(0, 1);
    OSQPInt embedded;
    std::string dir;

    std::tie( embedded, dir ) =
      GENERATE( table<OSQPInt, std::string>(
          { /* first is embedded mode, second is output directory */
            std::make_tuple( 1, CODEGEN1_DIR ),
            std::make_tuple( 2, CODEGEN2_DIR ) } ) );

    char name[100];
    snprintf(name, 100, "scaling_%d_embedded_%d_", scaling, embedded);

    CAPTURE(embedded);

    settings->scaling = scaling;
    defines->embedded_mode = embedded;

    // Setup solver
    exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                          data->A, data->l, data->u,
                          data->m, data->n, settings.get());
    solver.reset(tmpSolver);

    // Setup correct
    mu_assert("Setup error!", exitflag == 0);

    exitflag = osqp_codegen(solver.get(), dir.c_str(), name, defines.get());

    // Codegen should work
    mu_assert("Scaling not handled properly!",
              exitflag == OSQP_NO_ERROR);
  }

  // rho_is_vec changes some allocations (some vectors become null)
  SECTION( "rho_is_vec setting" ) {
    // Test with both rho_is_vec=0 and rho_is_vec=1
    OSQPInt rho_is_vec = GENERATE(0, 1);
    OSQPInt embedded;
    std::string dir;

    std::tie( embedded, dir ) =
      GENERATE( table<OSQPInt, std::string>(
          { /* first is embedded mode, second is output directory */
            std::make_tuple( 1, CODEGEN1_DIR ),
            std::make_tuple( 2, CODEGEN2_DIR ) } ) );

    char name[100];
    snprintf(name, 100, "rho_is_vec_%d_embedded_%d_", rho_is_vec, embedded);

    CAPTURE(embedded);

    settings->rho_is_vec = rho_is_vec;
    defines->embedded_mode = embedded;

    // Setup solver
    exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                          data->A, data->l, data->u,
                          data->m, data->n, settings.get());
    solver.reset(tmpSolver);

    // Setup correct
    mu_assert("Setup error!", exitflag == 0);

    exitflag = osqp_codegen(solver.get(), dir.c_str(), name, defines.get());

    // Codegen should work
    mu_assert("rho_is_vec not handled properly!",
              exitflag == OSQP_NO_ERROR);
  }
}

#endif
