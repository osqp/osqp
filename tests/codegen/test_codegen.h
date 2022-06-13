#include "osqp.h"    // OSQP API
#include "util.h"    // Utilities for testing
#include "osqp_tester.h" // Basic testing script header

#include "codegen/data.h"

#ifdef OSQP_CODEGEN
void test_codegen_basic()
{
  c_int exitflag;

  // Problem settings
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

  // Codegen defines
  OSQPCodegenDefines *defines = (OSQPCodegenDefines *)c_malloc(sizeof(OSQPCodegenDefines));

  // Structures
  OSQPSolver   *solver; // Solver
  OSQPTestData *data;      // Data
  codegen_sols_data *sols_data;

  // Populate data
  data = generate_problem_codegen();
  sols_data = generate_problem_codegen_sols_data();

  // Define Solver settings as default
  osqp_set_default_settings(settings);
  settings->max_iter      = 2000;
  settings->alpha         = 1.6;
  settings->polishing     = 1;
  settings->scaling       = 0;
  settings->verbose       = 1;
  settings->warm_starting = 0;

  // Define codegen settings
  defines->embedded_mode = 1;    // vector update
  defines->float_type = 1;       // floats
  defines->printing_enable = 0;  // no printing
  defines->profiling_enable = 0; // no timing
  defines->interrupt_enable = 0; // no interrupts

  // Setup solver
  exitflag = osqp_setup(&solver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings);

  // Setup correct
  mu_assert("Codegen test: Setup error!", exitflag == 0);

  exitflag = osqp_codegen(solver, CODEGEN_DIR, "basic_", defines);

  // Codegen should work
  mu_assert("Non Convex codegen: codegen type 1 should have worked!",
            exitflag == OSQP_NO_ERROR);

  defines->embedded_mode = 2;    // matrix update

  exitflag = osqp_codegen(solver, CODEGEN_DIR, "basic_", defines);

  // Codegen should work
  mu_assert("Non Convex codegen: codegen type 2 should have worked!",
            exitflag == OSQP_NO_ERROR);

  // Cleanup data
  clean_problem_codegen(data);
  clean_problem_codegen_sols_data(sols_data);

  // Cleanup
  c_free(settings);
  c_free(defines);
}

void test_codegen_defines()
{
  c_int exitflag;

  // Problem settings
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

  // Codegen defines
  OSQPCodegenDefines *defines = (OSQPCodegenDefines *)c_malloc(sizeof(OSQPCodegenDefines));

  // Structures
  OSQPSolver   *solver; // Solver
  OSQPTestData *data;      // Data
  codegen_sols_data *sols_data;

  // Populate data
  data = generate_problem_codegen();
  sols_data = generate_problem_codegen_sols_data();

  // Define Solver settings as default
  osqp_set_default_settings(settings);
  settings->max_iter      = 2000;
  settings->alpha         = 1.6;
  settings->polishing     = 1;
  settings->scaling       = 0;
  settings->verbose       = 1;
  settings->warm_starting = 0;

  // Define codegen settings
  defines->embedded_mode = 1;    // vector update
  defines->float_type = 1;       // floats
  defines->printing_enable = 0;  // no printing
  defines->profiling_enable = 0; // no timing
  defines->interrupt_enable = 0; // no interrupts

  // Setup solver
  exitflag = osqp_setup(&solver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings);

  // Setup correct
  mu_assert("Codegen test: Setup error!", exitflag == 0);

  SECTION( "codegen define: embedded_mode" ) {
    c_int test_input;
    c_int expected_flag;
    std::tie( test_input, expected_flag ) =
        GENERATE( table<c_int, c_int>(
            { /* first is input, second is expected error */
              std::make_tuple( 0, OSQP_CODEGEN_DEFINES_ERROR ),
              std::make_tuple( 1, OSQP_NO_ERROR ),
              std::make_tuple( 2, OSQP_NO_ERROR ),
              std::make_tuple( 3, OSQP_CODEGEN_DEFINES_ERROR ) } ) );

    defines->embedded_mode = test_input;

    exitflag = osqp_codegen(solver, CODEGEN_DIR, "basic_", defines);

    // Codegen should work or error as appropriate
    mu_assert("Non Convex codegen: embedded define should have worked!",
              exitflag == expected_flag);
  }

  SECTION( "codegen define: floats" ) {
    c_int test_input;
    c_int expected_flag;
    std::tie( test_input, expected_flag ) =
        GENERATE( table<c_int, c_int>(
            { /* first is input, second is expected error */
              std::make_tuple( -1, OSQP_CODEGEN_DEFINES_ERROR ),
              std::make_tuple(  0, OSQP_NO_ERROR ),
              std::make_tuple(  1, OSQP_NO_ERROR ),
              std::make_tuple(  2, OSQP_CODEGEN_DEFINES_ERROR ),
              std::make_tuple(  3, OSQP_CODEGEN_DEFINES_ERROR ) } ) );

    defines->float_type = test_input;

    exitflag = osqp_codegen(solver, CODEGEN_DIR, "basic_", defines);

    // Codegen should work or error as appropriate
    mu_assert("Non Convex codegen: float define should have worked!",
              exitflag == expected_flag);
  }

  SECTION( "codegen define: printing" ) {
    c_int test_input;
    c_int expected_flag;
    std::tie( test_input, expected_flag ) =
        GENERATE( table<c_int, c_int>(
            { /* first is input, second is expected error */
              std::make_tuple( -1, OSQP_CODEGEN_DEFINES_ERROR ),
              std::make_tuple(  0, OSQP_NO_ERROR ),
              std::make_tuple(  1, OSQP_NO_ERROR ),
              std::make_tuple(  2, OSQP_CODEGEN_DEFINES_ERROR ),
              std::make_tuple(  3, OSQP_CODEGEN_DEFINES_ERROR ) } ) );

    defines->printing_enable = test_input;

    exitflag = osqp_codegen(solver, CODEGEN_DIR, "basic_", defines);

    // Codegen should work or error as appropriate
    mu_assert("Non Convex codegen: printing define should have worked!",
              exitflag == expected_flag);
  }

  SECTION( "codegen define: profiling" ) {
    c_int test_input;
    c_int expected_flag;
    std::tie( test_input, expected_flag ) =
        GENERATE( table<c_int, c_int>(
            { /* first is input, second is expected error */
              std::make_tuple( -1, OSQP_CODEGEN_DEFINES_ERROR ),
              std::make_tuple(  0, OSQP_NO_ERROR ),
              std::make_tuple(  1, OSQP_NO_ERROR ),
              std::make_tuple(  2, OSQP_CODEGEN_DEFINES_ERROR ),
              std::make_tuple(  3, OSQP_CODEGEN_DEFINES_ERROR ) } ) );

    defines->profiling_enable = test_input;

    exitflag = osqp_codegen(solver, CODEGEN_DIR, "basic_", defines);

    // Codegen should work or error as appropriate
    mu_assert("Non Convex codegen: profiling define should have worked!",
              exitflag == expected_flag);
  }

  SECTION( "codegen define: interrupts" ) {
    c_int test_input;
    c_int expected_flag;
    std::tie( test_input, expected_flag ) =
        GENERATE( table<c_int, c_int>(
            { /* first is input, second is expected error */
              std::make_tuple( -1, OSQP_CODEGEN_DEFINES_ERROR ),
              std::make_tuple(  0, OSQP_NO_ERROR ),
              std::make_tuple(  1, OSQP_NO_ERROR ),
              std::make_tuple(  2, OSQP_CODEGEN_DEFINES_ERROR ),
              std::make_tuple(  3, OSQP_CODEGEN_DEFINES_ERROR ) } ) );

    defines->interrupt_enable = test_input;

    exitflag = osqp_codegen(solver, CODEGEN_DIR, "basic_", defines);

    // Codegen should work or error as appropriate
    mu_assert("Non Convex codegen: interrupt define should have worked!",
              exitflag == expected_flag);
  }

  // Cleanup data
  clean_problem_codegen(data);
  clean_problem_codegen_sols_data(sols_data);

  // Cleanup
  c_free(settings);
  c_free(defines);
}

void test_codegen_error_propagation()
{
  c_int exitflag;

  // Problem settings
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

  // Codegen defines
  OSQPCodegenDefines *defines = (OSQPCodegenDefines *)c_malloc(sizeof(OSQPCodegenDefines));

  // Structures
  OSQPSolver   *solver; // Solver
  OSQPTestData *data;      // Data
  codegen_sols_data *sols_data;

  // Populate data
  data = generate_problem_codegen();
  sols_data = generate_problem_codegen_sols_data();

  // Define Solver settings as default
  osqp_set_default_settings(settings);
  settings->max_iter      = 2000;
  settings->alpha         = 1.6;
  settings->polishing     = 1;
  settings->scaling       = 0;
  settings->verbose       = 1;
  settings->warm_starting = 0;

  // Define codegen settings
  defines->embedded_mode = 1;    // vector update
  defines->float_type = 1;       // floats
  defines->printing_enable = 0;  // no printing
  defines->profiling_enable = 0; // no timing
  defines->interrupt_enable = 0; // no interrupts

  // Setup solver
  exitflag = osqp_setup(&solver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings);

  // Setup correct
  mu_assert("Codegen test: Setup error!", exitflag == 0);

  SECTION( "codegen: missing linear system solver" ) {
    // Artificially delete a vector
    solver->work->linsys_solver = NULL;

    exitflag = osqp_codegen(solver, CODEGEN_DIR, "error_", defines);

    // Codegen should work
    mu_assert("codegen: missing linear system solver not handled!",
              exitflag == OSQP_WORKSPACE_NOT_INIT_ERROR);
  }

  SECTION( "codegen: missing data" ) {
    // Artificially delete a vector
    solver->work->data = NULL;

    exitflag = osqp_codegen(solver, CODEGEN_DIR, "error_", defines);

    // Codegen should work
    mu_assert("codegen: missing data not handled!",
              exitflag == OSQP_WORKSPACE_NOT_INIT_ERROR);
  }

  SECTION( "codegen: missing float vector" ) {
    // Artificially delete a vector
    solver->work->data->l = NULL;

    exitflag = osqp_codegen(solver, CODEGEN_DIR, "error_", defines);

    // Codegen should work
    mu_assert("codegen: missing codegen float vector not handled!",
              exitflag == OSQP_DATA_NOT_INITIALIZED);
  }

  SECTION( "codegen: missing integer vector" ) {
    defines->embedded_mode = 2;

    // Artificially delete a vector
    solver->work->constr_type = NULL;

    exitflag = osqp_codegen(solver, CODEGEN_DIR, "error_", defines);

    // Codegen should work
    mu_assert("codegen: missing codegen integer vector not handled!",
              exitflag == OSQP_DATA_NOT_INITIALIZED);
  }

  SECTION( "codegen: missing matrix" ) {
    // Artificially delete a matrix
    solver->work->data->A = NULL;

    exitflag = osqp_codegen(solver, CODEGEN_DIR, "error_", defines);

    // Codegen should work
    mu_assert("codegen: missing codegen matrix not handled!",
              exitflag == OSQP_DATA_NOT_INITIALIZED);
  }

  // Cleanup
  c_free(settings);
  c_free(defines);
}
#endif
