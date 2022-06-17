#include "osqp.h"    // OSQP API
#include "util.h"    // Utilities for testing
#include "osqp_tester.h" // Basic testing script header

#include "codegen/data.h"

#ifdef OSQP_CODEGEN
void test_codegen_basic()
{
  c_int exitflag;

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
  defines->embedded_mode = 1;    // vector update
  defines->float_type = 1;       // floats
  defines->printing_enable = 0;  // no printing
  defines->profiling_enable = 0; // no timing
  defines->interrupt_enable = 0; // no interrupts

  // Setup solver
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Codegen test: Setup error!", exitflag == 0);

  // Vector update
  defines->embedded_mode = 1;

  exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "basic_vector_", defines.get());

  mu_assert("Non Convex codegen: codegen type 1 should have worked!",
            exitflag == OSQP_NO_ERROR);

  // matrix update
  defines->embedded_mode = 2;

  exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "basic_matrix_", defines.get());

  mu_assert("Non Convex codegen: codegen type 2 should have worked!",
            exitflag == OSQP_NO_ERROR);
}

void test_codegen_defines()
{
  c_int exitflag;

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
  defines->embedded_mode = 1;    // vector update
  defines->float_type = 1;       // floats
  defines->printing_enable = 0;  // no printing
  defines->profiling_enable = 0; // no timing
  defines->interrupt_enable = 0; // no interrupts

  // Setup solver
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);

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

    exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "defines_embedded_", defines.get());

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

    exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "defines_float_", defines.get());

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

    exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "defines_printing_", defines.get());

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

    exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "defines_profiling_", defines.get());

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

    exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "defines_interrupts_", defines.get());

    // Codegen should work or error as appropriate
    mu_assert("Non Convex codegen: interrupt define should have worked!",
              exitflag == expected_flag);
  }
}

void test_codegen_error_propagation()
{
  c_int exitflag;

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
  defines->embedded_mode = 1;    // vector update
  defines->float_type = 1;       // floats
  defines->printing_enable = 0;  // no printing
  defines->profiling_enable = 0; // no timing
  defines->interrupt_enable = 0; // no interrupts

  // Setup solver
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Codegen test: Setup error!", exitflag == 0);

  SECTION( "codegen: missing linear system solver" ) {
    // Artificially delete the linsys solver
    void *tmpVar = solver->work->linsys_solver;
    solver->work->linsys_solver = NULL;

    exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "error_", defines.get());

    solver->work->linsys_solver = (LinSysSolver *) tmpVar;

    // Codegen should work
    mu_assert("codegen: missing linear system solver not handled!",
              exitflag == OSQP_WORKSPACE_NOT_INIT_ERROR);
  }

  SECTION( "codegen: missing data" ) {
    // Artificially delete all the data
    void *tmpVar = solver->work->data;
    solver->work->data = NULL;

    exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "error_", defines.get());

    solver->work->data = (OSQPData *) tmpVar;

    // Codegen should work
    mu_assert("codegen: missing data not handled!",
              exitflag == OSQP_WORKSPACE_NOT_INIT_ERROR);
  }

  SECTION( "codegen: missing float vector" ) {
    // Artificially delete a vector
    void *tmpVar = solver->work->data->l;
    solver->work->data->l = NULL;

    exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "error_", defines.get());

    solver->work->data->l = (OSQPVectorf *) tmpVar;

    // Codegen should work
    mu_assert("codegen: missing codegen float vector not handled!",
              exitflag == OSQP_DATA_NOT_INITIALIZED);
  }

  SECTION( "codegen: missing integer vector" ) {
    defines->embedded_mode = 2;

    // Artificially delete a vector
    void *tmpVar = solver->work->constr_type;
    solver->work->constr_type = NULL;

    exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "error_", defines.get());

    solver->work->constr_type = (OSQPVectori *) tmpVar;

    // Codegen should work
    mu_assert("codegen: missing codegen integer vector not handled!",
              exitflag == OSQP_DATA_NOT_INITIALIZED);
  }

  SECTION( "codegen: missing matrix" ) {
    // Artificially delete a matrix
    void *tmpVar = solver->work->data->A;
    solver->work->data->A = NULL;

    exitflag = osqp_codegen(solver.get(), CODEGEN_DIR, "error_", defines.get());

    solver->work->data->A = (OSQPMatrix *) tmpVar;

    // Codegen should work
    mu_assert("codegen: missing codegen matrix not handled!",
              exitflag == OSQP_DATA_NOT_INITIALIZED);
  }
}

void test_codegen_settings()
{
  c_int exitflag;

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
  defines->embedded_mode = 1;    // vector update
  defines->float_type = 1;       // floats
  defines->printing_enable = 0;  // no printing
  defines->profiling_enable = 0; // no timing
  defines->interrupt_enable = 0; // no interrupts

  // scaling changes some allocations (some vectors become null)
  SECTION( "codegen: scaling setting" ) {
    // Test with both scaling=0 and scaling=1
    c_int scaling  = GENERATE(0, 1);
    c_int embedded;
    std::string dir;

    std::tie( embedded, dir ) =
      GENERATE( table<c_int, std::string>(
          { /* first is embedded mode, second is output directory */
            std::make_tuple( 1, CODEGEN1_DIR ),
            std::make_tuple( 2, CODEGEN2_DIR ) } ) );

    char name[100];
    snprintf(name, 100, "scaling_%d_embedded_%d_", scaling, embedded);

    settings->scaling = scaling;
    defines->embedded_mode = embedded;

    // Setup solver
    exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                          data->A, data->l, data->u,
                          data->m, data->n, settings.get());
    solver.reset(tmpSolver);

    // Setup correct
    mu_assert("Codegen test: Setup error!", exitflag == 0);

    exitflag = osqp_codegen(solver.get(), dir.c_str(), name, defines.get());

    // Codegen should work
    mu_assert("codegen: Scaling not handled properly!",
              exitflag == OSQP_NO_ERROR);
  }

  // rho_is_vec changes some allocations (some vectors become null)
  SECTION( "codegen: rho_is_vec setting" ) {
    // Test with both rho_is_vec=0 and rho_is_vec=1
    c_int rho_is_vec = GENERATE(0, 1);
    c_int embedded;
    std::string dir;

    std::tie( embedded, dir ) =
      GENERATE( table<c_int, std::string>(
          { /* first is embedded mode, second is output directory */
            std::make_tuple( 1, CODEGEN1_DIR ),
            std::make_tuple( 2, CODEGEN2_DIR ) } ) );

    char name[100];
    snprintf(name, 100, "rho_is_vec_%d_embedded_%d_", rho_is_vec, embedded);

    settings->rho_is_vec = rho_is_vec;
    defines->embedded_mode = embedded;

    // Setup solver
    exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                          data->A, data->l, data->u,
                          data->m, data->n, settings.get());
    solver.reset(tmpSolver);

    // Setup correct
    mu_assert("Codegen test: Setup error!", exitflag == 0);

    exitflag = osqp_codegen(solver.get(), dir.c_str(), name, defines.get());

    // Codegen should work
    mu_assert("codegen: rho_is_vec not handled properly!",
              exitflag == OSQP_NO_ERROR);
  }
}
#endif
