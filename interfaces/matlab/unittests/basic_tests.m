classdef basic_tests < matlab.unittest.TestCase
    %TEST_BASIC_QP Solve Basic QP Problem

    properties
        P
        q
        A
        u
        l
        m,
        n
        solver
        options
        tol
    end

    methods(TestMethodSetup)
        function setup_problem(testCase)
            % Create Problem
            testCase.P = sparse([11 0; 0, 0]);
            testCase.q = [3; 4];
            testCase.A = sparse([-1. 0; 0 -1; -1 -3; 2  5; 3  4]);
            testCase.u = [0; 0; -15.; 100; 80];
            testCase.l = -1e20 * ones(length(testCase.u), 1);
            testCase.n = size(testCase.P, 1);
            testCase.m = size(testCase.A, 1);

            % Setup solver
            testCase.solver = osqp;
            testCase.solver.setup(testCase.P, testCase.q, ...
                testCase.A, testCase.l, testCase.u, 'verbose', 0);

            % Get options
            testCase.options = testCase.solver.current_settings();

            % Setup tolerance
            testCase.tol = 1e-04;

        end
    end

    methods (Test)
        function test_basic_qp(testCase)
            % Solve with OSQP
            results = testCase.solver.solve();

            % Check if they are close
            testCase.verifyEqual(results.x, [0.0; 5.0], 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.y, [1.6667; 0.0; 1.3333; 0.0; 0.0], 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.info.obj_val, 20.0, 'AbsTol', testCase.tol)

        end

        function test_update_q(testCase)
            % Update linear cost
            q_new = [10; 20];
            testCase.solver.update('q',q_new);

            % Solve again
            results = testCase.solver.solve();

            % Check if they are close
            testCase.verifyEqual(results.x, [0.0; 5.0], 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.y, [3.3333; 0.0; 6.6667; 0.0; 0.0], 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.info.obj_val, 100.0, 'AbsTol', testCase.tol)

        end

        function test_update_l(testCase)
            % Update lower bound
            l_new = -100 * ones(testCase.m, 1);
            testCase.solver.update('l',l_new);

            % Solve again
            results = testCase.solver.solve();

            % Check if they are close
            testCase.verifyEqual(results.x, [0.0; 5.0], 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.y, [1.6667; 0.0; 1.3333; 0.0; 0.0], 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.info.obj_val, 20.0, 'AbsTol', testCase.tol)

        end

        function test_update_u(testCase)
            % Update upper bound
            u_new = 100 * ones(testCase.m, 1);
            testCase.solver.update('u',u_new);

            % Solve again
            results = testCase.solver.solve();

            % Check if they are close
            testCase.verifyEqual(results.x, [-0.1515; -33.2828], 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.y, [0.0; 0.0; 1.3333; 0.0; 0.0], 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.info.obj_val, -133.4596, 'AbsTol', testCase.tol)

        end

        function test_update_bounds(testCase)
            % Update bounds
            l_new = -100 * ones(testCase.m, 1);
            u_new = 100 * ones(testCase.m, 1);
            testCase.solver.update('l', l_new, 'u',u_new);

            % Solve again
            results = testCase.solver.solve();

            % Check if they are close
            testCase.verifyEqual(results.x, [-0.1273; -19.9491], 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.y, [0.0; 0.0; 0.0; -0.8; 0.0], 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.info.obj_val, -80.0891, 'AbsTol', testCase.tol)

        end


        function test_update_max_iter(testCase)
            % Update max_iter
            opts = testCase.solver.current_settings();
            opts.max_iter = 30;
            testCase.solver.update_settings(opts);

            % Solve again
            results = testCase.solver.solve();

            % Check if they are close
            testCase.verifyEqual(results.info.status_val, ...
                testCase.solver.constant('OSQP_MAX_ITER_REACHED'), 'AbsTol',testCase.tol)

        end
        
        function test_update_early_termination(testCase)
            % Update max_iter
            opts = testCase.solver.current_settings();
            opts.early_terminate = 0;
            testCase.solver.update_settings(opts);

            % Solve again
            results = testCase.solver.solve();

            % Check if they are close
            testCase.verifyEqual(results.info.iter, testCase.options.max_iter, 'AbsTol',testCase.tol)

        end

    end

end
