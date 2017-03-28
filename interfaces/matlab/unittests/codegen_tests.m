classdef codegen_tests < matlab.unittest.TestCase
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
                testCase.A, testCase.l, testCase.u, 'verbose', 0, 'alpha', 1.6, ...
                'eps_abs', 1e-6, 'eps_rel', 1e-6, 'max_iter', 3000);
            
            % Generate code (supress command window output)
            clear mex
            cmd = 'testCase.solver.codegen(''code'', ''mexname'', ''emosqp'', ''force_rewrite'', true);';
            evalc(cmd);

            % Get options
            testCase.options = testCase.solver.current_settings();

            % Setup tolerance
            testCase.tol = 1e-05;

        end
    end

    methods (Test)
        function test_solve(testCase)
            % Solve with OSQP
            [x, y, ~, ~] = emosqp('solve');

            % Check if they are close
            testCase.verifyEqual(x, [0; 5], 'AbsTol',testCase.tol)
            testCase.verifyEqual(y, [1.666667; 0; 1.333333; 0; 0], 'AbsTol',testCase.tol)

        end

        function test_update_q(testCase)
            % Update linear cost
            q_new = [10; 20];
            emosqp('update_lin_cost', q_new);

            % Solve again
            [x, y, ~, ~] = emosqp('solve');

            % Check if they are close
            testCase.verifyEqual(x, [0.0; 5.0], 'AbsTol',testCase.tol)
            testCase.verifyEqual(y, [3.333333; 0; 6.666667; 0; 0], 'AbsTol',testCase.tol)
            
            % Update linear cost to the original value
            emosqp('update_lin_cost', testCase.q);

        end

        function test_update_l(testCase)
            % Update lower bound
            l_new = -100 * ones(testCase.m, 1);
            emosqp('update_lower_bound', l_new);

            % Solve again
            [x, y, ~, ~] = emosqp('solve');

            % Check if they are close
            testCase.verifyEqual(x, [0.0; 5.0], 'AbsTol',testCase.tol)
            testCase.verifyEqual(y, [1.666667; 0.0; 1.333333; 0.0; 0.0], 'AbsTol',testCase.tol)
            
            % Update lower bound to the original value
            emosqp('update_lower_bound', testCase.l);

        end

        function test_update_u(testCase)
            % Update upper bound
            u_new = 100 * ones(testCase.m, 1);
            emosqp('update_upper_bound', u_new);

            % Solve again
            [x, y, ~, ~] = emosqp('solve');

            % Check if they are close
            testCase.verifyEqual(x, [-0.151515; -33.282828], 'AbsTol',testCase.tol)
            testCase.verifyEqual(y, [0.0; 0.0; 1.333333; 0.0; 0.0], 'AbsTol',testCase.tol)

            % Update upper bound to the original value
            emosqp('update_upper_bound', testCase.u);
        end

        function test_update_bounds(testCase)
            % Update bounds
            l_new = -100 * ones(testCase.m, 1);
            u_new = 100 * ones(testCase.m, 1);
            emosqp('update_bounds', l_new, u_new);

            % Solve again
            [x, y, ~, ~] = emosqp('solve');

            % Check if they are close
            testCase.verifyEqual(x, [-0.127273; -19.949091], 'AbsTol',testCase.tol)
            testCase.verifyEqual(y, [0.0; 0.0; 0.0; -0.8; 0.0], 'AbsTol',testCase.tol)
            
            % Clean directory
            rmdir('code', 's');
            clear mex
            delete(['emosqp.', mexext]);
            
        end

    end

end
