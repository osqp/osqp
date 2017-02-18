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
        quadprog_opts
        tol
    end

    methods(TestMethodSetup)
        function setup_problem(testCase)
            % Create Problem
            testCase.P = sparse([11 0; 0, 0]);
            testCase.q = [3.; 4];
            testCase.A = sparse([-1. 0; 0 -1; -1 -3; 2  5; 3  4]);
            testCase.u = [0; 0; -15.; 100; 80];
            testCase.l = -1e20 * ones(length(testCase.u), 1); % quadprog doesn't like Inf
            testCase.n = size(testCase.P, 1);
            testCase.m = size(testCase.A, 1);

            % Setup solver
            testCase.solver = osqp;
            testCase.solver.setup(testCase.P, testCase.q, ...
                testCase.A, testCase.l, testCase.u, 'verbose', 0);

            % Get options
            testCase.options = testCase.solver.current_settings();

            % Setup quadprog options
            testCase.quadprog_opts.Display = 'off';

            % Setup tolerance
            testCase.tol = 1e-04;

        end
    end

    methods (Test)
        function test_basic_qp(testCase)
            % Solve with OSQP
            results = testCase.solver.solve();

            % Solve with quadprog
            [test_x, ~, ~] = quadprog(testCase.P, testCase.q, ...
                                      [testCase.A; -testCase.A], ...
                                      [testCase.u; -testCase.l], ...
                                      [],[],[],[],[], testCase.quadprog_opts);

            % Check if they are close
            testCase.verifyEqual(results.x, test_x, 'AbsTol',testCase.tol)

        end

        function test_update_q(testCase)
            % Update linear cost
            q_new = [10; 20];
            testCase.solver.update('q',q_new);

            % Solve again
            results = testCase.solver.solve();

            % Solve with quadprog
            [test_x, ~, ~] = quadprog(testCase.P, q_new, ...
                          [testCase.A; -testCase.A], ...
                          [testCase.u; -testCase.l], ...
                          [],[],[],[],[], testCase.quadprog_opts);

            % Check if they are close
            testCase.verifyEqual(results.x, test_x, 'AbsTol',testCase.tol)

        end

       function test_update_l(testCase)
            % Update lower bound
            l_new = -100 * ones(testCase.m, 1);
            testCase.solver.update('l',l_new);

            % Solve again
            results = testCase.solver.solve();

            % Solve with quadprog
            [test_x, ~, ~] = quadprog(testCase.P, testCase.q, ...
                          [testCase.A; -testCase.A], ...
                          [testCase.u; -l_new], ...
                          [],[],[],[],[], testCase.quadprog_opts);

            % Check if they are close
            testCase.verifyEqual(results.x, test_x, 'AbsTol',testCase.tol)

        end

        function test_update_u(testCase)
             % Update upper bound
             u_new = 100 * ones(testCase.m, 1);
             testCase.solver.update('u',u_new);

             % Solve again
             results = testCase.solver.solve();

             % Solve with quadprog
             [test_x, ~, ~] = quadprog(testCase.P, testCase.q, ...
                           [testCase.A; -testCase.A], ...
                           [u_new; -testCase.l], ...
                           [],[],[],[],[], testCase.quadprog_opts);

             % Check if they are close
             testCase.verifyEqual(results.x, test_x, 'AbsTol',testCase.tol)

         end

         function test_update_bounds(testCase)
              % Update bounds
              l_new = -100 * ones(testCase.m, 1);
              u_new = 100 * ones(testCase.m, 1);
              testCase.solver.update('l', l_new, 'u',u_new);

              % Solve again
              results = testCase.solver.solve();

              % Solve with quadprog
              [test_x, ~, ~] = quadprog(testCase.P, testCase.q, ...
                            [testCase.A; -testCase.A], ...
                            [u_new; -l_new], ...
                            [],[],[],[],[], testCase.quadprog_opts);

              % Check if they are close
              testCase.verifyEqual(results.x, test_x, 'AbsTol',testCase.tol)

          end


          function test_update_max_iter(testCase)
               % Update max_iter
               opts = testCase.solver.current_settings();
               opts.max_iter = 10;
               testCase.solver.update_settings(opts);

               % Solve again
               results = testCase.solver.solve();

               % Check if they are close
               testCase.verifyEqual(results.info.status_val, testCase.solver.constant('OSQP_MAX_ITER_REACHED'), 'AbsTol',testCase.tol)

           end
        % TODO: add ALL unittest functions as in the Python interface


    end

end
