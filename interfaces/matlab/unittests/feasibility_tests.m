classdef feasibility_tests < matlab.unittest.TestCase
    %FEASIBILITY_TESTS Solve equality constrained feasibility problem

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
        linprog_opts
        tol
    end

    methods(TestMethodSetup)
        function setup_problem(testCase)
            % Create Problem
            rng(4)
            testCase.n = 30;
            testCase.m = 30;
            testCase.P = sparse(testCase.n, testCase.n);
            testCase.q = zeros(testCase.n, 1);
            testCase.A = sprandn(testCase.m, testCase.n, 0.8);
            testCase.u = randn(testCase.m, 1);
            testCase.l = testCase.u;


            % Setup solver
            testCase.solver = osqp;
            testCase.solver.setup(testCase.P, testCase.q, ...
                testCase.A, testCase.l, testCase.u, 'verbose', 0, 'eps_abs', 1e-05, 'eps_rel', 1e-05);

            % Get options
            testCase.options = testCase.solver.current_settings();

            % Setup linprog options
            testCase.linprog_opts.Display = 'off';

            % Setup tolerance
            testCase.tol = 1e-04;

        end
    end

    methods (Test)
        function test_feasibility_problem(testCase)
            % Solve with OSQP
            results = testCase.solver.solve();

            % Solve with quadprog
            warning('off')
            [test_x, ~, ~] = linprog(testCase.q, ...
                                     [],[], ...
                                     testCase.A,testCase.u, ...
                                     [],[],[], testCase.linprog_opts);
            warning('on')

            % Check if they are close
            testCase.verifyEqual(results.x, test_x, 'AbsTol', testCase.tol)

        end

    end

end
