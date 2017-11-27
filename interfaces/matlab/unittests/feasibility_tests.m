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
            testCase.solver.setup(testCase.P, testCase.q, testCase.A, testCase.l, testCase.u, ...
                'rho', 1e-01, 'verbose', 0, 'eps_abs', 1e-05, 'eps_rel', 1e-05);

            % Get options
            testCase.options = testCase.solver.current_settings();

            % Setup tolerance
            testCase.tol = 1e-04;

        end
    end

    methods (Test)
        function test_feasibility_problem(testCase)
            % Solve with OSQP
            results = testCase.solver.solve();

            % Check if they are close
            x_test = [-4.7063; -8.3459; -8.9925; -15.2607; -4.5422; 18.0450; -1.1234; 1.4756; ...
                 -6.4514; 3.8592; -1.3098; 2.2815; 2.2068; 11.8055; -6.0677; 0.8960; -5.9434; ...
                 -34.0620; 18.4405; -24.3205; 4.4200; -4.9292; -2.2414; -0.2506; 30.2891; ...
                 0.7295; 4.5628; -23.1693; 3.6001; -9.6683];
             
            testCase.verifyEqual(results.x, x_test, 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.y, zeros(testCase.m, 1), 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.info.obj_val, 0.0, 'AbsTol', testCase.tol)

        end

    end

end
