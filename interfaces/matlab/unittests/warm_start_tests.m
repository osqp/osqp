classdef warm_start_tests < matlab.unittest.TestCase
    %WARM_START_TESTS Warm Start problems solution

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

            % Set options
            testCase.options = struct;
            testCase.options.verbose = 0;
            testCase.options.rho = 0.01;
            testCase.options.eps_abs = 1e-04;
            testCase.options.eps_rel = 1e-04;
            testCase.options.adaptive_rho = 0;
            testCase.options.check_termination = 1;

            % Setup tolerance
            testCase.tol = 1e-04;

        end
    end

    methods (Test)
        function test_warm_start(testCase)

            % big example
            rng(4)
            testCase.n = 100;
            testCase.m = 200;
            Pt = sprandn(testCase.n, testCase.n, 0.6);
            testCase.P = Pt' * Pt;
            testCase.q = randn(testCase.n, 1);
            testCase.A = sprandn(testCase.m, testCase.n, 0.8);
            testCase.u = 2*rand(testCase.m, 1);
            testCase.l = -2*rand(testCase.m, 1);

            % Setup solver
            testCase.solver = osqp;
            testCase.solver.setup(testCase.P, testCase.q, ...
                testCase.A, testCase.l, testCase.u, testCase.options);

            % Solve with OSQP
            results = testCase.solver.solve();

            % Store optimal values
            x_opt = results.x;
            y_opt = results.y;
            tot_iter = results.info.iter;

            % Warm start with zeros and check if the number of iterations is the same
            testCase.solver.warm_start('x', zeros(testCase.n, 1), 'y', zeros(testCase.m, 1));
            results = testCase.solver.solve();
            testCase.verifyEqual(results.info.iter, tot_iter, 'AbsTol', testCase.tol)

            % Warm start with optimal values and check that number of iterations is < 10
            testCase.solver.warm_start('x', x_opt, 'y', y_opt);
            results = testCase.solver.solve();
            testCase.verifyThat(results.info.iter, matlab.unittest.constraints.IsLessThan(10));

        end

    end

end
