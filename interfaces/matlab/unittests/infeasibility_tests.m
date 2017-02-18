classdef infeasibility_tests < matlab.unittest.TestCase
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
        quadprog_opts
        tol
    end

    methods(TestMethodSetup)
        function setup_problem(testCase)
  
            % Setup solver options
            testCase.options = struct;
            testCase.options.verbose = 0;
            testCase.options.rho = 0.01;
            testCase.options.eps_inf = 1e-05;
            testCase.options.max_iter = 2500;



            % Setup quadprog options
            testCase.quadprog_opts.Display = 'off';
            % testCase.quadprog_opts.Algorithm = 'dual-simplex';

            % Setup tolerance
            testCase.tol = 1e-04;

        end
    end

    methods (Test)
        function test_infeasibility_problem(testCase)
            
            % Create Problem
            rng(4)
            testCase.n = 50;
            testCase.m = 500;
            Pt = sprandn(testCase.n, testCase.n, 0.6);
            testCase.P = Pt' * Pt;
            testCase.q = randn(testCase.n, 1);
            testCase.A = sprandn(testCase.m, testCase.n, 0.8);
            testCase.u = 3 + randn(testCase.m, 1);
            testCase.l = -3 + randn(testCase.m, 1);

            % Make random problem infeasible
            nhalf = floor(testCase.n/2);
            testCase.A(nhalf, :) = ...
                testCase.A(nhalf + 1, :);
            testCase.l(nhalf) = testCase.u(nhalf + 1) + 10 * rand();
            testCase.u(nhalf) = testCase.l(nhalf) + 0.5;

            
            % Setup solver
            testCase.solver = osqp;
            testCase.solver.setup(testCase.P, testCase.q, ...
                testCase.A, testCase.l, testCase.u, testCase.options);
            
            % Solve with OSQP
            results = testCase.solver.solve();

            % Solve with quadprog to double-check that it is infeasible
            % [~, ~, ~] = quadprog(testCase.P, testCase.q, ...
            %               [testCase.A; -testCase.A], ...
            %               [testCase.u; -testCase.l], ...
            %               [],[],[],[],[], testCase.quadprog_opts);

            % Check if they are close
            testCase.verifyEqual(results.info.status_val, ...
                                 testCase.solver.constant('OSQP_INFEASIBLE'), ...
                                 'AbsTol', testCase.tol)

        end
        
        
        function test_unbounded_and_infeasible(testCase)
            testCase.n = 2;
            testCase.m = 4;
            testCase.P = sparse(2,2);
            testCase.q = [-1;-1];
            testCase.A = sparse([1 -1; -1 1;1 0;0 1]);
            testCase.l = [1; 1; 0; 0];
            testCase.u = Inf(testCase.m,1);

            % Setup solver
            testCase.solver = osqp;
            testCase.solver.setup(testCase.P, testCase.q, ...
                testCase.A, testCase.l, testCase.u, testCase.options);
            
            % Solve with OSQP
            results = testCase.solver.solve();

            % Check if they are close
            testCase.verifyEqual(results.info.status_val, ...
                                 testCase.solver.constant('OSQP_INFEASIBLE'), ...
                                 'AbsTol', testCase.tol)

        end

    end

end
