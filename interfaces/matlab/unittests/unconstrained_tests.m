classdef unconstrained_tests < matlab.unittest.TestCase
    %UNCONSTRAINED_TESTS Solve unconstrained quadratic program
    
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
            testCase.m = 0;
            testCase.P = sparse(diag(rand(testCase.n, 1))) + 0.2*speye(testCase.n);
            testCase.q = randn(testCase.n, 1);
            testCase.A = [];
            testCase.l = [];
            testCase.u = [];
            
            
            % Setup solver
            testCase.solver = osqp;
            testCase.solver.setup(testCase.P, testCase.q, ...
                testCase.A, testCase.l, testCase.u, 'verbose', 0, 'eps_abs', 1e-05, 'eps_rel', 1e-05);
            
            % Get options
            testCase.options = testCase.solver.current_settings();
            
            % Setup tolerance
            testCase.tol = 1e-04;
            
        end
    end
    
    methods (Test)
        function test_unconstrained_problem(testCase)
            % Solve with OSQP
            results = testCase.solver.solve();
            
            % Check if they are close
            testCase.verifyEqual(results.x, ...
                [-0.1139; -2.6464; -0.0414; 0.4539; 0.8058; -0.2232; -0.0399; -1.6256; 0.4702; ...
                 -0.6265; 0.3740; -0.6547; -0.4346; 0.9082; 0.8801; -0.0094; -2.3109; 0.5990; ...
                 -0.4948; -0.1263; -3.3029; 0.2563; -0.6106; 0.4830; 0.0081; 4.3573; -2.9165; ...
                 0.4514; -1.7058; 0.5228], 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.y, zeros(0,1), 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.info.obj_val, -16.3649, 'AbsTol', testCase.tol)
            
        end
        
    end
    
end
