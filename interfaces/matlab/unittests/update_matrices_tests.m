classdef update_matrices_tests < matlab.unittest.TestCase
    %TEST_BASIC_QP Solve Basic QP Problem

    properties
        P
        P_new
        q
        A
        A_new
        u
        l
        m
        n
        solver
        options
        tol
    end

    methods(TestMethodSetup)
        function setup_problem(testCase)
            rng(1)

            testCase.n = 5;
            testCase.m = 8;
            p = 0.7;

            Pt = sprandn(testCase.n, testCase.n, p);
            Pt_new = Pt;
            Pt_new(find(Pt)) = Pt(find(Pt)) + 0.1*randn(nnz(Pt), 1);

            % Create Problem
            testCase.P = Pt'*Pt + speye(testCase.n);
            testCase.P_new = Pt_new'*Pt_new + speye(testCase.n);
            testCase.q = randn(testCase.n, 1);
            testCase.A = sprandn(testCase.m, testCase.n, p);
            testCase.A_new = testCase.A;
            testCase.A_new(find(testCase.A)) = testCase.A(find(testCase.A)) + randn(nnz(testCase.A), 1);
            testCase.l = zeros(testCase.m, 1);
            testCase.u = 30 + randn(testCase.m, 1);

            % Setup solver
            testCase.solver = osqp;
            testCase.solver.setup(testCase.P, testCase.q, testCase.A, ...
                testCase.l, testCase.u, 'verbose', false, 'eps_rel', 1e-7, 'eps_abs', 1e-07, 'polish', true);

            % Setup tolerance
            testCase.tol = 1e-04;

        end
    end

    methods (Test)
        function test_solve(testCase)
            % Solve with OSQP
            results = testCase.solver.solve();

            % Check if they are close
            testCase.verifyEqual(results.x, ...
                [-0.0000; -0.0281; 0.2292; 0.0000; -0.0000], 'AbsTol',testCase.tol)
% Dual solution returned might be different
%             testCase.verifyEqual(results.y, ...
%                 [-1.5643; 0; -1.3562; -0.9056; -0.4113; 0; 0; 0], 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.info.obj_val, -0.0400, 'AbsTol', testCase.tol)

        end

        function test_update_P(testCase)
            % Update matrix P
            Pnew_triu = triu(testCase.P_new);
            Px = nonzeros(Pnew_triu);
            Px_idx = (1:nnz(Pnew_triu))';
            testCase.solver.update('Px', Px, 'Px_idx', Px_idx);

            % Solve again
            results = testCase.solver.solve();

            % Check if they are close
            testCase.verifyEqual(results.x, ...
                [-0.0000; -0.0261; 0.2129; 0.0000; -0.0000], 'AbsTol',testCase.tol)
% Dual solution returned might be different
%             testCase.verifyEqual(results.y, ...
%                 [-1.5616; 0; -1.3631; -0.8931; -0.4198; 0; 0; 0], 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.info.obj_val, -0.0372, 'AbsTol', testCase.tol)

        end
        
        function test_update_P_allind(testCase)
            % Update matrix P
            Pnew_triu = triu(testCase.P_new);
            Px = nonzeros(Pnew_triu);
            testCase.solver.update('Px', Px);

            % Solve again
            results = testCase.solver.solve();

            % Check if they are close
            testCase.verifyEqual(results.x, ...
                [-0.0000; -0.0261; 0.2129; 0.0000; -0.0000], 'AbsTol',testCase.tol)
% Dual solution returned might be different
%             testCase.verifyEqual(results.y, ...
%                 [-1.5616; 0; -1.3631; -0.8931; -0.4198; 0; 0; 0], 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.info.obj_val, -0.0372, 'AbsTol', testCase.tol)

        end

        function test_update_A(testCase)
            % Update matrix A
            Ax = nonzeros(testCase.A_new);
            Ax_idx = (1:nnz(testCase.A_new))';
            testCase.solver.update('Ax', Ax, 'Ax_idx', Ax_idx);

            % Solve again
            results = testCase.solver.solve();

            % Check if they are close
            testCase.verifyEqual(results.x, ...
                [-0.0398; -0.0761; -0.0292; -0.0000; -0.0199], 'AbsTol',testCase.tol)
% Dual solution returned might be different
%             testCase.verifyEqual(results.y, ...
%                 [-1.2329; 0; 0; 0; -0.3052; -0.4643; -0.0151; 0], 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.info.obj_val, -0.0093, 'AbsTol', testCase.tol)

        end
        
        function test_update_A_allind(testCase)
            % Update matrix A
            Ax = nonzeros(testCase.A_new);
            testCase.solver.update('Ax', Ax);

            % Solve again
            results = testCase.solver.solve();

            % Check if they are close
            testCase.verifyEqual(results.x, ...
                [-0.0398; -0.0761; -0.0292; -0.0000; -0.0199], 'AbsTol',testCase.tol)
% Dual solution returned might be different
%             testCase.verifyEqual(results.y, ...
%                 [-1.2329; 0; 0; 0; -0.3052; -0.4643; -0.0151; 0], 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.info.obj_val, -0.0093, 'AbsTol', testCase.tol)

        end
        
        function test_update_P_A_indP_indA(testCase)
            % Update matrices P and A
            Pnew_triu = triu(testCase.P_new);
            Px = nonzeros(Pnew_triu);
            Px_idx = (1:nnz(Pnew_triu))';
            Ax = nonzeros(testCase.A_new);
            Ax_idx = (1:nnz(testCase.A_new))';
            testCase.solver.update('Px', Px, 'Px_idx', Px_idx, 'Ax', Ax, 'Ax_idx', Ax_idx);

            % Solve again
            results = testCase.solver.solve();

            % Check if they are close
            testCase.verifyEqual(results.x, ...
                [-0.0439; -0.0840; -0.0322; 0.0000; -0.0219], 'AbsTol',testCase.tol)
% Dual solution returned might be different
%             testCase.verifyEqual(results.y, ...
%                 [-1.2386; 0; 0; 0; -0.3105; -0.4430; -0.0112; 0], 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.info.obj_val, -0.0103, 'AbsTol', testCase.tol)

        end
        
        function test_update_P_A_indP(testCase)
            % Update matrices P and A
            Pnew_triu = triu(testCase.P_new);
            Px = nonzeros(Pnew_triu);
            Px_idx = (1:nnz(Pnew_triu))';
            Ax = nonzeros(testCase.A_new);
            testCase.solver.update('Px', Px, 'Px_idx', Px_idx, 'Ax', Ax);

            % Solve again
            results = testCase.solver.solve();

            % Check if they are close
            testCase.verifyEqual(results.x, ...
                [-0.0439; -0.0840; -0.0322; 0.0000; -0.0219], 'AbsTol',testCase.tol)
% Dual solution returned might be different
%             testCase.verifyEqual(results.y, ...
%                 [-1.2386; 0; 0; 0; -0.3105; -0.4430; -0.0112; 0], 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.info.obj_val, -0.0103, 'AbsTol', testCase.tol)

        end
        
        function test_update_P_A_indA(testCase)
            % Update matrices P and A
            Pnew_triu = triu(testCase.P_new);
            Px = nonzeros(Pnew_triu);
            Ax = nonzeros(testCase.A_new);
            Ax_idx = (1:nnz(testCase.A_new))';
            testCase.solver.update('Px', Px, 'Ax', Ax, 'Ax_idx', Ax_idx);

            % Solve again
            results = testCase.solver.solve();

            % Check if they are close
            testCase.verifyEqual(results.x, ...
                [-0.0439; -0.0840; -0.0322; 0.0000; -0.0219], 'AbsTol',testCase.tol)
% Dual solution returned might be different
%             testCase.verifyEqual(results.y, ...
%                 [-1.2386; 0; 0; 0; -0.3105; -0.4430; -0.0112; 0], 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.info.obj_val, -0.0103, 'AbsTol', testCase.tol)

        end
        
        function test_update_P_A_allind(testCase)
            % Update matrices P and A
            Pnew_triu = triu(testCase.P_new);
            Px = nonzeros(Pnew_triu);
            Ax = nonzeros(testCase.A_new);
            testCase.solver.update('Px', Px, 'Ax', Ax);

            % Solve again
            results = testCase.solver.solve();

            % Check if they are close
            testCase.verifyEqual(results.x, ...
                [-0.0439; -0.0840; -0.0322; 0.0000; -0.0219], 'AbsTol',testCase.tol)
% Dual solution returned might be different
%             testCase.verifyEqual(results.y, ...
%                 [-1.2386; 0; 0; 0; -0.3105; -0.4430; -0.0112; 0], 'AbsTol',testCase.tol)
            testCase.verifyEqual(results.info.obj_val, -0.0103, 'AbsTol', testCase.tol)

        end
    
    end

end
