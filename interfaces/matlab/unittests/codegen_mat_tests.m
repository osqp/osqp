classdef codegen_mat_tests < matlab.unittest.TestCase
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
            testCase.P = sparse([11 0; 0, 0.1]);
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
            cmd = ['testCase.solver.codegen(''code'', ''mexname'', ''emosqp'',', ...
                  ' ''parameters'', ''matrices'', ''force_rewrite'', true);'];
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
            [x, y, ~, ~, ~] = emosqp('solve');

            % Check if they are close
            testCase.verifyEqual(x, [0; 5], 'AbsTol',testCase.tol)
            testCase.verifyEqual(y, [1.5; 0; 1.5; 0; 0], 'AbsTol',testCase.tol)

        end
        
        function test_update_P(testCase)
            % Update matrix P
            Pnew = speye(2);
            Pnew_triu = triu(Pnew);
            [~,~,Px] = find(Pnew_triu(:));
            Px_idx = (1:length(Px))';
            emosqp('update_P', Px, Px_idx, length(Px));
            
            % Solve with OSQP
            [x, y, ~, ~, ~] = emosqp('solve');

            % Check if they are close
            testCase.verifyEqual(x, [0; 5], 'AbsTol',testCase.tol)
            testCase.verifyEqual(y, [0; 0; 3; 0; 0], 'AbsTol',testCase.tol)
            
            % Update matrix P to the original value
            P_triu = triu(testCase.P);
            [~,~,Px] = find(P_triu(:));
            emosqp('update_P', Px, [], length(Px));

        end
        
        function test_update_P_allind(testCase)
            % Update matrix P
            Pnew = speye(2);
            Pnew_triu = triu(Pnew);
            [~,~,Px] = find(Pnew_triu(:));
            emosqp('update_P', Px, [], length(Px));
            
            % Solve with OSQP
            [x, y, ~, ~, ~] = emosqp('solve');

            % Check if they are close
            testCase.verifyEqual(x, [0; 5], 'AbsTol',testCase.tol)
            testCase.verifyEqual(y, [0; 0; 3; 0; 0], 'AbsTol',testCase.tol)
            
            % Update matrix P to the original value
            P_triu = triu(testCase.P);
            [~,~,Px] = find(P_triu(:));
            emosqp('update_P', Px, [], length(Px));

        end

        function test_update_A(testCase)
            % Update matrix A
            Anew = sparse([-1. 0; 0 -1; -2 -2; 2  5; 3  4]);
            [~,~,Ax] = find(Anew(:));
            Ax_idx = (1:length(Ax))';
            emosqp('update_A', Ax, Ax_idx, length(Ax));
            
            % Solve with OSQP
            [x, y, ~, ~, ~] = emosqp('solve');

            % Check if they are close
            testCase.verifyEqual(x, [0.15765766; 7.34234234], 'AbsTol',testCase.tol)
            testCase.verifyEqual(y, [0; 0; 2.36711712; 0; 0], 'AbsTol',testCase.tol)
            
            % Update matrix P to the original value
            [~,~,Ax] = find(testCase.A(:));
            emosqp('update_A', Ax, [], length(Ax));

        end
        
        function test_update_A_allind(testCase)
            % Update matrix A
            Anew = sparse([-1. 0; 0 -1; -2 -2; 2  5; 3  4]);
            [~,~,Ax] = find(Anew(:));
            emosqp('update_A', Ax, [], length(Ax));
            
            % Solve with OSQP
            [x, y, ~, ~, ~] = emosqp('solve');

            % Check if they are close
            testCase.verifyEqual(x, [0.15765766; 7.34234234], 'AbsTol',testCase.tol)
            testCase.verifyEqual(y, [0; 0; 2.36711712; 0; 0], 'AbsTol',testCase.tol)
            
            % Update matrix P to the original value
            [~,~,Ax] = find(testCase.A(:));
            emosqp('update_A', Ax, [], length(Ax));

        end
        
        function test_update_P_A_indP_indA(testCase)
            % Update matrix P
            Pnew = speye(2);
            Pnew_triu = triu(Pnew);
            [~,~,Px] = find(Pnew_triu(:));
            Px_idx = (1:length(Px))';
            Anew = sparse([-1. 0; 0 -1; -2 -2; 2  5; 3  4]);
            [~,~,Ax] = find(Anew(:));
            Ax_idx = (1:length(Ax))';
            emosqp('update_P_A', Px, Px_idx, length(Px), Ax, Ax_idx, length(Ax));
            
            % Solve with OSQP
            [x, y, ~, ~, ~] = emosqp('solve');

            % Check if they are close
            testCase.verifyEqual(x, [4.25; 3.25], 'AbsTol',testCase.tol)
            testCase.verifyEqual(y, [0; 0; 3.625; 0; 0], 'AbsTol',testCase.tol)
            
            % Update matrix P to the original value
            P_triu = triu(testCase.P);
            [~,~,Px] = find(P_triu(:));
            [~,~,Ax] = find(testCase.A(:));
            emosqp('update_P_A', Px, [], length(Px), Ax, [], length(Ax));

        end
        
        function test_update_P_A_indP(testCase)
            % Update matrix P
            Pnew = speye(2);
            Pnew_triu = triu(Pnew);
            [~,~,Px] = find(Pnew_triu(:));
            Px_idx = (1:length(Px))';
            Anew = sparse([-1. 0; 0 -1; -2 -2; 2  5; 3  4]);
            [~,~,Ax] = find(Anew(:));
            emosqp('update_P_A', Px, Px_idx, length(Px), Ax, [], length(Ax));
            
            % Solve with OSQP
            [x, y, ~, ~, ~] = emosqp('solve');

            % Check if they are close
            testCase.verifyEqual(x, [4.25; 3.25], 'AbsTol',testCase.tol)
            testCase.verifyEqual(y, [0; 0; 3.625; 0; 0], 'AbsTol',testCase.tol)
            
            % Update matrix P to the original value
            P_triu = triu(testCase.P);
            [~,~,Px] = find(P_triu(:));
            [~,~,Ax] = find(testCase.A(:));
            emosqp('update_P_A', Px, [], length(Px), Ax, [], length(Ax));

        end
        
        function test_update_P_A_indA(testCase)
            % Update matrix P
            Pnew = speye(2);
            Pnew_triu = triu(Pnew);
            [~,~,Px] = find(Pnew_triu(:));
            Anew = sparse([-1. 0; 0 -1; -2 -2; 2  5; 3  4]);
            [~,~,Ax] = find(Anew(:));
            Ax_idx = (1:length(Ax))';
            emosqp('update_P_A', Px, [], length(Px), Ax, Ax_idx, length(Ax));
            
            % Solve with OSQP
            [x, y, ~, ~, ~] = emosqp('solve');

            % Check if they are close
            testCase.verifyEqual(x, [4.25; 3.25], 'AbsTol',testCase.tol)
            testCase.verifyEqual(y, [0; 0; 3.625; 0; 0], 'AbsTol',testCase.tol)
            
            % Update matrix P to the original value
            P_triu = triu(testCase.P);
            [~,~,Px] = find(P_triu(:));
            [~,~,Ax] = find(testCase.A(:));
            emosqp('update_P_A', Px, [], length(Px), Ax, [], length(Ax));

        end
        
        function test_update_P_A_allind(testCase)
            % Update matrix P
            Pnew = speye(2);
            Pnew_triu = triu(Pnew);
            [~,~,Px] = find(Pnew_triu(:));
            Anew = sparse([-1. 0; 0 -1; -2 -2; 2  5; 3  4]);
            [~,~,Ax] = find(Anew(:));
            emosqp('update_P_A', Px, [], length(Px), Ax, [], length(Ax));
            
            % Solve with OSQP
            [x, y, ~, ~, ~] = emosqp('solve');

            % Check if they are close
            testCase.verifyEqual(x, [4.25; 3.25], 'AbsTol',testCase.tol)
            testCase.verifyEqual(y, [0; 0; 3.625; 0; 0], 'AbsTol',testCase.tol)
            
            % Update matrix P to the original value
            P_triu = triu(testCase.P);
            [~,~,Px] = find(P_triu(:));
            [~,~,Ax] = find(testCase.A(:));
            emosqp('update_P_A', Px, [], length(Px), Ax, [], length(Ax));

        end
        
    end

end
