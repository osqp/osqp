function [x, y, cost, status, iter] = osqpmatlab(problem, warm_start, settings) %#codegen
% OSQPMATLAB Pure Matlab implementation of the OSQP solver
%
% osqpmatlab solves the QP problem
%
%              min       .5*x'*P*x+q'*x
%              s.t.      l<=A*x<=u
%
% P=P'>=0, x\in R^n, l,u\in R^m.
%
% INPUTS
%   - problem:
%     + P: quadratic cost matrix
%     + q: linear component of cost function
%     + A: linear inequalities matrix
%     + l: lower bound of inequality constraint
%     + u: upper bound of inequality constraint
%   - warm_start:
%     + x0: primal variable warm start
%     + y0: dual variable warm start
%   - settings: structure of settings parameters
%     + rho: step-size rho
%     + sigma: step-size sigma
%     + alpha: overrelaxation parameter
%     + eps_abs: absolute convergence tolerance
%     + eps_rel: relative convergence tolerance
%     + eps_prim_inf: primal infeasibility tolerance
%     + eps_dual_inf: dual infeasibility tolerance
%     + max_iter: maximum number of iterations
%     + verbose: verbosity of the solver
%       TODO: Add these!
%     + linsys_solver: linear system solver 
%     + scaled_termination: evaluate termination criteria
%     + check_termination: evaluate termination criteria
%
% OUTPUTS
%   - x: primal solution
%   - y: dual solution
%   - cost: primal cost at the optimum
%   - status: solver status. Values: 0 Not solved (max iters reached)
%                                    1 Solved
%                                   -1 Primal infeasible
%   - iter: number of iterations
%
% (C) 2017 by B. Stellato, Lucca, January 11, 2016
%          adapted from Bemporad, May 19, 2016 (qp_admm_iterative_refinement)
%

% Get problem dimensions
n = size(problem.A, 2);
m = numel(problem.l);

% Initialize variables before iterations
if (~isempty(warm_start)) && (~isempty(warm_start.x0))
    x = warm_start.x0;
else
    x = zeros(n, 1);
end
if (~isempty(warm_start)) && (~isempty(warm_start.y0))
    y = warm_start.y0;
else
    y = zeros(m, 1);
end



z = zeros(m, 1,'like', problem.l);
xz_tilde = zeros(n+m, 1,'like', problem.l);

% Set initial status and cost
status = 0;

% Print header
if settings.verbose
    fprintf('\niter  \t     cost  \t pri_res \t dual_res\n');
end


KKT = [problem.P + settings.sigma * eye(n), problem.A';
       problem.A, -1/settings.rho * eye(m)];
[L,D,p] = ldl(KKT,'vector');

% ADMM Iterations
for iter = 1:settings.max_iter

    % Assign previous variables
    x_prev = x; z_prev = z; y_prev = y;

    % Step 1: compute x_tilde and z_tilde
    rhs = [settings.sigma * x - problem.q; z - 1./settings.rho * y];  % Form right-hand side
    xz_tilde(p, :) =  L'\(D\(L\rhs(p, :)));  % Solve linear system
    xz_tilde(n+1:end, :) = z + 1./settings.rho * (xz_tilde(n+1:end, :) - y); % Update z_tilde

    % Step 2: Perform projections to obtain x and z
    x = settings.alpha * xz_tilde(1:n) + (1 - settings.alpha) * x_prev;
    z = min(max(settings.alpha * xz_tilde(n+1:end) + (1 - settings.alpha) * z_prev + 1./settings.rho * y, problem.l), problem.u);

    % Step 3: Update dual variables y
    y = y + settings.rho * (settings.alpha * xz_tilde(n+1:end) + (1 - settings.alpha) * z_prev - z);

    % Update cost
    cost = .5 * x'* problem.P * x + problem.q' * x;

    % Compute residuals
    Ax = problem.A * x; % precompute A * x
    Px = problem.P * x; % precompute P * x
    Aty = problem.A' * y; % precompute A' * y
    pri_res = norm(Ax - z);
    dual_res = norm(Px + problem.q + Aty);

    % Compute delta_x and delta_y
    delta_x = x - x_prev;
    delta_y = y - y_prev;


     % Print stats
     if settings.verbose
         if mod(iter, 100) == 0 || iter == 1 || iter == settings.max_iter
            fprintf('%i \t %.4e \t%.4e \t%.4e\n', iter, cost, pri_res, dual_res);
         end
     end

    % Check primal infeasibility
    norm_delta_y = norm(delta_y);
    if norm_delta_y > settings.eps_prim_inf^2
        delta_y = delta_y/norm_delta_y;
        ineq_lhs = problem.u'*max(delta_y, 0) + problem.l'*min(delta_y, 0);
        if ineq_lhs < -settings.eps_prim_inf
            if norm(problem.A'*delta_y) < settings.eps_prim_inf
                  status = -3;
                  cost = Inf;
                  x = NaN(n, 1);
                  y = NaN(m, 1);
                  break;
            end
        end
    end

    % Check dual infeasibility
    norm_delta_x = norm(delta_x);
    if norm_delta_x > settings.eps_dual_inf^2
       delta_x = delta_x/norm_delta_x;
       if problem.q'*delta_x < -settings.eps_dual_inf
           if norm(problem.P*delta_x) < settings.eps_dual_inf
               Adelta_x = problem.A * delta_x;

               for i = 1:m
                % De Morgan Law Applied to dual infeasibility conditions for A * x
                   if (problem.u(i) < 1e+18) && (Adelta_x(i) > settings.eps_dual_inf) || ...
                       (problem.l(i) > -1e+18) && (Adelta_x(i) < -settings.eps_dual_inf)
                        % At least one condition is not satisfied for
                        % dual infeasibility
                        break
                   end
               end

               % All conditions passed. Problem dual infeasible
               status = -4;
               cost = -Inf;
               x = NaN(n, 1);
               y = NaN(m, 1);
               break;
           end
       end
    end



    % Check convergence by computing residualls using eps_abs and eps_rel
    eps_pri = settings.eps_abs*sqrt(m) + settings.eps_rel*norm(z);
    eps_dual = settings.eps_abs*sqrt(n) + settings.eps_rel* settings.rho*norm(Aty);
    if pri_res < eps_pri && dual_res < eps_dual
        status = 1;  % Problem solved
        break;
    end


end

if (iter ~= settings.max_iter) && (mod(iter, 100) == 0)
    % Print info for last iteration
    fprintf('%i \t %.4e \t%.4e \t%.4e\n', iter, cost, pri_res, dual_res);
end

end
