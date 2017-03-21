%% Simple problem
m = 50;
n = 100;
problem.A  = sparse(randn(m,n));
problem.l = -rand(m,1) * 2;
problem.u = +rand(m,1) * 2;
problem.P = sprandsym(n,0.1,0.1,2);
problem.q = randn(n,1);

%% Primal infeasible problem
rng(4)
n = 50;
m = 500;
Pt = sprandn(n, n, 0.6);
problem.P = Pt' * Pt;
problem.q = randn(n, 1);
problem.A = sprandn(m, n, 0.8);
problem.u = 3 + randn(m, 1);
problem.l = -3 + randn(m, 1);

% Make random problem primal infeasible
nhalf = floor(n/2);
problem.A(nhalf, :) = problem.A(nhalf + 1, :);
problem.l(nhalf) = problem.u(nhalf + 1) + 10 * rand();
problem.u(nhalf) = problem.l(nhalf) + 0.5;


%% Dual infeasible problem
problem.P = sparse(diag([4; 0]));
problem.q = [0; 2];
problem.A = sparse([1 1; -1 1]);
problem.l = [-Inf; -Inf];
problem.u = [2; 3];


%% Setup and solve the problem
% Setup settings
settings.alpha = 1.6;
settings.rho = 0.1;
settings.sigma = 0.1;
settings.eps_prim_inf = 1e-5;
settings.eps_dual_inf = 1e-5;
settings.eps_rel = 1e-5;
settings.eps_abs = 1e-5;
settings.max_iter = 2500;
settings.verbose = 1;
settings.scaling = 0; % Disable scaling. Pure Matlab implementation does not support it yet



% Solve with osqp
solver = osqp;
solver.setup(problem.P, problem.q, problem.A, problem.l, problem.u, settings);
resOSQP = solver.solve();


% Solve with osqpmatlab
[xmatlab, ymatlab, costmatlab, statusmatlab, itermatlab] = osqpmatlab(problem,[], settings);


% Check solution if the problem is solved
if resOSQP.info.status_val == 1
    assert(norm(xmatlab - resOSQP.x) < settings.eps_abs*10, 'Error primal solution')
    assert(norm(ymatlab - resOSQP.y) < settings.eps_abs*10, 'Error dual solution')
    assert(norm(costmatlab - resOSQP.info.obj_val) < settings.eps_abs*10, 'Error cost function')
end
