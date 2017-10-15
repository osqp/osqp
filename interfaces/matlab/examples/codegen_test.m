%% Simple problem
m = 50;
n = 100;
problem.A  = sparse(randn(m,n));
problem.l = -rand(m,1) * 2;
problem.u = +rand(m,1) * 2;
problem.P = sprandsym(n,0.1,0.1,2);
problem.q = randn(n,1);


m = osqp;
m.setup(problem.P, problem.q, problem.A, problem.l, problem.u);

m.codegen('code', 'parameters', 'matrices');