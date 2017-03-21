%% OSQPtest
  

%% makes a big problem
m = 500;
n = 1000;
A  = sparse(randn(m,n));
lA = -rand(m,1) * 2;
uA = +rand(m,1) * 2;
P = sprandsym(n,0.1,0.1,2);
q = randn(n,1);


%% makes a small problem
A = sparse(eye(2)); 
P = sparse(eye(2));
lA = [-2;-2];
q = [-3;-4];
uA = [10;10]*1;


%% SOLVER TEST
%make a solver object
osqpSolver = osqp;

%setup solver with data
osqpSolver.setup(P,q,A,lA,uA);

% solve it
tic,
out = osqpSolver.solve();
toc, toc, toc

%get the problem dimensions
% [m,n] = osqpSolver.get_dimensions();

return;

%% modify some parameters and try again
lnew = [-2;-2];
qnew = [-3 -4];
unew = [1 1];
osqpSolver.update('q',qnew,'u',unew,'l',lnew)

%also update some settings
opts = osqpSolver.default_settings();
opts = osqpSolver.current_settings();
opts.alpha = 1;  %update something
osqpSolver.update_settings(opts);


out = osqpSolver.solve()
out.x

%% delete the solver
%NB : need to call "clear osqpSolver" to fully clear the 
%invalid handle from the workspace.  This is the expected
%behavior
osqpSolver.delete()

return;
%% QUADPROG SANITY CHECK

Aqp = [A;-A];
bqp = [uA;-lA];
tic
[x,f,flag] = quadprog(P,q,Aqp,bqp);
toc


%% Some random configuration feature checks
h = osqp;

%% OTHER PROBLEM CHECK
P = sparse([4., 1.; 1., 2.]);
q = [1; 1];
A = sparse([1., 1; 1, 0; 0, 1]);
l = [1.0; 0.0; 0.0];
u = [1.0; 0.7; 0.7];

ostest = osqp;
ostest.setup(P, q, A, l, u, 'verbose', 1);
resOSQP = ostest.solve();

% Solve with quadprog
[x,f,flag] = quadprog(P,q,[A; -A], [u;-l]);
% Solve with CPLEX
% cpxopts = cplexoptimset();
% resCPLEX = cplexqp(P, q, [A; -A], [u;-l], cpxopts);

