%%******************************************************************************
%%                                                                             *
%%  ONLINE QP BENCHMARK COLLECTION                                             *
%%  http://homes.esat.kuleuven.be/~optec/software/onlineQP                     *
%%                                                                             *
%%  maintained by: Moritz Diehl and Hans Joachim Ferreau                       *
%%                                                                             *
%%******************************************************************************


%%
%% filename:     loadBenchmark.m
%% author:       Hans Joachim Ferreau, joachim.ferreau@esat.kuleuven.be
%%
%% description:  this script loads all benchmark data into the workspace
%%               (attention: existing data may be overridden!)
%%


disp('INFO (loadBenchmark):  Loading benchmark data... ');

%% load dimensions
dims = load('dims.oqp');
nQP = dims(1);
nV  = dims(2);
nC  = dims(3);
nEC = dims(4);
clear dims;

%% load other data ...
H  = load('H.oqp');
g  = load('g.oqp');
lb = load('lb.oqp');
ub = load('ub.oqp');

%% ... including constraints (if any)
if ( nC > 0 )
    A   = load('A.oqp');
    lbA = load('lbA.oqp');
    ubA = load('ubA.oqp');
end

%% load optimal solutions
x_opt = load('x_opt.oqp');
y_opt = load('y_opt.oqp');
obj_opt = load('obj_opt.oqp');


%% finally, perform some consistency checks ...
successful = 1;

if ( ( nQP <= 0 ) || ( nV <= 0 ) || ( nC < 0 ) || ( nEC < 0 ) || ( nEC > nC ) )
    disp('ERROR (loadBenchmark):  Dimension data invalid!');
    return;
end

[N,M] = size(H);
if ( ( N ~= M ) || ( N ~= nV ) )
    disp('ERROR (loadBenchmark):  Hessian matrix has wrong dimensions!');
    successful = 0;
end

[N,M] = size(g);
if ( ( N ~= nQP ) || ( M ~= nV ) )
    disp('ERROR (loadBenchmark):  Gradient series has wrong dimensions!');
    successful = 0;
end

[N,M] = size(lb);
if ( ( N ~= nQP ) || ( M ~= nV ) )
    disp('ERROR (loadBenchmark):  Lower bound series has wrong dimensions!');
    successful = 0;
end

[N,M] = size(ub);
if ( ( N ~= nQP ) || ( M ~= nV ) )
    disp('ERROR (loadBenchmark):  Upper bound series has wrong dimensions!');
    successful = 0;
end

%% ... including constraints (if any)
if ( nC > 0 )
    [N,M] = size(A);
    if ( ( N ~= nC ) || ( M ~= nV ) )
        disp('ERROR (loadBenchmark):  Constraint matrix has wrong dimensions!');
        successful = 0;
    end

    [N,M] = size(lbA);
    if ( ( N ~= nQP ) || ( M ~= nC ) )
        disp('ERROR (loadBenchmark):  Lower bound series has wrong dimensions!');
        successful = 0;
    end

    [N,M] = size(ubA);
    if ( ( N ~= nQP ) || ( M ~= nC ) )
        disp('ERROR (loadBenchmark):  Upper bound series has wrong dimensions!');
        successful = 0;
    end
end

if ( successful == 0 )
    disp('INFO  (loadBenchmark):  Errors occured while loading benchmark data!');
else
    disp('INFO  (loadBenchmark):  Benchmark data loaded successfully!');
end

clear N M successful;


%
% end of file
%
