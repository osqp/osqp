1;
return;

%% scratch script to compile mex interface for testing
% NB: This should eventually be done via the osqp makefile

clear osqpSolver osqp_mex

%compile all of the osqp source files.  This is not the right way,
%but serves to put off figuring out how to pass the mex-appropriate
%configuration via the makefile
mex -g -c -v -DPRINTLEVEL=3 -I"../include" ...
                         -I"../lin_sys/direct/external/suitesparse/" ...
                         -I"../lin_sys/direct/external/suitesparse/amd/include" ...
                         -I"../lin_sys/direct/external/suitesparse/ldl/include" ...
                         -outdir ./mexObjects ...
                         ../src/*.c ...
                         ../lin_sys/*.c ...
                         ../lin_sys/direct/*.c ...
                         ../lin_sys/direct/external/*.c ...
                         ../lin_sys/direct/external/*.c ...
                         ../lin_sys/direct/external/suitesparse/*.c ...
                         ../lin_sys/direct/external/suitesparse/amd/src/*.c ...
                         ../lin_sys/direct/external/suitesparse/ldl/src/*.c 

return;
                     
%%

%NB : It would be preferable to compile the solver to a 
%dynamic library and include via -llibosqpdir

clear osqpSolver osqp_mex

delete osqp_mex.o
mex -g -v -I"../include" -largeArrayDims osqp_mex.cpp mexObjects/*.o

%make a test object
osqpSolver = osqp
return;

%%
%make a dummy problem
A = sparse(eye(2)); 
P = sparse(eye(2));
uA = [1;1];
lA = [-1;-1];
q = [3;4];

%setup solver with data
osqpSolver.setup(P,q,A,lA,uA);



%% Try to solve it
osqpSolver.solve();
