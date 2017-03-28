function make_emosqp(target_dir, mex_cfile, EMBEDDED_FLAG)
% Matlab MEX makefile for code generated solver.


% Get make and mex commands
mex_cmd = sprintf('mex -O -silent');

% Add arguments to mex compiler
mexoptflags = '-DMATLAB';

% Add specific generators for windows linux or mac
if (ispc)
    mexoptflags = sprintf('%s %s', mexoptflags, '-DIS_WINDOWS');
else
    if (ismac)
        mexoptflags = sprintf('%s %s', mexoptflags, '-DIS_MAC');
    elseif (isunix)
        mexoptflags = sprintf('%s %s', mexoptflags, '-DIS_LINUX');
    end
end

% Add parameters options to mex
mexoptflags =  sprintf('%s %s %s', mexoptflags, '-DDLONG', '-DDFLOAT');
mexoptflags =  sprintf('%s -DEMBEDDED=%d', mexoptflags, EMBEDDED_FLAG);

% Set optimizer flag
if (~ispc)
    mexoptflags = sprintf('%s %s', mexoptflags, 'COPTIMFLAGS=''-O3''');
end

% Include directory
inc_dir = fullfile(sprintf(' -I%s', target_dir), 'include');

% Source files
cfiles = '';
src_files = dir(fullfile(target_dir, 'src', 'osqp', '*c'));
for i = 1 : length(src_files)
   cfiles = sprintf('%s %s', cfiles, ...
       fullfile(target_dir, 'src', 'osqp', src_files(i).name));
end

% Compile interface
fprintf('Compiling and linking osqpmex...');

% Compile command
cmd = sprintf('%s %s %s %s %s', mex_cmd, mexoptflags, inc_dir, mex_cfile, cfiles);

% Compile
eval(cmd);
fprintf('\t\t\t\t[done]\n');


end
