function make_emosqp(target_dir, mex_cfile, EMBEDDED_FLAG)
% Matlab MEX makefile for code generated solver.


% Get make and mex commands
mex_cmd = sprintf('mex -O -silent');

% Add arguments to mex compiler
mexoptflags = '-DMATLAB';

% Add embedded flag
cmake_args = sprintf('-DEMBEDDED:INT=%i', EMBEDDED_FLAG);


% Generate glop_opts.h file by running cmake
current_dir = pwd;
build_dir = fullfile(target_dir, 'build');
cd(target_dir);
if exist(build_dir, 'dir')
    rmdir('build', 's');
end
mkdir('build');
cd('build');
[status, output] = system(sprintf('%s %s ..', 'cmake', cmake_args));
if(status)
    fprintf('\n');
    disp(output);
    error('Error generating glob_opts.h');
end
cd(current_dir);

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
