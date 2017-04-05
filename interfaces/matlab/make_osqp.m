function make_osqp(varargin)
% Matlab MEX makefile for OSQP.
%
%    MAKEMEX(VARARGIN) is a make file for OSQP solver. It
%    builds OSQP and its components from source.
%
%    WHAT is the last element of VARARGIN and cell array of strings,
%    with the following options:
%
%    {}, '' (empty string) or 'all': build all components and link.
%
%    'osqp': builds the OSQP solver using CMake
%
%    'osqp_mex': builds the OSQP mex interface and links it to the OSQP
%    library
%
%    VARARGIN{1:NARGIN-1} specifies the optional flags passed to the compiler
%
%    Additional commands:
%
%    makemex clean - delete all object files (.o and .obj)
%    makemex purge - same as above, and also delete the mex files.


if( nargin == 0 )
    what = {'all'};
else
    what = varargin{nargin};
    if(isempty(strfind(what, 'all'))         && ...
        isempty(strfind(what, 'osqp'))        && ...
        isempty(strfind(what, 'osqp_mex')) && ...
        isempty(strfind(what, 'clean'))       && ...
        isempty(strfind(what, 'purge')))
    fprintf('No rule to make target "%s", exiting.\n', what);
    end
end


% Default parameters
PRINTING = true;
PROFILING = true;
CTRLC = true;
DFLOAT = false;
DLONG = true;


%% Basic compile commands

% Get make and mex commands
make_cmd = 'cmake --build .';
mex_cmd = sprintf('mex -O -silent');
mex_libs = '';


% Add arguments to cmake and mex compiler
cmake_args = '-DUNITTESTS=OFF -DMATLAB=ON';
mexoptflags = '';

% Add specific generators for windows linux or mac
if (ispc)
    cmake_args = sprintf('%s %s', cmake_args, '-G "MinGW Makefiles"');
    mexoptflags = sprintf('%s %s', mexoptflags, '-DIS_WINDOWS');
else
    cmake_args = sprintf('%s %s', cmake_args, '-G "Unix Makefiles"');
    if (ismac)
      mexoptflags = sprintf('%s %s', mexoptflags, '-DIS_MAC');
    else
        if (isunix)
          mexoptflags = sprintf('%s %s', mexoptflags, '-DIS_LINUX');
        end
    end
end

% Pass Matlab root to cmake
Matlab_ROOT = strrep(matlabroot, '\', '/');
cmake_args = sprintf('%s %s%s%s', cmake_args, ...
    '-DMatlab_ROOT_DIR="', Matlab_ROOT, '"');

% Add parameters options to mex and cmake
if PROFILING
   cmake_args = sprintf('%s %s', cmake_args, '-DPROFILING:BOOL=ON');
   mexoptflags =  sprintf('%s %s', mexoptflags, '-DPROFILING');
end

if PRINTING
   cmake_args = sprintf('%s %s', cmake_args, '-DPRINTING:BOOL=ON');
   mexoptflags =  sprintf('%s %s', mexoptflags, '-DPRINTING');
end

if CTRLC
   cmake_args = sprintf('%s %s', cmake_args, '-DCTRLC:BOOL=ON');
   mexoptflags =  sprintf('%s %s', mexoptflags, '-DCTRLC');
   mex_libs = sprintf('%s %s', mex_libs, '-lut');
end

if DLONG
   cmake_args = sprintf('%s %s', cmake_args, '-DDLONG:BOOL=ON');
   mexoptflags =  sprintf('%s %s', mexoptflags, '-DDLONG');
   mexoptflags =  sprintf('%s %s', mexoptflags, '-DDLONG');
end

if DFLOAT
   cmake_args = sprintf('%s %s', cmake_args, '-DDFLOAT:BOOL=ON');
   mexoptflags =  sprintf('%s %s', mexoptflags, '-DDFLOAT');
end


% Add large arrays support if computer 64 bit
if (~isempty (strfind (computer, '64')))
    mexoptflags = sprintf('%s %s', mexoptflags, '-largeArrayDims');
end


% Pass MATLAB flag to mex compiler
mexoptflags = sprintf('%s %s', mexoptflags, '-DMATLAB');

% Set optimizer flag
if (~ispc)
    mexoptflags = sprintf('%s %s', mexoptflags, 'COPTIMFLAGS=''-O3''');
end

% Set library extension
lib_ext = '.a';
lib_name = sprintf('libosqpdirstatic%s', lib_ext);


% Set osqp directory and osqp_build directory
osqp_dir = fullfile('..', '..');
osqp_build_dir = fullfile(osqp_dir, 'build');
suitesparse_dir = fullfile(osqp_dir, 'lin_sys', 'direct', 'suitesparse');
cg_sources_dir = fullfile('codegen', 'sources');

% Include directory
inc_dir = [
    fullfile(sprintf(' -I%s', osqp_dir), 'include'), ...
    sprintf(' -I%s', suitesparse_dir), ...
    fullfile(sprintf(' -I%s', suitesparse_dir), 'ldl', 'include'), ...
    fullfile(sprintf(' -I%s', suitesparse_dir), 'amd', 'include')
    ];


%% OSQP Solver
if( any(strcmpi(what,'osqp')) || any(strcmpi(what,'all')) )
   fprintf('Compiling OSQP solver...');

    % Create build directory and go inside
    if exist(osqp_build_dir, 'dir')
        rmdir(osqp_build_dir, 's');
    end
    mkdir(osqp_build_dir);
    cd(osqp_build_dir);

    % Extend path for CMAKE mac (via Homebrew)
    PATH = getenv('PATH');
    if ((ismac) && (isempty(strfind(PATH, '/usr/local/bin'))))
        setenv('PATH', [PATH ':/usr/local/bin']);
    end

    % Compile static library with CMake
    [status, output] = system(sprintf('%s %s ..', 'cmake', cmake_args));
    if(status)
        fprintf('\n');
        disp(output);
        error('Error configuring CMake environment');
    end

    [status, output] = system(sprintf('%s %s', make_cmd, '--target osqpdirstatic'));
    if (status)
        fprintf('\n');
        disp(output);
        error('Error compiling OSQP');
    end


    % Change directory back to matlab interface
    cd(fullfile('..', 'interfaces', 'matlab'));

    % Copy static library to current folder
    lib_origin = fullfile(osqp_build_dir, 'out', lib_name);
    copyfile(lib_origin, lib_name);

    fprintf('\t\t\t\t\t\t[done]\n');

end

%% osqpmex
if( any(strcmpi(what,'osqp_mex')) || any(strcmpi(what,'all')) )
    % Compile interface
    fprintf('Compiling and linking osqpmex...');

    % Compile command
    %cmd = sprintf('%s %s %s %s osqp_mex.cpp', mex_cmd, mexoptflags, inc_dir, lib_name);
    cmd = sprintf('%s %s %s %s osqp_mex.cpp %s', ...
        mex_cmd, mexoptflags, inc_dir, lib_name, mex_libs);
    
    % Compile
    eval(cmd);
    fprintf('\t\t\t\t[done]\n');

end


%% codegen
if( any(strcmpi(what,'codegen')) || any(strcmpi(what,'all')) )
    fprintf('Copying source files for codegen...');

    % Copy C files
    cg_src_dir = fullfile(cg_sources_dir, 'src');
    if ~exist(cg_src_dir, 'dir')
        mkdir(cg_src_dir);
    end
    cfiles = [dir(fullfile(osqp_dir, 'src', '*.c'));
              dir(fullfile(suitesparse_dir, '*.c'));
              dir(fullfile(suitesparse_dir, 'ldl', 'src', '*.c'))];
    for i = 1 : length(cfiles)
        if ~any(strcmp(cfiles(i).name, {'cs.c', 'ctrlc.c', 'polish.c', 'SuiteSparse_config.c'}))
            copyfile(fullfile(cfiles(i).folder, cfiles(i).name), ...
                fullfile(cg_src_dir, cfiles(i).name));
        end
    end

    % Copy H files
    cg_include_dir = fullfile(cg_sources_dir, 'include');
    if ~exist(cg_include_dir, 'dir')
        mkdir(cg_include_dir);
    end
    hfiles = [dir(fullfile(osqp_dir, 'include', '*.h'));
              dir(fullfile(suitesparse_dir, '*.h'));
              dir(fullfile(suitesparse_dir, 'ldl', 'include', '*.h'))];
    for i = 1 : length(hfiles)
        if ~any(strcmp(hfiles(i).name, {'cs.h', 'ctrlc.h', 'polish.h', 'SuiteSparse_config.h'}))
            copyfile(fullfile(hfiles(i).folder, hfiles(i).name), ...
                fullfile(cg_include_dir, hfiles(i).name));
        end
    end

    fprintf('\t\t\t\t[done]\n');

end


%% clean
if( any(strcmpi(what,'clean')) || any(strcmpi(what,'purge')) )
    fprintf('Cleaning mex files and library...');

    % Delete mex file
    mexfiles = dir(['*.', mexext]);
    for i = 1 : length(mexfiles)
        delete(mexfiles(i).name);
    end

    % Delete static library
    if( exist(lib_name,'file') )
        delete(lib_name);
    end

    fprintf('\t\t\t\t[done]\n');
end


%% purge
if( any(strcmpi(what,'purge')) )
    fprintf('Cleaning OSQP build and codegen directories...');

    % Delete OSQP build directory
    if exist(osqp_build_dir, 'dir')
        rmdir(osqp_build_dir, 's');
    end

    % Delete codegen files
    if exist(cg_sources_dir, 'dir')
        rmdir(cg_sources_dir, 's');
    end

    fprintf('\t[done]\n');
end


end
