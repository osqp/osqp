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
DFLOAT = false;
DLONG = true;


%% Basic compile commands

% Get make and mex commands
make_cmd = 'cmake --build .';
mex_cmd = sprintf('mex -O -silent');


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

if DLONG
   cmake_args = sprintf('%s %s', cmake_args, '-DDLONG:BOOL=ON');
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


% Set library extension
lib_ext = '.a';
lib_name = sprintf('libosqpdirstatic%s', lib_ext);


% Set osqp directory and osqp_build directory
osqp_dir = fullfile('..', '..');
osqp_build_dir = sprintf('%s/build', osqp_dir);

% Include directory
inc_dir = fullfile(sprintf('-I%s', osqp_dir), 'include');


%% OSQP Solver
if( any(strcmpi(what,'osqp')) || any(strcmpi(what,'all')) )
   fprintf('Compiling OSQP solver...');

    % Create build directory and go inside
    if ~exist(osqp_build_dir, 'dir')
        mkdir(osqp_build_dir);
    end
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
        fprintf(output);
        error('Error configuring CMake environment');
    end

    [status, output] = system(sprintf('%s %s', make_cmd, '--target osqpdirstatic'));
    if (status)
        fprintf('\n');
        fprintf(output);
        error('Error compiling OSQP');
    end


    % Change directory back to matlab interface
    cd(fullfile('..', 'interfaces', 'matlab'));

    % Copy static library to current folder
    lib_origin = fullfile(osqp_build_dir, 'out', lib_name);
    copyfile(lib_origin, lib_name);

    fprintf('\t\t\t\t\t[done]\n');

end

%% osqpmex
if( any(strcmpi(what,'osqp_mex')) || any(strcmpi(what,'all')) )
    % Compile interface
    fprintf('Compiling and linking osqpmex...');

    % Compile command
    cmd = sprintf('%s %s %s %s osqp_mex.cpp', mex_cmd, mexoptflags, inc_dir, lib_name);

    % Compile
    eval(cmd);
    fprintf('\t\t\t[done]\n');

end



%% clean
if( any(strcmpi(what,'clean')) || any(strcmpi(what,'purge')) )
    fprintf('Cleaning mex files...  ');

    % Delete mex file
    clear osqp
    binfile = ['osqp_mex.', mexext];
    if( exist(binfile,'file') )
        delete(['osqp_mex.',mexext]);
    end

    % Delete static library
    if( exist(lib_name,'file') )
        delete(lib_name);
    end

    fprintf('\t\t\t\t\t\t[done]\n');
end


%% purge
if( any(strcmpi(what,'purge')) )
    fprintf('Cleaning mex files and OSQP build...  ');

    % Delete OSQP build directory
    if exist(osqp_build_dir, 'dir')
        rmdir(osqp_build_dir, 's');
    end

    % Delete mex file
    clear osqp
    binfile = ['osqp_mex.', mexext];
    if( exist(binfile,'file') )
        delete(['osqp_mex.',mexext]);
    end

    % Delete static library
    if( exist(lib_name,'file') )
        delete(lib_name);
    end

    fprintf('\t\t[done]\n');
end


end
