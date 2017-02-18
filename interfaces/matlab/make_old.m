function make(varargin)
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
%    'lin_solver': build the sparse LDL solver with the AMD package
%
%    'osqp': builds the OSQP solver
%
%    'osqp_mex': builds the OSQP mex interface - this involves linking of
%    the packages LDL, AMD, OSQP that must have been built
%    before.
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
        isempty(strfind(what, 'lin_solver'))  && ...
        isempty(strfind(what, 'clean'))       && ...
        isempty(strfind(what, 'purge'))       && ...
        isempty(strfind(what, 'osqp'))        && ...
        isempty(strfind(what, 'osqp_mex')) )
    fprintf('No rule to make target "%s", exiting.\n', what);
    end
end


% Set optimization flags
optflags = sprintf('-DPRINTING -DPROFILING');

if(nargin > 1)
    for i=1:nargin-1
        optflags = sprintf('%s %s', optflags, varargin{i});
    end
end


%% Basic compile commands
makeobjcmd = sprintf('mex -g -c -O -silent');
makemexcmd = sprintf('mex -g -O -silent');


% Set mex linking flags
mexoptflags = sprintf('');

if ( isunix && ~ismac )
    mexoptflags = sprintf('%s %s', mexoptflags, '-lm -lut -lrt');
elseif  ( ismac )
    mexoptflags = sprintf('%s %s', mexoptflags, '-lm -lut');
else
    mexoptflags = sprintf('%s %s', mexoptflags, '-lut');
end

if (~isempty (strfind (computer, '64')))
    mexoptflags = sprintf('%s %s', mexoptflags, '-largeArrayDims');
else
    mexoptflags = '';
end


% Set specific objects output for windows or unix
if( ispc )
    objext = 'obj';
else
    if ( ismac  || isunix)
    objext = 'o';
    end
end



% Set output
outdir = './out/';
outopt = sprintf('-outdir %s', outdir);

% Suitesparse Directory
linsys_direct_dir = sprintf('./osqp/lin_sys/direct');
ss_dir = sprintf('./osqp/lin_sys/direct/external/suitesparse');

% Include directory
inc_dir = sprintf(' -I%s/ -I%s/ldl/include -I%s/amd/include -I./osqp/include', ss_dir, ss_dir, ss_dir);

%% Linear Systems Solver
if( any(strcmpi(what,'lin_solver')) || any(strcmpi(what,'all')) )
    fprintf('Compiling linear systems solver...');

    % Basic command
    cmd = sprintf('%s %s %s', makeobjcmd, optflags, inc_dir);

    % Suitesparse main files
    ss_files = dir(strcat(ss_dir,'/','*.c'));
    ss_files = {ss_files.name};
    for i = 1:length(ss_files)
       cmd = sprintf('%s %s/%s', cmd, ss_dir, ss_files{i});
    end

    % AMD files
    amd_files = dir(strcat(ss_dir,'/amd/src/','*.c'));
    amd_files = {amd_files.name};
    for i = 1:length(amd_files)
       cmd = sprintf('%s %s/amd/src/%s', cmd, ss_dir, amd_files{i});
    end

    % LDL files
    ldl_files = dir(strcat(ss_dir,'/ldl/src/','*.c'));
    ldl_files = {ldl_files.name};
    for i = 1:length(ldl_files)
       cmd = sprintf('%s %s/ldl/src/%s', cmd, ss_dir, ldl_files{i});
    end

    % Linear Systems Direct Solver
    linsys_files = dir(strcat(linsys_direct_dir, '/','*.c'));
    linsys_files = {linsys_files.name};
     for i = 1:length(linsys_files)
       cmd = sprintf('%s %s/%s', cmd, linsys_direct_dir, linsys_files{i});
    end

    % Add output directive
    cmd = sprintf('%s %s', cmd, outopt);

    % Compile
    eval(cmd);

    fprintf('\t\t\t[done]\n');

end

%% OSQP Solver
if( any(strcmpi(what,'osqp')) || any(strcmpi(what,'all')) )
   fprintf('Compiling OSQP solver...');
    % Basic command
    cmd = sprintf('%s %s %s', makeobjcmd, optflags, inc_dir);

    % Add files
    osqp_files = dir('./osqp/src/*.c');
    osqp_files = {osqp_files.name};
    for i = 1:length(osqp_files)
       cmd = sprintf('%s ./osqp/src/%s', cmd, osqp_files{i});
    end

    % Add output directive
    cmd = sprintf('%s %s', cmd, outopt);

    % Compile
    eval(cmd);

    fprintf('\t\t\t\t[done]\n');

end

%% osqpmex
if( any(strcmpi(what,'osqp_mex')) || any(strcmpi(what,'all')) )
    % Compile interface
    fprintf('Compiling and linking osqpmex...');

    % Basic command
    cmd = sprintf('%s %s %s %s', makemexcmd, optflags, inc_dir, mexoptflags);

    % Add files
    cmd = sprintf('%s osqp_mex.cpp ./out/*.%s', cmd, objext);

    % Compile
    eval(cmd);
    fprintf('\t\t\t[done]\n');

end




%% clean
if( any(strcmpi(what,'clean')) || any(strcmpi(what,'purge')) )
    fprintf('Cleaning up objects...  ');
    if exist(outdir, 'dir')
        rmdir(outdir, 's');
    end
    fprintf('\t\t\t\t[done]\n');
end


%% purge
if( any(strcmpi(what,'purge')) )
    fprintf('Deleting mex files...  ');
    clear osqp
    binfile = ['osqp_mex.', mexext];
    if( exist(binfile,'file') )
        delete(['osqp_mex.',mexext]);
    end
    fprintf('\t\t\t\t\t[done]\n');
end


end
