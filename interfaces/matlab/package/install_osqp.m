function install_osqp
    % Install the OSQP solver Matlab interface

    % Get current operating system
    if ispc
        platform = 'win';
    elseif ismac
        platform = 'mac';
    elseif isunix
        platform = 'linux';
    end

    % Package name
    package_name = sprintf('https://github.com/oxfordcontrol/osqp/releases/download/v0.0.0/osqp-0.0.0-matlab-%s.tar.gz', platform);
    urlwrite(package_name,'osqp.tar.gz');
    untar('osqp.tar.gz','osqp')
    cd osqp
    addpath(pwd);
    savepath
    cd ..

end
