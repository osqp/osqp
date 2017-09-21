function install_osqp
    % Install the OSQP solver Matlab interface

    % Get current operating system
    if ispc
        platform = 'windows';
    elseif ismac
        platform = 'mac';
    elseif isunix
        platform = 'linux';
    end

    fprintf('Downloading binaries...');
    package_name = sprintf('https://github.com/oxfordcontrol/osqp/releases/download/v0.1.202/osqp-0.1.202-matlab-%s.tar.gz', platform);
    websave('osqp.tar.gz', package_name);
    fprintf('\t\t\t\t[done]\n');

    fprintf('Unpacking...');
    untar('osqp.tar.gz','osqp')
    fprintf('\t\t\t\t\t[done]\n');

    fprintf('Updating path...');
    cd osqp
    addpath(genpath(pwd));
    savepath
    cd ..
    fprintf('\t\t\t\t[done]\n');

    fprintf('Deleting temporary files...');
    delete('osqp.tar.gz');
    fprintf('\t\t\t[done]\n');

    fprintf('OSQP is successfully installed!\n');


end
