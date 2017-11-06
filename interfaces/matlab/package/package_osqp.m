function package_osqp()
%   Create OSQP matlab interface package

% Get operative system
if ismac
    platform = 'mac';
elseif isunix
    platform = 'linux';
else
    ispc
    platform = 'windows';
end

% Compile OSQP and get version
fprintf('Compiling OSQP solver\n');
fprintf('---------------------\n');
osqp_dir_matlab = fullfile('..');
cur_dir = pwd;
cd(osqp_dir_matlab);
make_osqp purge;
make_osqp;
% Get OSQP version
s = osqp;
version = s.version;
clear s;
cd(cur_dir)


% Create package
fprintf('Creating Matlab OSQP v%s package\n', version);
fprintf('--------------------------------\n');

% Get package name
package_name = sprintf('osqp-%s-matlab-%s64', version, platform);

% Create package directory
fprintf('Creating package directory %s/...\n', package_name);
if exist(package_name, 'dir')
    rmdir(package_name, 's');
end
mkdir(package_name);
fprintf('[done]\n');

% Copying folders
fprintf('Copying folders...\n');
folders_to_copy = {'codegen', 'unittests'};
for i = 1:length(folders_to_copy)
    folder = folders_to_copy{i};
    fprintf('  Copying  %s/%s/...\n', package_name, folder);
    copyfile(fullfile(osqp_dir_matlab, folder), ...
        fullfile(package_name, folder));
end
fprintf('[done]\n');

% Copying files
fprintf('Copying files...\n');
files_to_copy = {sprintf('osqp_mex.%s', mexext),...
    'osqp.m', ...
    'run_osqp_tests.m'};
for i=1:length(files_to_copy)
    file = files_to_copy{i};
    fprintf('  Copying  %s/%s...\n', package_name, file);
    copyfile(fullfile(osqp_dir_matlab, file), ...
        fullfile(package_name, file));
end
fprintf('[done]\n');


% Compress tar.gz archive
compress_dir(package_name);

% Upload to github
fprintf('Uploading to Bintray.com v%s ...\n', version);
bintray_api_key = input('Bintray API key: ', 's');

interface_upload = input('Do you want to upload the interface archive? [y/n] ', 's');
if interface_upload == 'y'
    fprintf('Uploading %s.tar.gz file\n', package_name);
       
    % Create command
    command = sprintf('curl');
    command = sprintf('%s -T %s', command, sprintf('%s.tar.gz', package_name));
    command = sprintf('%s -ubstellato:%s', command, bintray_api_key);
    command = sprintf('%s -H "X-Bintray-Package:OSQP" -H "X-Bintray-Version:%s"', command, sprintf('%s', version));
    command = sprintf('%s https://api.bintray.com/content/bstellato/generic/OSQP/%s/', command, sprintf('%s', version));
    
    % Run command
    [status, output] = system(command);
    if(status)
        fprintf('\n');
        disp(output);
        error('Error uploading binaries');
    end
    
end

% Upload install_osqp.m
install_osqp_upload = input('Do you also want to upload the install_osqp.m file? [y/n] ', 's');
if install_osqp_upload == 'y'
    fprintf('Uploading install_osqp.m file\n');
    
    
    % Create command
    command = sprintf('curl');
    command = sprintf('%s -T %s', command, 'install_osqp.m');
    command = sprintf('%s -ubstellato:%s', command, bintray_api_key);
    command = sprintf('%s -H "X-Bintray-Package:OSQP" -H "X-Bintray-Version:%s"', command, sprintf('%s', version));
    command = sprintf('%s https://api.bintray.com/content/bstellato/generic/OSQP/%s/', command, sprintf('%s', version));
    
    % Run command
    [status, output] = system(command);
    if(status)
        fprintf('\n');
        disp(output);
        error('Error uploading binaries');
    end
    
end



fprintf('Publishing binaries\n');
command = sprintf('curl -X POST');
command = sprintf('%s -ubstellato:%s', command, bintray_api_key);
command = sprintf('%s https://api.bintray.com/content/bstellato/generic/OSQP/%s/publish', command, sprintf('%s', version));

% Run command
[status, output] = system(command);
if(status)
    fprintf('\n');
    disp(output);
    error('Error publishing binaries');
end




end


function compress_dir(directory)
% COMPRESS_DIR - Compress directory "directory" to tar.gz

fprintf('Compressing files to %s.tar.gz\n', directory);

tar(sprintf('%s.tar.gz', directory), directory);

end