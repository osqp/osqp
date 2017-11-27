import matlab.unittest.TestSuite;
import matlab.unittest.constraints.IsLessThan;

[osqp_path,~,~] = fileparts(which('osqp.m'));
unittest_dir = fullfile(osqp_path, 'unittests');
suiteFolder = TestSuite.fromFolder(unittest_dir);

% Solve individual test file
% suiteFolder = TestSuite.fromFile('unittests/update_matrices_tests.m');


% Run all suite
result = run(suiteFolder);
