import matlab.unittest.TestSuite;
import matlab.unittest.constraints.IsLessThan;


suiteFolder = TestSuite.fromFolder('./unittests/');

% Run all suite
result = run(suiteFolder);



% res = run(suiteFolder(13));
