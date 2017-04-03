@echo on

echo %APPVEYOR_BUILD_FOLDER%
echo %PATH%

:: Perform tests
cd %APPVEYOR_BUILD_FOLDER%\build
out\osqp_tester_direct.exe

@echo off
