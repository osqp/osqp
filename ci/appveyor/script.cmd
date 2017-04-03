@echo on

:: Perform C Tests
cd %APPVEYOR_BUILD_FOLDER%\build
out\osqp_tester_direct.exe

:: Perform Python tests
cd %APPVEYOR_BUILD_FOLDER%\interfaces\python
nosetests


@echo off
