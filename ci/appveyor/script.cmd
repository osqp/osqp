@echo on


:: Perform C Tests
:: -----------------------------------------------------
:: Build C
cd %APPVEYOR_BUILD_FOLDER%
mkdir build
cd build
cmake -G "%CMAKE_PROJECT%" ..
cmake --build .

:: Test C
%APPVEYOR_BUILD_FOLDER%\build\out\osqp_tester.exe
if errorlevel 1 exit /b 1


@echo off
