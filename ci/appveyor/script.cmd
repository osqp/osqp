@echo on

:: Set environment if 64bit
IF "%PLATFORM%"=="x64" (
"C:\Program Files\Microsoft SDKs\Windows\v7.1\Bin\SetEnv.cmd" /x64
)

:: Perform C Tests
:: -----------------------------------------------------
:: Build C
cd %APPVEYOR_BUILD_FOLDER%
mkdir build
cd build
cmake -G "%CMAKE_PROJECT%" ..
cmake --build .

:: Test C
%APPVEYOR_BUILD_FOLDER%\build\out\osqp_tester_direct.exe
if errorlevel 1 exit /b 1

:: Perform Python tests
:: -------------------------------------------------------
:: Install python interface
cd %APPVEYOR_BUILD_FOLDER%\interfaces\python
python setup.py install

:: Test python interface
cd %APPVEYOR_BUILD_FOLDER%\interfaces\python
pytest
if errorlevel 1 exit /b 1

@echo off
