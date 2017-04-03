@echo on


:: Remove entry with sh.exe from PATH to fix error with MinGW toolchain
:: (For MinGW make to work correctly sh.exe must NOT be in your path)
:: http://stackoverflow.com/a/3870338/2288008
set PATH=%PATH:C:\Program Files\Git\usr\bin;=%

:: Workaround for CMake not wanting sh.exe on PATH for MinGW
REM set PATH=%PATH:C:\Program Files (x86)\Git\bin;=%
REM REM set PATH=C:\MinGW\bin;%PATH%

:: Activate test environment anaconda
set PATH=%MINICONDA%;%MINICONDA%\\Scripts;%PATH%
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create -q -n test-environment python=%PYTHON_VERSION% numpy scipy nose
call activate test-environment :: Need to run with call otherwise the script hangs

:: Build OSQP
mkdir build
cd build
cmake -G "%CMAKE_PROJECT%" ..
cmake --build .


@echo off
