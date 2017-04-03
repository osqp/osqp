@echo on

:: Workaround for CMake not wanting sh.exe on PATH for MinGW
set PATH=%PATH:C:\Program Files (x86)\Git\bin;=%
set PATH=C:\MinGW\bin;%PATH%

:: Activate test environment anaconda
set PATH=%MINICONDA%;%MINICONDA%\\Scripts;%PATH%
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create -q -n test-environment python=%PYTHON_VERSION% numpy scipy nose
call activate test-environment

:: Build OSQP
mkdir build
cd build
cmake -G "%CMAKE_PROJECT%" ..
cmake --build .


@echo off
