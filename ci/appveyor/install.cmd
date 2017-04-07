@echo on


:: Remove entry with sh.exe from PATH to fix error with MinGW toolchain
:: (For MinGW make to work correctly sh.exe must NOT be in your path)
:: http://stackoverflow.com/a/3870338/2288008
set PATH=%PATH:C:\Program Files\Git\usr\bin;=%
set PATH=C:\MinGW\bin;%PATH%


:: Activate test environment anaconda

:: N.B. If you split it over multiple lines it does not work (why!??!!?)
IF "%PLATFORM%"=="x86" (set PATH=%MINICONDA%;%MINICONDA%\\Scripts;%PATH%) ELSE (set PATH=%MINICONDA%-%PLATFORM%;%MINICONDA%-%PLATFORM%\\Scripts;%PATH%)

conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda install conda-build anaconda-client
conda create -q -n test-environment python=%PYTHON_VERSION% numpy scipy pytest future
conda install -c conda-forge twine
:: N.B. Need to run with call otherwise the script hangs
call activate test-environment





@echo off
