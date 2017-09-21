@echo on

:: Force symlinks on linux to work on windows (needed for python interface)
git config --global core.symlinks true
git reset --hard


:: Remove entry with sh.exe from PATH to fix error with MinGW toolchain
:: (For MinGW make to work correctly sh.exe must NOT be in your path)
:: http://stackoverflow.com/a/3870338/2288008
set PATH=%PATH:C:\Program Files\Git\usr\bin;=%


IF "%PLATFORM%"=="x86" (
    set MINGW_PATH=C:\MinGW\bin
) ELSE (
    :: Install 64bit MinGW from chocolatey
    choco install -y mingw
    set MINGW_PATH=C:\Tools\mingw64\bin
)
set PATH=%MINGW_PATH%;%PATH%


:: Activate test environment anaconda

IF "%PLATFORM%"=="x86" (
	set MINICONDA_PATH=%MINICONDA%
) ELSE (
	set MINICONDA_PATH=%MINICONDA%-%PLATFORM%
)
set PATH=%MINICONDA_PATH%;%MINICONDA_PATH%\\Scripts;%PATH%


conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda install conda-build anaconda-client
conda create -q -n test-environment python=%PYTHON_VERSION% numpy scipy pytest future
conda install -c conda-forge twine
:: N.B. Need to run with call otherwise the script hangs
call activate test-environment


:: Set environment for build if 64bit
:: N.B. Needed during conda build!
IF "%PLATFORM%"=="x64" (
call "C:\Program Files\Microsoft SDKs\Windows\v7.1\Bin\SetEnv.cmd" /x64
)


@echo off
