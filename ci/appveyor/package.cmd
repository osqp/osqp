@echo on

REM Needed to enable to define OSQP_BIN within the file
@setlocal enabledelayedexpansion

REM Build C libraries
cd %APPVEYOR_BUILD_FOLDER%
del /F /Q build
mkdir build
cd build
cmake -G "%CMAKE_PROJECT%" ..
cmake --build .

REM Go to output folder
cd %APPVEYOR_BUILD_FOLDER%\build\out

IF "%PLATFORM%" == "x86" (
    set OSQP_BIN="osqp-!OSQP_VERSION!-windows32"
) ELSE (
    set OSQP_BIN="osqp-!OSQP_VERSION!-windows64"
)
REM Create directories
REM NB. We force expansion of the variable at execution time!
mkdir !OSQP_BIN!
mkdir !OSQP_BIN!\lib
mkdir !OSQP_BIN!\include

REM Copy License
xcopy ..\..\LICENSE !OSQP_BIN!

REM Copy includes
xcopy ..\..\include\*.h !OSQP_BIN!\include

REM Copy static library
xcopy libosqp.a !OSQP_BIN!\lib

REM Copy shared library
xcopy libosqp.dll !OSQP_BIN!\lib

REM Compress package
7z a -ttar !OSQP_BIN!.tar !OSQP_BIN!
7z a -tgzip !OSQP_BIN!.tar.gz !OSQP_BIN!.tar

rem Copy archive to main folder
mkdir %APPVEYOR_BUILD_FOLDER%\dist
xcopy !OSQP_BIN!.tar.gz %APPVEYOR_BUILD_FOLDER%\dist
cd %APPVEYOR_BUILD_FOLDER%

@echo off
