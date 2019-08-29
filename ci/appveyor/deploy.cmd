@echo on

REM Needed to enable to define OSQP_BIN within the file
@setlocal enabledelayedexpansion

IF "%APPVEYOR_REPO_TAG%" == "true" (

    IF NOT "%OSQP_VERSION%"=="%OSQP_VERSION:dev=%" (
    rem We are using a development version
    set OSQP_PACKAGE_NAME="OSQP-dev"
    ) ELSE (
    rem We are using standard version
    set OSQP_PACKAGE_NAME="OSQP"
    )
    
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

    REM Deploy to Bintray
    curl -T !OSQP_BIN!.tar.gz -ubstellato:!BINTRAY_API_KEY! -H X-Bintray-Package:!OSQP_PACKAGE_NAME! -H X-Bintray-Version:!OSQP_VERSION! -H "X-Bintray-Override: 1" https://api.bintray.com/content/bstellato/generic/!OSQP_PACKAGE_NAME!/!OSQP_VERSION!/
    if errorlevel 1 exit /b 1

    REM Publish
    curl -X POST -ubstellato:!BINTRAY_API_KEY! https://api.bintray.com/content/bstellato/generic/!OSQP_PACKAGE_NAME!/!OSQP_VERSION!/publish
    if errorlevel 1 exit /b 1

rem End of IF for appveyor tag
)

@echo off
