@echo on
IF "%APPVEYOR_REPO_TAG%" == "true" (

REM Create shared library archive for Bintray only ig Python 3.6
IF "%PYTHON_VERSION%" == "3.6" (
    REM Build C libraries
    cd %APPVEYOR_BUILD_FOLDER%
    mkdir build
    cd build
    cmake -G "%CMAKE_PROJECT%" ..
    cmake --build .
    
    REM Go to output folder
    cd %APPVEYOR_BUILD_FOLDER%\build\out

    IF "%PLATFORM%" == "x86" (
        set "OSQP_DEPLOY_DIR=osqp-0.2.0.dev4-windows32"
    ) ELSE (
        set "OSQP_DEPLOY_DIR=osqp-0.2.0.dev4-windows64"
    )
    REM Create directories
    REM NB. We force expansion of the variable at execution time!
    mkdir !OSQP_DEPLOY_DIR!
    mkdir !OSQP_DEPLOY_DIR!\lib
    mkdir !OSQP_DEPLOY_DIR!\include

    REM Copy License
    xcopy ..\..\LICENSE !OSQP_DEPLOY_DIR!

    REM Copy includes
    xcopy ..\..\include\*.h !OSQP_DEPLOY_DIR!\include

    REM Copy static library
    xcopy libosqpstatic.a !OSQP_DEPLOY_DIR!\lib

    REM Copy shared library
    xcopy libosqp.dll !OSQP_DEPLOY_DIR!\lib

    REM Compress package
    7z a -ttar !OSQP_DEPLOY_DIR!.tar !OSQP_DEPLOY_DIR!
    7z a -tgzip !OSQP_DEPLOY_DIR!.tar.gz !OSQP_DEPLOY_DIR!.tar

    REM Deploy to Bintray
    curl -T !OSQP_DEPLOY_DIR!.tar.gz -ubstellato:%BINTRAY_API_KEY% -H "X-Bintray-Package:OSQP" -H "X-Bintray-Version:0.2.0.dev4" https://api.bintray.com/content/bstellato/generic/OSQP/0.2.0.dev4/
    if errorlevel 1 exit /b 1

    REM Publish
    curl -X POST -ubstellato:%BINTRAY_API_KEY% https://api.bintray.com/content/bstellato/generic/OSQP/0.2.0.dev4/publish
    if errorlevel 1 exit /b 1
)


cd %APPVEYOR_BUILD_FOLDER%\interfaces\python\conda_recipe

call conda build --python %PYTHON_VERSION% osqp --output-folder conda-bld\
if errorlevel 1 exit /b 1

call anaconda -t %ANACONDA_TOKEN% upload conda-bld/**/osqp-*.tar.bz2 --user oxfordcontrol
if errorlevel 1 exit /b 1


cd %APPVEYOR_BUILD_FOLDER%\interfaces\python
call activate test-environment
python setup.py bdist_wheel
IF "%TEST_PYPI%" == "true" (
    twine upload --repository testpypi --config-file ..\..\ci\pypirc -p %PYPI_PASSWORD% dist/*
    if errorlevel 1 exit /b 1
) ELSE (
    twine upload --repository pypi --config-file ..\..\ci\pypirc -p %PYPI_PASSWORD% dist/*
    if errorlevel 1 exit /b 1
)

REM Close parenthesis for deploying only if it is a tagged commit
)
@echo off
