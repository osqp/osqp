IF "%APPVEYOR_REPO_TAG%" == "true" (

REM Create shared library archive for Bintray only ig Python 3.6
if "%PYTHON_VERSION% == "3.6" (
    cd %APPVEYOR_BUILD_FOLDER%\build\out

    if "%PLATFORM%"=="x86" (
        set OSQP_DEPLOY_DIR=osqp-0.1.3-windows32
    ) ELSE (
        set OSQP_DEPLOY_DIR=osqp-0.1.3-windows64
    )
    REM Create directories
    mkdir %OSQP_DEPLOY_DIR%
    mkdir %OSQP_DEPLOY_DIR%\lib
    mkdir %OSQP_DEPLOY_DIR%\include

    REM Copy includes
    xcopy ..\..\include\* %OSQP_DEPLOY_DIR%\include

    REM Copy shared library
    xcopy libosqp.dll %OSQP_DEPLOY_DIR%\lib\

    REM Compress package
    7z a -ttar %OSQP_DEPLOY_DIR%.tar %OSQP_DEPLOY_DIR%\
    7z a -tgzip %OSQP_DEPLOY_DIR%.tar.gz %OSQP_DEPLOY_DIR%.tar

    REM Deploy to Bintray
    curl -T %OSQP_DEPLOY_DIR%.tar.gz -ubstellato:%BINTRAY_API_KEY% -H "X-Bintray-Package:OSQP" -H "X-Bintray-Version:0.1.3" https://api.bintray.com/content/bstellato/generic/OSQP/0.1.3/

)


cd %APPVEYOR_BUILD_FOLDER%\interfaces\python\conda_recipe

call conda build --python %PYTHON_VERSION% osqp --output-folder conda-bld\
if errorlevel 1 exit /b 1

call anaconda -t %ANACONDA_TOKEN% upload conda-bld/**/osqp-*.tar.bz2 --user oxfordcontrol
if errorlevel 1 exit /b 1


cd %APPVEYOR_BUILD_FOLDER%\interfaces\python
call activate test-environment
python setup.py bdist_wheel
REM  twine upload --repository pypi --config-file ..\..\ci\pypirc -p %PYPI_PASSWORD% dist/*
twine upload --repository testpypi --config-file ..\..\ci\pypirc -p %PYPI_PASSWORD% dist/*

)
