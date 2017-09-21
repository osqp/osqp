IF "%APPVEYOR_REPO_TAG%" == "true" (

cd %APPVEYOR_BUILD_FOLDER%\interfaces\python\conda_recipe

call conda build --python %PYTHON_VERSION% osqp --output-folder conda-bld\
if errorlevel 1 exit /b 1

call anaconda -t %ANACONDA_TOKEN% upload conda-bld/**/osqp-*.tar.bz2 --user oxfordcontrol
if errorlevel 1 exit /b 1


cd %APPVEYOR_BUILD_FOLDER%\interfaces\python
call activate test-environment
python setup.py bdist_wheel
twine upload --repository pypi --config-file ..\..\ci\pypirc -p %PYPI_PASSWORD% dist/*
REM  twine upload --repository testpypi --config-file ..\..\ci\pypirc -p %PYPI_PASSWORD% dist/*

)
