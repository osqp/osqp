@echo on


cd %APPVEYOR_BUILD_FOLDER%\interfaces\python\conda_recipe

call conda build --python %PYTHON_VERSION% osqp --output-folder conda-bld\
if errorlevel 1 exit /b 1

call anaconda -t %ANACONDA_TOKEN% upload conda-bld/**/osqp-*.tar.bz2 --user oxfordcontrol
if errorlevel 1 exit /b 1



python setup.py bdist_wheel

twine
REM twine upload -u %PYPI_USERNAME% -p %PYPI_PASSWORD% dist/*


@echo off
