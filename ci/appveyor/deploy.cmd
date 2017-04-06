@echo on


cd %APPVEYOR_BUILD_FOLDER%\interfaces\python\conda_recipe

conda build --python %PYTHON_VERSION% osqp --output-folder conda-bld\
if errorlevel 1 exit /b 1

anaconda -t %ANACONDA_TOKEN% upload conda-bld/**/osqp-*.tar.bz2
if errorlevel 1 exit /b 1
