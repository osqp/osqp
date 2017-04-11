IF "%APPVEYOR_REPO_TAG%" == "true" (

cd %APPVEYOR_BUILD_FOLDER%\interfaces\python\conda_recipe

call conda build --python %PYTHON_VERSION% osqp --output-folder conda-bld\
if errorlevel 1 exit /b 1

call anaconda -t %ANACONDA_TOKEN% upload conda-bld/**/osqp-*.tar.bz2 --user oxfordcontrol
if errorlevel 1 exit /b 1


:: Specify account details for PyPI
echo [distutils]                                      > %USERPROFILE%\\.pypirc
echo index-servers = testpypi                        >> %USERPROFILE%\\.pypirc
echo [testpypi]                                      >> %USERPROFILE%\\.pypirc
echo repository=https://testpypi.python.org/pypi     >> %USERPROFILE%\\.pypirc
echo username=bstellato                              >> %USERPROFILE%\\.pypirc
echo password=%PYPIPASSWORD%                         >> %USERPROFILE%\\.pypirc


cd %APPVEYOR_BUILD_FOLDER%\interfaces\python
call activate test-environment
python setup.py bdist_wheel
twine upload dist/*

)
