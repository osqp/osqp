from __future__ import print_function
from builtins import input
import os
import zipfile
from platform import system
import osqp  # Need python OSQP module to run
import shutil as sh
from subprocess import call


def zipdir(path):

    zipf = zipfile.ZipFile('%s.zip' % path, 'w', zipfile.ZIP_DEFLATED)

    # zipf is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            zipf.write(os.path.join(root, file))

    zipf.close()


if __name__ == '__main__':

    # Get oeprative system
    if system() == 'Windows':
        platform = 'windows'
        matlab_ext = 'mexw64'
    elif system() == 'Linux':
        platform = 'linux'
        matlab_ext = 'mexa64'
    elif system() == 'Darwin':
        platform = 'mac'
        matlab_ext = 'mexmaci64'

    # Get OSQP version
    m = osqp.OSQP()
    version = m.version()

    # Get package name
    package_name = "osqp-%s-matlab-%s" % (version, platform)

    # Create build directory
    if os.path.exists(package_name):
        sh.rmtree(package_name)
    os.makedirs(package_name)

    # Copy folders
    folders_to_copy = ['codegen', 'unittests']
    for folder_name in folders_to_copy:
        sh.copytree(os.path.join('..', folder_name),
                    os.path.join(package_name, folder_name))

    # Copy interface files
    files_to_copy = ['osqp_mex.%s' % matlab_ext,
                     'osqp.m']

    for file_name in files_to_copy:
        sh.copy(os.path.join('..', file_name),
                os.path.join(package_name))

    # Create zip file
    zipdir(package_name)

    # Upload on github
    print("We are ready to upload the package to GitHub")
    gh_token = input("GitHub token: ")

    call(['github-release', 'upload',
          '--user', 'oxfordcontrol',
          '--security-token', gh_token,
          '--repo', 'osqp',
          '--tag', 'v%s' % version,
          '--name', package_name + '.zip',
          '--file', package_name + '.zip'])
