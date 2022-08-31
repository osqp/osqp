#!/bin/sh
########################################################################
# Usage: install.sh <target_dir>
########################################################################

if [ $# -ne 1 ]
then
    echo "Usage: install.sh <target_dir>"
    exit 1
fi

TARGET_DIR=$1

if [ ! -d $TARGET_DIR ]
then
    mkdir -p $TARGET_DIR
fi

cd `dirname $0`

if [ ! -d build ]
then
    mkdir build
fi

cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=${TARGET_DIR} -G "Unix Makefiles" ..
cmake --build .
cmake --build . --target install
exit 0
