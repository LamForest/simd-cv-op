#!/usr/bin/env bash

set -e
set -x

basepath=$(cd `dirname $0`/; pwd)

BUILD_DIR=${basepath}/build/command-line

rm -rf ${BUILD_DIR}
if [[ ! -d ${BUILD_DIR} ]]; then
    mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}

cmake -DBUILD_TEST=ON \
../..

make

${basepath}/build/command-line/test/run_test.out

