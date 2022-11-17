//
//  main.cpp
//  MNN
//
//  Created by MNN on 2018/07/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "test_common.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char *argv[])
{

    // SIMDTestSuite::runAll(0);

    // SIMDTestSuite::run("cv/BGR2Gray", 0);
    // SIMDTestSuite::run("op/dwconv3x3", 0);
    SIMDTestSuite::run("cv/transpose", 0);

    return 0;
}
