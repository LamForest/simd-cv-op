//
//  SIMDTestSuite.cpp
//  SIMD
//
//  Created by SIMD on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "test_common.hpp"
#include <stdlib.h>

SIMDTestSuite *SIMDTestSuite::gInstance = NULL;

SIMDTestSuite *SIMDTestSuite::get()
{
    if (gInstance == NULL)
        gInstance = new SIMDTestSuite;
    return gInstance;
}

SIMDTestSuite::~SIMDTestSuite()
{
    for (int i = 0; i < mTests.size(); ++i)
    {
        delete mTests[i];
    }
    mTests.clear();
}

void SIMDTestSuite::add(SIMDTestCase *test, const char *name)
{
    test->name = name;
    mTests.push_back(test);
}

void SIMDTestSuite::run(const char *key, int precision)
{
    if (key == NULL || strlen(key) == 0)
        return;

    auto suite = SIMDTestSuite::get();
    std::string prefix = key;
    std::vector<std::string> wrongs;
    size_t runUnit = 0;
    for (int i = 0; i < suite->mTests.size(); ++i)
    {
        SIMDTestCase *test = suite->mTests[i];
        if (test->name.find(prefix) == 0)
        {
            runUnit++;
            printf("\trunning %s.\n", test->name.c_str());
            auto res = test->run(precision);
            if (!res)
            {
                wrongs.emplace_back(test->name);
            }
        }
    }
    if (runUnit == 0)
    {
        printf("!!! No tests Execute.\n");
    }

    if (wrongs.empty() && runUnit > 0)
    {
        printf("√√√ all <%s> tests passed.\n", key);
    }
    for (auto &wrong : wrongs)
    {
        printf("Error: %s\n", wrong.c_str());
    }
    printf("### Wrong/Total: %zu / %zu ###\n", wrongs.size(), runUnit);
}

void SIMDTestSuite::runAll(int precision)
{
    auto suite = SIMDTestSuite::get();
    std::vector<std::string> wrongs;
    for (int i = 0; i < suite->mTests.size(); ++i)
    {
        SIMDTestCase *test = suite->mTests[i];
        if (test->name.find("speed") != std::string::npos)
        {
            // Don't test for speed because cost
            continue;
        }
        if (test->name.find("model") != std::string::npos)
        {
            // Don't test for model because need resource
            continue;
        }
        printf("\trunning %s.\n", test->name.c_str());
        auto res = test->run(precision);
        if (!res)
        {
            wrongs.emplace_back(test->name);
        }
    }
    if (wrongs.empty())
    {
        printf("√√√ all tests passed.\n");
    }
    for (auto &wrong : wrongs)
    {
        printf("Error: %s\n", wrong.c_str());
    }
    printf("### Wrong/Total: %zu / %zu ###\n", wrongs.size(), suite->mTests.size());
}
