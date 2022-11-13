//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TEST_SIMDTEST_H
#define TEST_SIMDTEST_H

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

/** test case */
class SIMDTestCase
{
    friend class SIMDTestSuite;

public:
    /**
     * @brief deinitializer
     */
    virtual ~SIMDTestCase() = default;
    /**
     * @brief run test case with runtime precision: BackendConfig::PrecisionMode
     */
    virtual bool run(int precision) = 0;

private:
    /** case name */
    std::string name;
};

/** test suite */
class SIMDTestSuite
{
public:
    /**
     * @brief deinitializer
     */
    ~SIMDTestSuite();
    /**
     * @brief get shared instance
     * @return shared instance
     */
    static SIMDTestSuite *get();

public:
    /**
     * @brief register runable test case
     * @param test test case
     * @param name case name
     */
    void add(SIMDTestCase *test, const char *name);
    /**
     * @brief run all registered test case
     */
    static void runAll(int precision);
    /**
     * @brief run registered test case that matches in name
     * @param name case name
     */
    static void run(const char *name, int precision);

private:
    /** get shared instance */
    static SIMDTestSuite *gInstance;
    /** registered test cases */
    std::vector<SIMDTestCase *> mTests;
};

/**
 static register for test case
 */
template <class Case>
class SIMDTestRegister
{
public:
    /**
     * @brief initializer. register test case to suite.
     * @param name test case name
     */
    SIMDTestRegister(const char *name)
    {
        SIMDTestSuite::get()->add(new Case, name);
    }
    /**
     * @brief deinitializer
     */
    ~SIMDTestRegister()
    {
    }
};

#define SIMDTestSuiteRegister(Case, name) static SIMDTestRegister<Case> __r##Case(name)
#define SIMDTEST_ASSERT(x)                                        \
    {                                                             \
        int res = (x);                                            \
        if (!res)                                                 \
        {                                                         \
            SIMD_ERROR("Error for %s, %d\n", __func__, __LINE__); \
            return false;                                         \
        }                                                         \
    }

#endif
