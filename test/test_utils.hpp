#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <iostream>

template <typename T>
double mean(const std::vector<T> &vec);

//方差
template <typename T>
double variance(const std::vector<T> &vec);

template <typename T> //不能放在cpp中， 为何？什么情况下可以放在cpp中？
bool isEqualPrecise(T *pa, T *pb, int len, const std::string &name)
{
    int not_equal_num = 0;
    for (int i = 0; i < len; ++i)
    {
        if (pb[i] != pa[i])
        {
            not_equal_num += 1;
            std::cout << int(pa[i]) << ", " << int(pb[i]) << std::endl;
        }
    }
    if (not_equal_num == 0)
        printf("%s Precisely Equal\n", name.c_str());
    else
        printf("%s NOT Precisely Equal, num = %d\n", name.c_str(), not_equal_num);

    return not_equal_num == 0;
}

template <typename T> //不能放在cpp中， 为何？什么情况下可以放在cpp中？
bool isEqualAlmost(T *pa, T *pb, int len, float eps, const std::string &name)
{
    int not_equal_num = 0;
    for (int i = 0; i < len; ++i)
    {
        if (std::abs(pb[i] - pa[i]) > eps)
        {
            not_equal_num += 1;
            std::cout << int(pa[i]) << ", " << int(pb[i]) << std::endl;
        }
    }
    if (not_equal_num == 0)
        printf("%s Almost Equal\n", name.c_str());
    else
        printf("%s NOT Almost Equal, num = %d\n", name.c_str(), not_equal_num);

    return not_equal_num == 0;
}

void TimeProfile(std::string name, int run_time, std::function<void()> &func);
