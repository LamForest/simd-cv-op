
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <thread>

template <typename T>
double mean(const std::vector<T> &vec)
{
    const size_t sz = vec.size();
    // Calculate the mean
    const double mean = std::accumulate(vec.begin(), vec.end(), 0.0) / sz;
    return mean;
}

template <typename T>
double variance(const std::vector<T> &vec)
{
    const size_t sz = vec.size();
    if (sz == 1)
    {
        return 0.0;
    }

    // Calculate the mean
    const double mean = std::accumulate(vec.begin(), vec.end(), 0.0) / sz;

    // Now calculate the variance
    auto variance_func = [&mean, &sz](double accumulator, const T &val)
    {
        return accumulator + ((val - mean) * (val - mean) / (sz - 1));
    };

    return std::accumulate(vec.begin(), vec.end(), 0.0, variance_func);
}

void TimeProfile(std::string name, int run_time, std::function<void()> &func)
{
    // Warm Up
    for (int i = 0; i < 10; ++i)
    {
        func();
    }

    using namespace std::chrono;

    float time_squared_sum = 0;
    float time_sum = 0;

    //连续执行CONTINUE_RUN_TIMES次后，休息REST_TIMES * avg_time的时间
    constexpr int CONTINUE_RUN_TIMES = 30;
    constexpr int REST_TIMES = 50;

    // main loop
    for (int i = 0; i < run_time; ++i)
    {
        auto time_start = steady_clock::now();
        func();
        auto time_end = steady_clock::now();

        float cost_time = duration_cast<microseconds>(time_end - time_start).count() /
                          1000.0f;
        time_sum += cost_time;
        time_squared_sum += cost_time * cost_time;

        // rest
        if ((i + 1) % CONTINUE_RUN_TIMES == 0 && (i + 1) != run_time)
        {
            float avg_time_per_execution = time_sum / i;
            float sleep_time = avg_time_per_execution * REST_TIMES;
            std::this_thread::sleep_for(
                static_cast<std::chrono::milliseconds>(static_cast<long long>(sleep_time)));
        }
    }

    float avg_time = time_sum / run_time;
    // max防止下溢
    float std_time = std::sqrt(std::max(0.0f, time_squared_sum / run_time - avg_time * avg_time));
    printf("Function %s: mean[std] - %.4f[%.4f]ms\n",
           name.c_str(),
           avg_time,
           std_time);
}