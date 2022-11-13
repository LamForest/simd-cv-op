

#include "test_common.hpp"
#include "test_utils.hpp"

#include "arm_op.hpp"

#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>

#include <cmath>
#include <thread>

class NaiveDwConvTest : public SIMDTestCase
{
public:
    virtual ~NaiveDwConvTest() = default;
    virtual bool run(int precision)
    {
        bool ret = 1;

        ret &= run_();
        return ret;
    }
    bool run_()
    {

        std::vector<float> kernel{
            1, 2, 3, 4, 5, 6, 7, 8, 9};

        std::vector<float> bottom{
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

        std::vector<float> gth_top{
            348, 393, 528, 573};

        std::vector<float> top(4);

        simd::naive_dwconv(bottom.data(), 1, 4, 4, top.data(), kernel.data(), 3, false);

        if (!isEqualPrecise(top.data(), gth_top.data(), gth_top.size(), "naive_dwconv"))
        {
            printf("[%s]: , Test Failed!\n",
                   "naive_dwconv 正确性验证");
            return false;
        }

        std::vector<cv::Size>
            input_sizes{
                cv::Size(4, 4),
                cv::Size(16, 16),
                cv::Size(64, 64),
                cv::Size(256, 256),
                // cv::Size(1024, 1024),
            };

        std::vector<std::tuple<int, std::string>> input_params{
            std::tuple<int, std::string>(
                8,
                "naive_dwconv ch=8"),
            std::tuple<int, std::string>(
                32,
                "naive_dwconv ch=32"),
            std::tuple<int, std::string>(
                128,
                "naive_dwconv ch=128"),
            // std::tuple<int, std::string>(
            //     512,
            //     "naive_dwconv ch=512"),
        };

        for (int is = 0; is < input_sizes.size(); ++is)
        {
            int src_w = input_sizes[is].width;
            int src_h = input_sizes[is].height;
            for (int ip = 0; ip < input_params.size(); ++ip)
            {
                printf("\n\n");
                auto ch = std::get<0>(input_params[ip]);
                std::string name = std::get<1>(input_params[ip]);

                cv::Mat kernel(ch, 3 * 3, CV_32FC1);
                cv::randu(kernel, -10000.f, 10000.f);

                cv::Mat input(ch, src_w * src_h, CV_32FC1);
                cv::randu(input, -10000.f, 10000.f);

                cv::Mat output(ch, (src_w - 2) * (src_h - 2), CV_32FC1);

                std::function<void()> wrapper = [&]
                {
                    simd::naive_dwconv(input.ptr<float>(), ch, src_w, src_h, output.ptr<float>(), kernel.ptr<float>(), 3, false);
                };
                TimeProfile(
                    name, 100, wrapper);
            }
        }
        return true;
    }
};

SIMDTestSuiteRegister(NaiveDwConvTest, "op/naive_dwconv3x3");
