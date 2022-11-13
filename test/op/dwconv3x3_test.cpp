

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

class DwConv3x3Test : public SIMDTestCase
{
public:
    virtual ~DwConv3x3Test() = default;
    virtual bool run(int precision)
    {
        bool ret = 1;

        ret &= run_();
        return ret;
    }
    bool run_()
    {
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
                "fast_dwconv3x3 ch=8"),
            std::tuple<int, std::string>(
                32,
                "fast_dwconv3x3 ch=32"),
            std::tuple<int, std::string>(
                128,
                "fast_dwconv3x3 ch=128"),
            // std::tuple<int, std::string>(
            //     512,
            //     "naive_dwconv ch=512"),
        };

        for (int is = 0; is < input_sizes.size(); ++is)
        {
            int src_w = input_sizes[is].width;
            int src_h = input_sizes[is].height;
            int out_feat_sz = (src_w - 2) * (src_h - 2);
            for (int ip = 0; ip < input_params.size(); ++ip)
            {
                printf("\n\n");
                auto ch = std::get<0>(input_params[ip]);
                std::string name = std::get<1>(input_params[ip]);

                cv::Mat kernel(ch, 3 * 3, CV_32FC1);
                cv::randu(kernel, -10000.f, 10000.f);

                cv::Mat input(ch, src_w * src_h, CV_32FC1);
                cv::randu(input, -10000.f, 10000.f);

                cv::Mat output(ch, out_feat_sz, CV_32FC1);
                cv::Mat output_gth(ch, out_feat_sz, CV_32FC1);

                simd::naive_dwconv(input.ptr<float>(), ch, src_w, src_h, output_gth.ptr<float>(), kernel.ptr<float>(), 3, false);
                simd::fast_dwconv3x3(input.ptr<float>(), ch, src_w, src_h, output.ptr<float>(), kernel.ptr<float>(), false);

                if (!isEqualPrecise(output_gth.ptr<float>(), output.ptr<float>(), ch * out_feat_sz, name))
                {
                    printf("[%s]: , Test Failed!\n",
                           "naive_dwconv 正确性验证");
                    return false;
                }

                std::function<void()> wrapper = [&]
                {
                    simd::naive_dwconv(input.ptr<float>(), ch, src_w, src_h, output.ptr<float>(), kernel.ptr<float>(), 3, false);
                };
                TimeProfile(
                    "naive_dwconv3x3", 100, wrapper);

                std::function<void()> fast_wrapper = [&]
                {
                    simd::fast_dwconv3x3(input.ptr<float>(), ch, src_w, src_h, output.ptr<float>(), kernel.ptr<float>(), false);
                };
                TimeProfile(
                    name, 100, fast_wrapper);
            }
        }
        return true;
    }
};

SIMDTestSuiteRegister(DwConv3x3Test, "op/dwconv3x3");
