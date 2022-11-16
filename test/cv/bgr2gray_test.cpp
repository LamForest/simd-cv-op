

#include "test_common.hpp"
#include "test_utils.hpp"

#include "arm_cv.hpp"

#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>

#include <cmath>
#include <thread>
#include <iostream>

class BGR2GrayTest : public SIMDTestCase
{
public:
    virtual ~BGR2GrayTest() = default;
    virtual bool run(int precision)
    {
        bool ret = 1;
        ret &= run_();
        return ret;
    }
    bool run_()
    {
        std::vector<cv::Size> input_sizes{
            cv::Size(320, 240),   // 240p
            cv::Size(480, 360),   // 360p
            cv::Size(1280, 720),  // 720p
            cv::Size(1920, 1080), // 1080p
            cv::Size(2560, 1440), // 2k
            cv::Size(3840, 2160), // 4K
        };

        std::vector<std::tuple<int, std::string>> input_params{
            std::tuple<int, std::string>(
                CV_8UC3,
                "CV_8U_BGR")};

        cv::setNumThreads(0);

        for (int is = 0; is < input_sizes.size(); ++is)
        {
            int src_w = input_sizes[is].width;
            int src_h = input_sizes[is].height;
            for (int ip = 0; ip < input_params.size(); ++ip)
            {
                printf("\n\n");
                auto cv_dtype = std::get<0>(input_params[ip]);
                std::string name = std::get<1>(input_params[ip]);

                cv::Mat input_img(src_h, src_w, cv_dtype);
                cv::randu(input_img, 0, 255);

                cv::Mat output_gth(src_h, src_w, CV_8UC1);
                cv::cvtColor(input_img, output_gth, cv::COLOR_BGR2GRAY);

                cv::Mat output_simd(src_h, src_w, CV_8UC1);
                // simd::naive_bgr2gray(input_img.data, output_simd.data, src_h * src_w);
                simd::bgr2gray_neon_intrinsics(input_img.data, output_simd.data, src_h * src_w);

                if (!isEqualAlmost(output_gth.data, output_simd.data, src_h * src_w, 1.0f, name))
                {
                    printf("[%s]: w/h [%d,%d], Test Failed!\n",
                           name.c_str(),
                           src_w,
                           src_h);
                    return false;
                }
                else
                {
                    printf("[%s]: w/h [%d,%d], Test Okay!\n",
                           name.c_str(),
                           src_w,
                           src_h);
                }
                std::function<void()> naive_wrapper = [&]
                {
                    simd::naive_bgr2gray(input_img.data, output_simd.data, src_h * src_w);
                };
                TimeProfile(
                    std::string("naive"), 100, naive_wrapper);

                std::function<void()> opencv_wrapper = [&]
                {
                    cv::cvtColor(input_img, output_gth, cv::COLOR_BGR2GRAY);
                };
                TimeProfile(
                    std::string("opencv"), 100, opencv_wrapper);

                std::function<void()> simd_wrapper = [&]
                {
                    simd::bgr2gray_neon_intrinsics(input_img.data, output_simd.data, src_h * src_w);
                };
                TimeProfile(
                    std::string("neon"), 100, simd_wrapper);

                std::function<void()> simd_wrapper_2 = [&]
                {
                    simd::bgr2gray_neon_intrinsics_v2(input_img.data, output_simd.data, src_h * src_w);
                };
                TimeProfile(
                    std::string("neon_v2"), 100, simd_wrapper_2);
                std::cout << "---------" << std::endl;
            }
        }
        return true;
    }
};

SIMDTestSuiteRegister(BGR2GrayTest, "cv/BGR2Gray");
