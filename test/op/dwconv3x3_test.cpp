

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
                // cv::Size(4, 4),
                // cv::Size(16, 16),
                // cv::Size(64, 64),
                // cv::Size(256, 256),
                cv::Size(512, 512),
            };

        std::vector<std::tuple<int, std::string>> input_params{
            std::tuple<int, std::string>(
                8,
                "dwconv3x3 ch=8"),
            std::tuple<int, std::string>(
                32,
                "dwconv3x3 ch=32"),
            std::tuple<int, std::string>(
                128,
                "dwconv3x3 ch=128"),
            std::tuple<int, std::string>(
                512,
                "naive_dwconv ch=512"),
        };

        for (int is = 0; is < input_sizes.size(); ++is)
        {
            int src_w = input_sizes[is].width;
            int src_h = input_sizes[is].height;
            int out_feat_sz = (src_w - 2) * (src_h - 2);
            for (int ip = 0; ip < input_params.size(); ++ip)
            {
                auto ch = std::get<0>(input_params[ip]);
                std::string name = std::get<1>(input_params[ip]);
                printf("\n ------- Case : (h, w) = (%d, %d), #channel = %d, Float -------\n", src_h, src_w, ch);

                cv::Mat kernel(ch, 3 * 3, CV_32FC1);
                cv::randu(kernel, -16.f, 16.f);

                cv::Mat input(ch, src_w * src_h, CV_32FC1);
                cv::randu(input, -16.f, 16.f);

                cv::Mat output(ch, out_feat_sz, CV_32FC1);

                /* 1. naive 方法，其结果作为groundtruth */
                cv::Mat output_gth(ch, out_feat_sz, CV_32FC1);
                simd::naive_dwconv(input.ptr<float>(), ch, src_w, src_h, output_gth.ptr<float>(), kernel.ptr<float>(), 3, false);
                std::function<void()> wrapper = [&]
                {
                    simd::naive_dwconv(input.ptr<float>(), ch, src_w, src_h, output.ptr<float>(), kernel.ptr<float>(), 3, false);
                };
                TimeProfile("naive_dwconv", 100, wrapper);

                /* 2. 去掉微内核循环 的 3x3算法  */
                simd::naive_dwconv3x3(input.ptr<float>(), ch, src_w, src_h, output.ptr<float>(), kernel.ptr<float>(), false);
                if (!isEqualPrecise(output_gth.ptr<float>(), output.ptr<float>(), ch * out_feat_sz, name))
                {
                    printf("[%s]: , Test Failed!\n", "naive_dwconv3x3 正确性验证");
                    return false;
                }
                std::function<void()> naive_3x3_wrapper = [&]
                {
                    simd::naive_dwconv3x3(input.ptr<float>(), ch, src_w, src_h, output.ptr<float>(), kernel.ptr<float>(), false);
                };
                TimeProfile("naive_dwconv3x3", 100, naive_3x3_wrapper);

                /* 3. naive 3x3 方法，neon优化 */
                simd::naive_dwconv3x3_intrinsics(input.ptr<float>(), ch, src_w, src_h, output.ptr<float>(), kernel.ptr<float>(), false);
                if (!isEqualAlmost(output_gth.ptr<float>(), output.ptr<float>(), ch * out_feat_sz, 0.001, name))
                {
                    printf("[%s]: , Test Failed!\n",
                           "naive_dwconv3x3_intrinsics 正确性验证");
                    return false;
                }
                std::function<void()> naive_3x3_intrinsics_wrapper = [&]
                {
                    simd::naive_dwconv3x3_intrinsics(input.ptr<float>(), ch, src_w, src_h, output.ptr<float>(), kernel.ptr<float>(), false);
                };
                TimeProfile(
                    "naive_dwconv3x3_intrinsics", 100, naive_3x3_intrinsics_wrapper);

                /* 4. 同时做两行的3x3卷积算法  */
                simd::dwconv3x3_2row(input.ptr<float>(), ch, src_w, src_h, output.ptr<float>(), kernel.ptr<float>(), false);
                if (!isEqualPrecise(output_gth.ptr<float>(), output.ptr<float>(), ch * out_feat_sz, name))
                {
                    printf("[%s]: , Test Failed!\n", "dwconv3x3_2row 正确性验证");
                    return false;
                }
                std::function<void()> wrapper_2row = [&]
                {
                    simd::dwconv3x3_2row(input.ptr<float>(), ch, src_w, src_h, output.ptr<float>(), kernel.ptr<float>(), false);
                };
                TimeProfile("dwconv3x3_2row", 100, wrapper_2row);

                /* 4. 同时做两行的3x3卷积算法, neon intrinsics优化  */
                // simd::dwconv3x3_2row_intrinsics(input.ptr<float>(), ch, src_w, src_h, output.ptr<float>(), kernel.ptr<float>(), false);
                // if (!isEqualPrecise(output_gth.ptr<float>(), output.ptr<float>(), ch * out_feat_sz, name))
                // {
                //     printf("[%s]: , Test Failed!\n", "dwconv3x3_2row 正确性验证");
                //     return false;
                // }
                // std::function<void()> wrapper_2row_intrinsics = [&]
                // {
                //     simd::dwconv3x3_2row_intrinsics(input.ptr<float>(), ch, src_w, src_h, output.ptr<float>(), kernel.ptr<float>(), false);
                // };
                // TimeProfile("dwconv3x3_2row_intrinsics", 100, wrapper_2row_intrinsics);

                /* 同时做4列的3x3卷积  */
                simd::dwconv3x3_4col(input.ptr<float>(), ch, src_w, src_h, output.ptr<float>(), kernel.ptr<float>(), false);
                if (!isEqualPrecise(output_gth.ptr<float>(), output.ptr<float>(), ch * out_feat_sz, name))
                {
                    printf("[%s]: , Test Failed!\n", "dwconv3x3_4col 正确性验证");
                    return false;
                }
                std::function<void()> wrapper_4col = [&]
                {
                    simd::dwconv3x3_4col(input.ptr<float>(), ch, src_w, src_h, output.ptr<float>(), kernel.ptr<float>(), false);
                };
                TimeProfile(
                    "dwconv3x3_4col", 100, wrapper_4col);

                /* 同时做4列的3x3卷积, neon intrinsics 优化  */
                simd::dwconv3x3_4col_intrinsics(input.ptr<float>(), ch, src_w, src_h, output.ptr<float>(), kernel.ptr<float>(), false);
                if (!isEqualPrecise(output_gth.ptr<float>(), output.ptr<float>(), ch * out_feat_sz, name))
                {
                    printf("[%s]: , Test Failed!\n", "dwconv3x3_4col_intrinsics 正确性验证");
                    return false;
                }

                std::function<void()> wrapper_4col_intrinsics = [&]
                {
                    simd::dwconv3x3_4col_intrinsics(input.ptr<float>(), ch, src_w, src_h, output.ptr<float>(), kernel.ptr<float>(), false);
                };
                TimeProfile(
                    "dwconv3x3_4col_intrinsics", 100, wrapper_4col_intrinsics);

                /* 同时做2行和4列的3x3卷积, neon intrinsics 优化  */
                std::function<void()> wrapper_2row4col_intrinsics = [&]
                {
                    simd::dwconv3x3_2row4col_intrinsics(input.ptr<float>(), ch, src_w, src_h, output.ptr<float>(), kernel.ptr<float>(), false);
                };
                TimeProfile(
                    "dwconv3x3_2row4col_intrinsics", 100, wrapper_2row4col_intrinsics);

                /* 同时做2行和4列的3x3卷积, neon assembly 优化  */
                simd::dwconv3x3_2row4col_intrinsics(input.ptr<float>(), ch, src_w, src_h, output.ptr<float>(), kernel.ptr<float>(), false);
                if (!isEqualPrecise(output_gth.ptr<float>(), output.ptr<float>(), ch * out_feat_sz, name))
                {
                    printf("[%s]: , Test Failed!\n", "dwconv3x3_2row4col_intrinsics 正确性验证");
                    return false;
                }

                simd::dwconv3x3_2row4col_asm(input.ptr<float>(), src_w, src_h, ch, kernel.ptr<float>(), output.ptr<float>(), src_w - 2, src_h - 2, ch);
                if (!isEqualAlmost(output_gth.ptr<float>(), output.ptr<float>(), ch * out_feat_sz, 1.0f, name))
                {
                    printf("[%s]: , Test Failed!\n", "dwconv3x3_2row4col_intrinsics 正确性验证");
                    return false;
                }

                std::function<void()> wrapper_2row4col_asm = [&]
                {
                    simd::dwconv3x3_2row4col_asm(input.ptr<float>(), src_w, src_h, ch, kernel.ptr<float>(), output.ptr<float>(), src_w - 2, src_h - 2, ch);
                };
                TimeProfile(
                    "dwconv3x3_2row4col_asm", 100, wrapper_2row4col_asm);
            }
        }
        return true;
    }
};

SIMDTestSuiteRegister(DwConv3x3Test, "op/dwconv3x3");
