#include "arm_op.hpp"
#include <assert.h>

namespace simd
{
    void naive_dwconv(
        const float *bottom, const int bottom_ch, const int bottom_w, const int bottom_h,
        float *top,
        const float *kernel, const int kernel_w,
        bool need_padding)
    {
        const int kernel_size = kernel_w * kernel_w;
        const int k_radius = kernel_w / 2;

        int top_w, top_h;
        if (need_padding)
        {
            top_w = bottom_w;
            top_h = bottom_h;
        }
        else
        {
            top_w = bottom_w - 2 * k_radius;
            top_h = bottom_h - 2 * k_radius;
        }

        for (int k = 0; k < bottom_ch; ++k)
        {
            const float *kth_kernel = kernel + k * kernel_size;
            const float *in = bottom + k * bottom_h * bottom_w;
            float *out = top + k * top_h * top_w;
            for (int h = k_radius; h < bottom_h - k_radius; ++h)
            {
                for (int w = k_radius; w < bottom_w - k_radius; ++w)
                {
                    *out = 0;
                    const float *p_kernel = kth_kernel;

                    //卷积
                    for (int kh = h - k_radius; kh <= h + k_radius; ++kh)
                    {
                        for (int kw = w - k_radius; kw <= w + k_radius; ++kw)
                        {
                            *out += (*p_kernel) * in[kh * bottom_w + kw];

                            p_kernel += 1;
                        }
                    }
                    out += 1;
                }
            }
        }
        return;
    }

    void fast_dwconv3x3(
        const float *bottom, const int bottom_ch, const int bottom_w, const int bottom_h,
        float *top,
        const float *kernel,
        bool need_padding)
    {
        assert(bottom_h % 2 == 0); // bottom_h只能是偶数

        const int kernel_size = 9;
        const int k_radius = 1;

        int top_w, top_h;
        if (need_padding)
        {
            top_w = bottom_w;
            top_h = bottom_h;
        }
        else
        {
            top_w = bottom_w - 2 * k_radius;
            top_h = bottom_h - 2 * k_radius;
        }

        for (int k = 0; k < bottom_ch; ++k)
        {
            const float *kth_kernel = kernel + k * kernel_size;
            float k0 = kth_kernel[0], k1 = kth_kernel[1], k2 = kth_kernel[2];
            float k3 = kth_kernel[3], k4 = kth_kernel[4], k5 = kth_kernel[5];
            float k6 = kth_kernel[6], k7 = kth_kernel[7], k8 = kth_kernel[8];
            const float *in = bottom + k * bottom_h * bottom_w;

            const float *cur_in_line = in;
            float *cur_out_line = top + k * top_h * top_w;

            int h = 0;
            for (; h + 3 < bottom_h; h += 2)
            {
                const float *in_0 = cur_in_line;
                const float *in_1 = cur_in_line + bottom_w;
                const float *in_2 = cur_in_line + 2 * bottom_w;
                const float *in_3 = cur_in_line + 3 * bottom_w;
                float *out_0 = cur_out_line;
                float *out_1 = cur_out_line + top_w;
                for (int w = 0; w + 2 < bottom_w; w += 1)
                {
                    float a0 = in_0[w + 0], a1 = in_0[w + 1], a2 = in_0[w + 2];
                    float a3 = in_1[w + 0], a4 = in_1[w + 1], a5 = in_1[w + 2];
                    float a6 = in_2[w + 0], a7 = in_2[w + 1], a8 = in_2[w + 2];
                    float a9 = in_3[w + 0], a10 = in_3[w + 1], a11 = in_3[w + 2];

                    out_0[w] =
                        a0 * k0 + a1 * k1 + a2 * k2 + a3 * k3 + a4 * k4 + a5 * k5 + a6 * k6 + a7 * k7 + a8 * k8;

                    out_1[w] =
                        a3 * k0 + a4 * k1 + a5 * k2 + a6 * k3 + a7 * k4 + a8 * k5 + a9 * k6 + a10 * k7 + a11 * k8;
                }

                cur_in_line += bottom_w * 2;
                cur_out_line += top_w * 2;
            }
        }
        return;
    }
}