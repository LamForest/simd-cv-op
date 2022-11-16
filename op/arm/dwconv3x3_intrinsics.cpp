
#include "arm_op.hpp"
#include <assert.h>
#include <stdio.h>
#include <arm_neon.h>

namespace simd
{
    void naive_dwconv3x3_intrinsics(
        float *bottom, const int bottom_ch, const int bottom_w, const int bottom_h,
        float *top,
        float *kernel,
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
            float *kth_kernel = kernel + k * kernel_size;
            float32x4_t k0 = vld1q_f32(kth_kernel);
            k0 = vsetq_lane_f32(0.0f, k0, 3);
            float32x4_t k1 = vld1q_f32(kth_kernel + 3);
            k1 = vsetq_lane_f32(0.0f, k1, 3);
            float32x4_t k2 = vld1q_f32(kth_kernel + 6);
            k2 = vsetq_lane_f32(0.0f, k2, 3); //访存越界？

            float *cur_in_line = bottom + k * bottom_h * bottom_w;
            float *cur_out_line = top + k * top_h * top_w;

            int h = 0;
            for (; h + 2 < bottom_h; h += 1)
            {
                float *in_0 = cur_in_line;
                float *in_1 = cur_in_line + bottom_w;
                float *in_2 = cur_in_line + 2 * bottom_w;

                for (int w = 0; w + 2 < bottom_w; w += 1)
                {
                    float32x4_t a0 = vld1q_f32(in_0);
                    float32x4_t a1 = vld1q_f32(in_1);
                    float32x4_t a2 = vld1q_f32(in_2);

                    float32x4_t sum = vmulq_f32(a0, k0);
                    sum = vmlaq_f32(sum, a1, k1);
                    sum = vmlaq_f32(sum, a2, k2);

                    float32x2_t s = vpadd_f32(vget_high_f32(sum), vget_low_f32(sum));
                    cur_out_line[w] = vget_lane_f32(s, 0) + vget_lane_f32(s, 1);
                    in_0 += 1;
                    in_1 += 1;
                    in_2 += 1;
                }

                cur_in_line += bottom_w;
                cur_out_line += top_w;
            }
        }
        return;
    }

    void dwconv3x3_2row_intrinsics(
        const float *bottom, const int bottom_ch, const int bottom_w, const int bottom_h,
        float *top,
        const float *kernel,
        bool need_padding)
    {
        assert(false && "not implemented");
        return;
    }

    void dwconv3x3_4col_intrinsics(
        const float *bottom, const int bottom_ch, const int bottom_w, const int bottom_h,
        float *top,
        const float *kernel,
        bool need_padding)
    {
        assert(bottom_w % 4 == 0); // bottom_h只能是偶数

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

            const float *cur_in_line = bottom + k * bottom_h * bottom_w;
            float *cur_out_line = top + k * top_h * top_w;

            int h = 0;
            for (; h + 2 < bottom_h; h += 1)
            {
                const float *in_0 = cur_in_line;
                const float *in_1 = cur_in_line + bottom_w;
                const float *in_2 = cur_in_line + 2 * bottom_w;

                for (int w = 0; w + 5 < bottom_w; w += 4)
                {
                    /* row 0 */
                    float32x4_t r_left = vld1q_f32(in_0);
                    float32x4_t r_right = vld1q_f32(in_0 + 4);

                    float32x4_t r1 = vextq_f32(r_left, r_right, 1);
                    float32x4_t r2 = vextq_f32(r_left, r_right, 2);

                    float32x4_t s0 = vmulq_n_f32(r_left, k0);
                    s0 = vmlaq_n_f32(s0, r1, k1);
                    s0 = vmlaq_n_f32(s0, r2, k2);

                    /* row 1 */

                    r_left = vld1q_f32(in_1);
                    r_right = vld1q_f32(in_1 + 4);

                    r1 = vextq_f32(r_left, r_right, 1);
                    r2 = vextq_f32(r_left, r_right, 2);

                    s0 = vmlaq_n_f32(s0, r_left, k3);
                    s0 = vmlaq_n_f32(s0, r1, k4);
                    s0 = vmlaq_n_f32(s0, r2, k5);

                    /* row 2 */

                    r_left = vld1q_f32(in_2);
                    r_right = vld1q_f32(in_2 + 4);

                    r1 = vextq_f32(r_left, r_right, 1);
                    r2 = vextq_f32(r_left, r_right, 2);

                    s0 = vmlaq_n_f32(s0, r_left, k6);
                    s0 = vmlaq_n_f32(s0, r1, k7);
                    s0 = vmlaq_n_f32(s0, r2, k8);

                    vst1q_f32(cur_out_line + w, s0);

                    in_0 += 4;
                    in_1 += 4;
                    in_2 += 4;
                }

                cur_in_line += bottom_w;
                cur_out_line += top_w;
            }
        }
        return;
    }

    void dwconv3x3_2row4col_intrinsics(
        const float *bottom, const int bottom_ch, const int bottom_w, const int bottom_h,
        float *top,
        const float *kernel,
        bool need_padding)
    {
        assert(bottom_w % 4 == 0); // bottom_h只能是偶数

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

            const float *cur_in_line = bottom + k * bottom_h * bottom_w;
            const float *cur_in_line_1 = cur_in_line + bottom_w;

            float *cur_out_line = top + k * top_h * top_w;
            float *cur_out_line_1 = cur_out_line + top_w;
            int h = 0;
            for (; h + 3 < bottom_h; h += 2)
            {
                const float *in_0 = cur_in_line;
                const float *in_1 = cur_in_line + bottom_w;
                const float *in_2 = cur_in_line + 2 * bottom_w;
                const float *in_3 = cur_in_line + 3 * bottom_w;

                for (int w = 0; w + 5 < bottom_w; w += 4)
                {

                    /* row 0 */
                    float32x4_t r_left = vld1q_f32(in_0);
                    float32x4_t r_right = vld1q_f32(in_0 + 4);

                    float32x4_t r1 = vextq_f32(r_left, r_right, 1);
                    float32x4_t r2 = vextq_f32(r_left, r_right, 2);

                    float32x4_t s0 = vmulq_n_f32(r_left, k0);
                    s0 = vmlaq_n_f32(s0, r1, k1);
                    s0 = vmlaq_n_f32(s0, r2, k2);

                    /* row 1 */

                    r_left = vld1q_f32(in_1);
                    r_right = vld1q_f32(in_1 + 4);

                    r1 = vextq_f32(r_left, r_right, 1);
                    r2 = vextq_f32(r_left, r_right, 2);

                    s0 = vmlaq_n_f32(s0, r_left, k3);
                    s0 = vmlaq_n_f32(s0, r1, k4);
                    s0 = vmlaq_n_f32(s0, r2, k5);

                    float32x4_t s1 = vmulq_n_f32(r_left, k0);
                    s1 = vmlaq_n_f32(s1, r1, k1);
                    s1 = vmlaq_n_f32(s1, r2, k2);

                    /* row 2 */

                    r_left = vld1q_f32(in_2);
                    r_right = vld1q_f32(in_2 + 4);

                    r1 = vextq_f32(r_left, r_right, 1);
                    r2 = vextq_f32(r_left, r_right, 2);

                    s0 = vmlaq_n_f32(s0, r_left, k6);
                    s0 = vmlaq_n_f32(s0, r1, k7);
                    s0 = vmlaq_n_f32(s0, r2, k8);

                    s1 = vmlaq_n_f32(s1, r_left, k3);
                    s1 = vmlaq_n_f32(s1, r1, k4);
                    s1 = vmlaq_n_f32(s1, r2, k5);

                    /* row 3 */
                    r_left = vld1q_f32(in_3);
                    r_right = vld1q_f32(in_3 + 4);

                    r1 = vextq_f32(r_left, r_right, 1);
                    r2 = vextq_f32(r_left, r_right, 2);

                    s1 = vmlaq_n_f32(s1, r_left, k6);
                    s1 = vmlaq_n_f32(s1, r1, k7);
                    s1 = vmlaq_n_f32(s1, r2, k8);

                    vst1q_f32(cur_out_line + w, s0);
                    vst1q_f32(cur_out_line_1 + w, s1);

                    in_0 += 4;
                    in_1 += 4;
                    in_2 += 4;
                    in_3 += 4;
                }

                cur_in_line += 2 * bottom_w;
                cur_out_line += 2 * top_w;
                cur_out_line_1 += 2 * top_w;
            }
        }
        return;
    }

    // void dwconv3x3_2row4col_assembly(
    //     const float *bottom, const int bottom_ch, const int bottom_w, const int bottom_h,
    //     float *top,
    //     const float *kernel,
    //     bool need_padding)
    // {
    //     }
}

// 利用更多的neon寄存器，缓解interlock，但是用处不大
// void dwconv3x3_2row4col_intrinsics(
//     const float *bottom, const int bottom_ch, const int bottom_w, const int bottom_h,
//     float *top,
//     const float *kernel,
//     bool need_padding)
// {
//     assert(bottom_w % 4 == 0); // bottom_h只能是偶数

//     const int kernel_size = 9;
//     const int k_radius = 1;

//     int top_w, top_h;
//     if (need_padding)
//     {
//         top_w = bottom_w;
//         top_h = bottom_h;
//     }
//     else
//     {
//         top_w = bottom_w - 2 * k_radius;
//         top_h = bottom_h - 2 * k_radius;
//     }

//     for (int k = 0; k < bottom_ch; ++k)
//     {
//         const float *kth_kernel = kernel + k * kernel_size;
//         float k0 = kth_kernel[0], k1 = kth_kernel[1], k2 = kth_kernel[2];
//         float k3 = kth_kernel[3], k4 = kth_kernel[4], k5 = kth_kernel[5];
//         float k6 = kth_kernel[6], k7 = kth_kernel[7], k8 = kth_kernel[8];

//         const float *cur_in_line = bottom + k * bottom_h * bottom_w;
//         const float *cur_in_line_1 = cur_in_line + bottom_w;

//         float *cur_out_line = top + k * top_h * top_w;
//         float *cur_out_line_1 = cur_out_line + top_w;
//         int h = 0;
//         for (; h + 3 < bottom_h; h += 2)
//         {
//             const float *in_0 = cur_in_line;
//             const float *in_1 = cur_in_line + bottom_w;
//             const float *in_2 = cur_in_line + 2 * bottom_w;
//             const float *in_3 = cur_in_line + 3 * bottom_w;

//             for (int w = 0; w + 5 < bottom_w; w += 4)
//             {

//                 /* row 0 */
//                 float32x4_t r_left = vld1q_f32(in_0);
//                 float32x4_t r_right = vld1q_f32(in_0 + 4);

//                 float32x4_t r0_1 = vextq_f32(r_left, r_right, 1);
//                 float32x4_t r0_2 = vextq_f32(r_left, r_right, 2);

//                 float32x4_t s0 = vmulq_n_f32(r_left, k0);
//                 s0 = vmlaq_n_f32(s0, r0_1, k1);
//                 s0 = vmlaq_n_f32(s0, r0_2, k2);

//                 /* row 1 */

//                 float32x4_t r1_left = vld1q_f32(in_1);
//                 float32x4_t r1_right = vld1q_f32(in_1 + 4);

//                 float32x4_t r1_1 = vextq_f32(r1_left, r1_right, 1);
//                 float32x4_t r1_2 = vextq_f32(r1_left, r1_right, 2);

//                 s0 = vmlaq_n_f32(s0, r1_left, k3);
//                 s0 = vmlaq_n_f32(s0, r1_1, k4);
//                 s0 = vmlaq_n_f32(s0, r1_2, k5);

//                 float32x4_t s1 = vmulq_n_f32(r1_left, k0);
//                 s1 = vmlaq_n_f32(s1, r1_1, k1);
//                 s1 = vmlaq_n_f32(s1, r1_2, k2);

//                 /* row 2 */

//                 float32x4_t r2_left = vld1q_f32(in_2);
//                 float32x4_t r2_right = vld1q_f32(in_2 + 4);

//                 float32x4_t r2_1 = vextq_f32(r2_left, r2_right, 1);
//                 float32x4_t r2_2 = vextq_f32(r2_left, r2_right, 2);

//                 s0 = vmlaq_n_f32(s0, r2_left, k6);
//                 s0 = vmlaq_n_f32(s0, r2_1, k7);
//                 s0 = vmlaq_n_f32(s0, r2_2, k8);

//                 s1 = vmlaq_n_f32(s1, r2_left, k3);
//                 s1 = vmlaq_n_f32(s1, r2_1, k4);
//                 s1 = vmlaq_n_f32(s1, r2_2, k5);

//                 /* row 3 */
//                 float32x4_t r3_left = vld1q_f32(in_3);
//                 float32x4_t r3_right = vld1q_f32(in_3 + 4);

//                 float32x4_t r3_1 = vextq_f32(r3_left, r3_right, 1);
//                 float32x4_t r3_2 = vextq_f32(r3_left, r3_right, 2);

//                 s1 = vmlaq_n_f32(s1, r3_left, k6);
//                 s1 = vmlaq_n_f32(s1, r3_1, k7);
//                 s1 = vmlaq_n_f32(s1, r3_2, k8);

//                 vst1q_f32(cur_out_line + w, s0);
//                 vst1q_f32(cur_out_line_1 + w, s1);

//                 in_0 += 4;
//                 in_1 += 4;
//                 in_2 += 4;
//                 in_3 += 4;
//             }

//             cur_in_line += 2 * bottom_w;
//             cur_out_line += 2 * top_w;
//             cur_out_line_1 += 2 * top_w;
//         }
//     }
//     return;
// }