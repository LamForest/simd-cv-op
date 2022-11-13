#include "arm_cv.hpp"
#include <cmath>
#include <arm_neon.h>

namespace simd
{
    void naive_bgr2gray(uint8_t *in, uint8_t *out, int len)
    {
        float r = 0.299, g = 0.587, b = 0.114;
        for (int i = 0; i < len; ++i)
        {
            out[i] = static_cast<uint8_t>(std::round(in[i * 3] * b + in[i * 3 + 1] * g + in[i * 3 + 2] * r));
        }
        return;
    }

    void bgr2gray_neon_intrinsics(uint8_t *in, uint8_t *out, int len)
    {
        /*
        NEON 无法做不同类型的操作数的运算
        故要将 u8的像素值转为fp16/fp32，或将fp32的参数(0.114, 0.587, 0.299)转为u8/u16
        一般而言，转为整型进行计算会更快

        这里选择转为u8进行计算

        */
        uint8x8_t paramb = vdup_n_u8(29);  // 0.114 * 255
        uint8x8_t paramg = vdup_n_u8(150); // 0.587 * 255
        uint8x8_t paramr = vdup_n_u8(77);  // 0.299 * 255

        uint8_t *src = in;
        uint8_t *dst = out;
        int i = 0;
#ifdef __ARM_NEON
        for (; i + 8 <= len; i += 8)
        {
            uint8x8x3_t v_bgr = vld3_u8(src);

            uint8x16_t v_sum = vmull_u8(v_bgr.val[0], paramb);
            v_sum = vmlal_u8(v_sum, v_bgr.val[1], paramg);
            v_sum = vmlal_u8(v_sum, v_bgr.val[2], paramr);

            uint8x8_t v_gray = vshrn_n_u16(v_sum, 8);

            vst1_u8(dst, v_gray);

            src += 24;
            dst += 8;
        }
#endif
        for (; i < len; i += 1)
        {
            dst[i] = static_cast<uint8_t>((src[0] * 29 + src[1] * 150 + src[2] * 76) >> 8);
            src += 3;
            dst += 1;
        }
    }

    void bgr2gray_neon_intrinsics_v2(uint8_t *in, uint8_t *out, int len)
    {

        uint8x8_t paramb = vdup_n_u8(29);  // 0.114 * 255
        uint8x8_t paramg = vdup_n_u8(150); // 0.587 * 255
        uint8x8_t paramr = vdup_n_u8(77);  // 0.299 * 255

        uint8_t *src = in;
        uint8_t *dst = out;
        int i = 0;
#ifdef __ARM_NEON
        for (; i + 16 <= len; i += 16)
        {
            uint8x16x3_t v_bgr = vld3q_u8(src); // load 16 (b,g,r) pixel

            uint8x16_t v_high_sum = vmull_u8(vget_high_u8(v_bgr.val[0]), paramb);
            v_high_sum = vmlal_u8(v_high_sum, vget_high_u8(v_bgr.val[1]), paramg);
            v_high_sum = vmlal_u8(v_high_sum, vget_high_u8(v_bgr.val[2]), paramr);

            uint8x16_t v_low_sum = vmull_u8(vget_low_u8(v_bgr.val[0]), paramb);
            v_low_sum = vmlal_u8(v_low_sum, vget_low_u8(v_bgr.val[1]), paramg);
            v_low_sum = vmlal_u8(v_low_sum, vget_low_u8(v_bgr.val[2]), paramr);

            uint8x8_t v_high_gray = vshrn_n_u16(v_high_sum, 8);
            uint8x8_t v_low_gray = vshrn_n_u16(v_low_sum, 8);

            uint8x16_t v_gray = vcombine_u8(v_low_gray, v_high_gray);

            vst1q_u8(dst, v_gray); // write 16 grayscale pixel

            src += 48;
            dst += 16;
        }
#endif
        for (; i < len; i += 1)
        {
            dst[i] = static_cast<uint8_t>((src[0] * 29 + src[1] * 150 + src[2] * 76) >> 8);
            src += 3;
            dst += 1;
        }
    }

}
