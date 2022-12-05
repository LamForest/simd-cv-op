#include "arm_activation.hpp"
#include <arm_neon.h>

namespace simd
{
    void naive_relu(float *in, float *out, int len)
    {
        float *src = in;
        float *dst = out;
        for (int i = 0; i < len; ++i)
        {
            *(dst++) = std::max(*(src++), 0.0f);
        }
    }

    void relu_intrinsics(float *in, float *out, int len)
    {
        assert(len % 4 == 0);

        float32x4_t v_lower_bound = vdupq_n_f32(0.0f);

        for (int i = 0; i + 3 < len; i += 4)
        {
            float32x4_t v_src = vld1q_f32(in);
            float32x4_t v_dst = vmaxq_f32(v_src, v_lower_bound);
            vst1q_f32(out, v_dst);
            out += 4;
            in += 4;
        }
    }

    void relu_intrinsics2x(float *in, float *out, int len)
    {
        assert(len % 4 == 0);

        float32x4_t v_lower_bound = vdupq_n_f32(0.0f);

        for (int i = 0; i + 3 < len; i += 8)
        {
            float32x4x2_t v_src = vld2q_f32(in);
            float32x4_t v_dst_0 = vmaxq_f32(v_src.val[0], v_lower_bound);
            float32x4_t v_dst_1 = vmaxq_f32(v_src.val[1], v_lower_bound);
            float32x4x2_t v_dst;
            v_dst.val[0] = v_dst_0;
            v_dst.val[1] = v_dst_1;

            vst2q_f32(out, v_dst);
            out += 8;
            in += 8;
        }
    }
}