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
            float32x4_t v_dst = vmaxq_f16(in, v_lower_bound);
            vst1q_f32(out, v_dst);
            out += 4;
        }
    }
}