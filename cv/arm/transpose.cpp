#include <arm_neon.h>

namespace simd
{
    void transpose4x4_naive(float *data)
    {
    }

    void transpose4x4_intrinsics(float *in, float *out)
    {

        float32x4x4_t transposed = vld4q_f32(in);

        vst1q_f32_x4(out, transposed);
    }
}