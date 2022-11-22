#pragma once

#include <stdint.h>
#include <algorithm>

namespace simd
{
    void naive_relu(float *in, float *out, int len);

    void relu_intrinsics(float *in, float *out, int len);
}