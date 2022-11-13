#pragma once

#include <stdint.h>

namespace simd
{
    void naive_dwconv(
        const float *bottom, const int bottom_ch, const int bottom_w, const int bottom_h,
        float *top,
        const float *kernel, const int kernel_w,
        bool need_padding);

    void fast_dwconv3x3(
        const float *bottom, const int bottom_ch, const int bottom_w, const int bottom_h,
        float *top,
        const float *kernel,
        bool need_padding);
}