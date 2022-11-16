#pragma once

#include <stdint.h>

namespace simd
{
    void naive_dwconv(
        const float *bottom, const int bottom_ch, const int bottom_w, const int bottom_h,
        float *top,
        const float *kernel, const int kernel_w,
        bool need_padding);
    void naive_dwconv3x3(
        const float *bottom, const int bottom_ch, const int bottom_w, const int bottom_h,
        float *top,
        const float *kernel,
        bool need_padding);

    void naive_dwconv3x3_intrinsics(
        float *bottom, const int bottom_ch, const int bottom_w, const int bottom_h,
        float *top,
        float *kernel,
        bool need_padding);

    void dwconv3x3_2row(
        const float *bottom, const int bottom_ch, const int bottom_w, const int bottom_h,
        float *top,
        const float *kernel,
        bool need_padding);

    void dwconv3x3_2row_intrinsics(
        const float *bottom, const int bottom_ch, const int bottom_w, const int bottom_h,
        float *top,
        const float *kernel,
        bool need_padding);

    void dwconv3x3_4col(
        const float *bottom, const int bottom_ch, const int bottom_w, const int bottom_h,
        float *top,
        const float *kernel,
        bool need_padding);

    void dwconv3x3_4col_intrinsics(
        const float *bottom, const int bottom_ch, const int bottom_w, const int bottom_h,
        float *top,
        const float *kernel,
        bool need_padding);

    void dwconv3x3_2row4col_intrinsics(
        const float *bottom, const int bottom_ch, const int bottom_w, const int bottom_h,
        float *top,
        const float *kernel,
        bool need_padding);

    void dwconv3x3_2row4col_asm(float *const &src, const int &inWidth, const int &inHeight, const int &inChannel, float *const kernel,
                                float *dest, const int &outWidth, const int &outHeight, const int &outChannel);
}