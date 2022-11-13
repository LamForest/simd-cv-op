#pragma once

#include <stdint.h>

namespace simd
{
    void naive_bgr2gray(uint8_t *in, uint8_t *out, int len);

    void bgr2gray_neon_intrinsics(uint8_t *in, uint8_t *out, int len);

    void bgr2gray_neon_intrinsics_v2(uint8_t *in, uint8_t *out, int len);

}