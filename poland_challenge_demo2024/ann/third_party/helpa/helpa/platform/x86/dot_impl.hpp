#pragma once

#if defined(__AVX2__)

#include <immintrin.h>

#include "ann/third_party/helpa/helpa/dot.hpp"
#include "ann/third_party/helpa/helpa/platform/x86/utils.hpp"
#include "ann/third_party/helpa/helpa/ref/dot_ref.hpp"
#include "ann/third_party/helpa/helpa/types.hpp"

namespace helpa {

inline float
dot_fp32_fp32(const float* x, const float* y, const int32_t d) {
    int32_t da = d / 16 * 16;
    return dota_fp32_fp32(x, y, da) + dot_fp32_fp32_ref(x + da, y + da, d - da);
}

inline float
dot_fp32_fp16(const float* x, const fp16* y, const int32_t d) {
    int32_t da = d / 16 * 16;
    return dota_fp32_fp16(x, y, da) + dot_fp32_fp16_ref(x + da, y + da, d - da);
}

inline float
dot_fp16_fp16(const fp16* x, const fp16* y, const int32_t d) {
    int32_t da = d / 32 * 32;
    return dota_fp16_fp16(x, y, da) + dot_fp16_fp16_ref(x + da, y + da, d - da);
}

inline float
dot_fp32_bf16(const float* x, const bf16* y, const int32_t d) {
    int32_t da = d / 16 * 16;
    return dota_fp32_bf16(x, y, da) + dot_fp32_bf16_ref(x + da, y + da, d - da);
}

inline float
dot_bf16_bf16(const bf16* x, const bf16* y, const int32_t d) {
    int32_t da = d / 32 * 32;
    return dota_bf16_bf16(x, y, da) + dot_bf16_bf16_ref(x + da, y + da, d - da);
}

inline int32_t
dot_u8_s8(const uint8_t* x, const int8_t* y, const int32_t d) {
    int32_t da = d / 64 * 64;
    return dota_u8_s8(x, y, da) + dot_u8_s8_ref(x + da, y + da, d - da);
}
inline int32_t
dot_s8_s8(const int8_t* x, const int8_t* y, const int32_t d) {
    int32_t da = d / 64 * 64;
    return dota_s8_s8(x, y, da) + dot_s8_s8_ref(x + da, y + da, d - da);
}

inline float
dota_fp32_fp32(const float* x, const float* y, const int32_t d) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (int32_t i = 0; i < d; i += 16) {
        auto xx = _mm512_loadu_ps(x + i);
        auto yy = _mm512_loadu_ps(y + i);
        sum = _mm512_fmadd_ps(xx, yy, sum);
    }
    return -reduce_add_f32x16(sum);
#else
    __m256 sum = _mm256_setzero_ps();
    for (int32_t i = 0; i < d; i += 8) {
        auto xx = _mm256_loadu_ps(x + i);
        auto yy = _mm256_loadu_ps(y + i);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(xx, yy));
    }
    return -reduce_add_f32x8(sum);
#endif
}

inline float
dota_fp32_fp16(const float* x, const fp16* y, const int32_t d) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        auto xx = _mm512_loadu_ps(x + i);
        auto zz = _mm256_loadu_si256((__m256i*)(y + i));
        auto yy = _mm512_cvtph_ps(zz);
        sum = _mm512_fmadd_ps(xx, yy, sum);
    }
    return -reduce_add_f32x16(sum);
#else
    __m256 sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        {
            auto xx = _mm256_loadu_ps(x + i);
            auto zz = _mm_loadu_si128((__m128i*)(y + i));
            auto yy = _mm256_cvtph_ps(zz);
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(xx, yy));
        }
        {
            auto xx = _mm256_loadu_ps(x + i + 8);
            auto zz = _mm_loadu_si128((__m128i*)(y + i + 8));
            auto yy = _mm256_cvtph_ps(zz);
            sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(xx, yy));
        }
    }
    sum1 = _mm256_add_ps(sum1, sum2);
    return -reduce_add_f32x8(sum1);
#endif
}

inline float
dota_fp16_fp16(const fp16* x, const fp16* y, const int32_t d) {
#if defined(USE_AVX512FP16) && defined(__AVX512FP16__)
    auto sum = _mm512_setzero_ph();
    for (int i = 0; i < d; i += 32) {
        auto xx = _mm512_loadu_ph(x + i);
        auto yy = _mm512_loadu_ph(y + i);
        sum = _mm512_fmadd_ph(xx, yy, sum);
    }
    return -reduce_add_f16x32(sum);
#elif defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        auto xxx = _mm256_loadu_si256((__m256i*)(x + i));
        auto xx = _mm512_cvtph_ps(xxx);
        auto zz = _mm256_loadu_si256((__m256i*)(y + i));
        auto yy = _mm512_cvtph_ps(zz);
        sum = _mm512_fmadd_ps(xx, yy, sum);
    }
    return -reduce_add_f32x16(sum);
#else
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < d; i += 8) {
        auto xxx = _mm_loadu_si128((__m128i*)(x + i));
        auto xx = _mm256_cvtph_ps(xxx);
        auto zz = _mm_loadu_si128((__m128i*)(y + i));
        auto yy = _mm256_cvtph_ps(zz);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(xx, yy));
    }
    return -reduce_add_f32x8(sum);
#endif
}

inline float
dota_fp32_bf16(const float* x, const bf16* y, const int32_t d) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        auto xx = _mm512_loadu_ps(x + i);
        auto zz = _mm256_loadu_si256((__m256i*)(y + i));
        auto yy = _mm512_cvtepu16_epi32(zz);
        yy = _mm512_slli_epi32(yy, 16);
        sum = _mm512_fmadd_ps(xx, (__m512)yy, sum);
    }
    return -reduce_add_f32x16(sum);
#else
    __m256 sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        {
            auto xx = _mm256_loadu_ps(x + i);
            auto zz = _mm_loadu_si128((__m128i*)(y + i));
            auto yy = _mm256_cvtepu16_epi32(zz);
            yy = _mm256_slli_epi32(yy, 16);
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(xx, (__m256)yy));
        }
        {
            auto xx = _mm256_loadu_ps(x + i + 8);
            auto zz = _mm_loadu_si128((__m128i*)(y + i + 8));
            auto yy = _mm256_cvtepu16_epi32(zz);
            yy = _mm256_slli_epi32(yy, 16);
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(xx, (__m256)yy));
        }
    }
    sum1 = _mm256_add_ps(sum1, sum2);
    return -reduce_add_f32x8(sum1);
#endif
}

inline float
dota_bf16_bf16(const bf16* x, const bf16* y, const int32_t d) {
#if defined(USE_AVX512BF16) && defined(__AVX512BF16__)
    auto sum = _mm512_setzero_ps();
    for (int i = 0; i < d; i += 32) {
        auto xx = (__m512bh)_mm512_loadu_si512(x + i);
        auto yy = (__m512bh)_mm512_loadu_si512(y + i);
        sum = _mm512_dpbf16_ps(sum, xx, yy);
    }
    return -reduce_add_f32x16(sum);
#elif defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        auto xxx = _mm256_loadu_si256((__m256i*)(x + i));
        auto xx = _mm512_cvtepu16_epi32(xxx);
        xx = _mm512_slli_epi32(xx, 16);
        auto zz = _mm256_loadu_si256((__m256i*)(y + i));
        auto yy = _mm512_cvtepu16_epi32(zz);
        yy = _mm512_slli_epi32(yy, 16);
        sum = _mm512_fmadd_ps((__m512)xx, (__m512)yy, sum);
    }
    return -reduce_add_f32x16(sum);
#else
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < d; i += 8) {
        auto xxx = _mm_loadu_si128((__m128i*)(x + i));
        auto xx = _mm256_cvtepu16_epi32(xxx);
        xx = _mm256_slli_epi32(xx, 16);
        auto zz = _mm_loadu_si128((__m128i*)(y + i));
        auto yy = _mm256_cvtepu16_epi32(zz);
        yy = _mm256_slli_epi32(yy, 16);
        sum = _mm256_add_ps(sum, _mm256_mul_ps((__m256)xx, (__m256)yy));
    }
    return -reduce_add_f32x8(sum);
#endif
}

inline int32_t
dota_s8_s8(const int8_t* x, const int8_t* y, const int32_t d) {
    auto sum = _mm256_setzero_si256();
    for (int i = 0; i < d; i += 32) {
        auto xx = _mm256_loadu_si256((__m256i*)(x + i));
        auto yy = _mm256_loadu_si256((__m256i*)(y + i));
        auto axx = _mm256_sign_epi8(xx, xx);
        auto syy = _mm256_sign_epi8(yy, xx);
        sum = dp_u8s8x32(sum, axx, syy);
    }
    return -reduce_add_i32x8(sum);
}

inline int32_t
dota_u8_s8_256(const uint8_t* x, const int8_t* y) {
    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();

    // Load 32-byte chunks of x and y into __m256i registers
    auto xx0 = _mm256_load_si256((__m256i*)(x));
    auto xx1 = _mm256_load_si256((__m256i*)(x + 32));
    auto xx2 = _mm256_load_si256((__m256i*)(x + 64));
    auto xx3 = _mm256_load_si256((__m256i*)(x + 96));
    auto yy0 = _mm256_load_si256((__m256i*)(y));
    auto yy1 = _mm256_load_si256((__m256i*)(y + 32));
    auto yy2 = _mm256_load_si256((__m256i*)(y + 64));
    auto yy3 = _mm256_load_si256((__m256i*)(y + 96));

    // Convert unsigned bytes in x to signed 16-bit integers
    auto x0_16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(xx0));
    auto x1_16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(xx0, 1));
    auto x2_16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(xx1));
    auto x3_16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(xx1, 1));
    auto x4_16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(xx2));
    auto x5_16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(xx2, 1));
    auto x6_16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(xx3));
    auto x7_16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(xx3, 1));

    // Convert signed bytes in y to signed 16-bit integers
    auto y0_16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(yy0));
    auto y1_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(yy0, 1));
    auto y2_16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(yy1));
    auto y3_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(yy1, 1));
    auto y4_16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(yy2));
    auto y5_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(yy2, 1));
    auto y6_16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(yy3));
    auto y7_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(yy3, 1));

    // Multiply and accumulate
    sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(x0_16, y0_16));
    sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(x1_16, y1_16));
    sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(x2_16, y2_16));
    sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(x3_16, y3_16));
    sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(x4_16, y4_16));
    sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(x5_16, y5_16));
    sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(x6_16, y6_16));
    sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(x7_16, y7_16));

    // Sum all results into a single 256-bit vector
    sum0 = _mm256_add_epi32(sum0, sum1);
    sum2 = _mm256_add_epi32(sum2, sum3);
    sum0 = _mm256_add_epi32(sum0, sum2);

    // Horizontal sum to get the final result
    int32_t result[8];
    _mm256_store_si256((__m256i*)result, sum0);
    int32_t total = 0;
    for (int i = 0; i < 8; i++) {
        total += result[i];
    }

    return -total;
}
// inline int32_t
// dota_u8_s8_256(const uint8_t* x, const int8_t* y) {
//     __m512i sum0 = _mm512_setzero_si512();
//     __m512i sum1 = _mm512_setzero_si512();
//     auto xx0 = _mm512_load_si512(x);
//     auto xx1 = _mm512_load_si512(x + 64);
//     auto xx2 = _mm512_load_si512(x + 128);
//     auto xx3 = _mm512_load_si512(x + 192);
//     auto yy0 = _mm512_load_si512(y);
//     auto yy1 = _mm512_load_si512(y + 64);
//     auto yy2 = _mm512_load_si512(y + 128);
//     auto yy3 = _mm512_load_si512(y + 192);
//     asm("vpdpbusd %1, %2, %0" : "+x"(sum0) : "mx"(xx0), "x"(yy0));
//     asm("vpdpbusd %1, %2, %0" : "+x"(sum1) : "mx"(xx1), "x"(yy1));
//     asm("vpdpbusd %1, %2, %0" : "+x"(sum0) : "mx"(xx2), "x"(yy2));
//     asm("vpdpbusd %1, %2, %0" : "+x"(sum1) : "mx"(xx3), "x"(yy3));
//     sum0 = _mm512_add_epi32(sum0, sum1);
//     return -_mm512_reduce_add_epi32(sum0);
// }

inline int32_t
dota_u8_s8(const uint8_t* x, const int8_t* y, const int32_t d) {
#if defined(__AVX512VNNI__)
    __m512i sum = _mm512_setzero_epi32();
    for (int i = 0; i < d; i += 64) {
        auto xx = _mm512_loadu_si512(x + i);
        auto yy = _mm512_loadu_si512(y + i);
        // GCC bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=94663
        // sum = _mm512_dpbusd_epi32(sum, t, t);
        asm("vpdpbusd %1, %2, %0" : "+x"(sum) : "mx"(xx), "x"(yy));
    }
    return -reduce_add_i32x16(sum);
#else
    __m256i sum1 = _mm256_setzero_si256(), sum2 = _mm256_setzero_si256();
    for (int i = 0; i < d; i += 64) {
        {
            auto xx = _mm256_loadu_si256((__m256i*)(x + i));
            auto yy = _mm256_loadu_si256((__m256i*)(y + i));
            auto tmp = _mm256_maddubs_epi16(xx, yy);
            sum1 = _mm256_add_epi32(sum1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(tmp, 0)));
            sum1 = _mm256_add_epi32(sum1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(tmp, 1)));
        }
        {
            auto xx = _mm256_loadu_si256((__m256i*)(x + i + 32));
            auto yy = _mm256_loadu_si256((__m256i*)(y + i + 32));
            auto tmp = _mm256_maddubs_epi16(xx, yy);
            sum2 = _mm256_add_epi32(sum2, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(tmp, 0)));
            sum2 = _mm256_add_epi32(sum2, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(tmp, 1)));
        }
    }
    sum1 = _mm256_add_epi32(sum1, sum2);
    return -reduce_add_i32x8(sum1);
#endif
}

inline int32_t
dota_u4_u4(const uint8_t* x, const uint8_t* y, const int32_t d) {
#if defined(__AVX512VNNI__)
    __m512i sum1 = _mm512_setzero_epi32(), sum2 = _mm512_setzero_epi32();
    __m512i mask = _mm512_set1_epi8(0xf);
    for (int i = 0; i < d; i += 128) {
        auto xx = _mm512_loadu_si512((__m512i*)(x + i / 2));
        auto yy = _mm512_loadu_si512((__m512i*)(y + i / 2));
        auto xx1 = _mm512_and_si512(xx, mask);
        auto xx2 = _mm512_and_si512(_mm512_srli_epi16(xx, 4), mask);
        auto yy1 = _mm512_and_si512(yy, mask);
        auto yy2 = _mm512_and_si512(_mm512_srli_epi16(yy, 4), mask);
        // sum1 = _mm512_dpbusd_epi32(sum1, d1, d1);
        // sum2 = _mm512_dpbusd_epi32(sum2, d2, d2);
        asm("vpdpbusd %1, %2, %0" : "+x"(sum1) : "mx"(xx1), "x"(yy1));
        asm("vpdpbusd %1, %2, %0" : "+x"(sum2) : "mx"(xx2), "x"(yy2));
    }
    sum1 = _mm512_add_epi32(sum1, sum2);
    return -reduce_add_i32x16(sum1);
#else
    __m256i sum1 = _mm256_setzero_si256(), sum2 = _mm256_setzero_si256();
    __m256i mask = _mm256_set1_epi8(0xf);
    for (int i = 0; i < d; i += 64) {
        auto xx = _mm256_loadu_si256((__m256i*)(x + i / 2));
        auto yy = _mm256_loadu_si256((__m256i*)(y + i / 2));
        auto xx1 = _mm256_and_si256(xx, mask);
        auto xx2 = _mm256_and_si256(_mm256_srli_epi16(xx, 4), mask);
        auto yy1 = _mm256_and_si256(yy, mask);
        auto yy2 = _mm256_and_si256(_mm256_srli_epi16(yy, 4), mask);
        auto dot1 = _mm256_maddubs_epi16(xx1, yy1);
        auto dot2 = _mm256_maddubs_epi16(xx2, yy2);
        auto dot11 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(dot1, 0));
        auto dot21 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(dot2, 0));
        auto dot12 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(dot1, 1));
        auto dot22 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(dot2, 1));
        sum1 = _mm256_add_epi32(sum1, _mm256_add_epi32(dot11, dot12));
        sum2 = _mm256_add_epi32(sum1, _mm256_add_epi32(dot21, dot22));
    }
    sum1 = _mm256_add_epi32(sum1, sum2);
    return -reduce_add_i32x8(sum1);
#endif
}

}  // namespace helpa

#endif
