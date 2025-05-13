#pragma once

#include <immintrin.h>

#include <cmath>

#ifndef __AVX512F__
#error "AVX512 support is required"
#endif

#include <cstdint>
#include <limits>

namespace simd_random {

class minstd_rand final {
 public:
  explicit minstd_rand(std::uint32_t seed) noexcept : state_(initState(seed)) {}

  auto operator()() noexcept {
    // split state into two 256-bit chunks for 64-bit processing
    const auto state_low = _mm512_extracti32x8_epi32(state_, 0);
    const auto state_high = _mm512_extracti32x8_epi32(state_, 1);

    // convert to 64 bit for multiplication
    const auto state_low_64 = _mm512_cvtepu32_epi64(state_low);
    const auto state_high_64 = _mm512_cvtepu32_epi64(state_high);
    const auto a_64 = _mm512_set1_epi64(kMultiplier);
    auto product_low = _mm512_mul_epu32(state_low_64, a_64);
    auto product_high = _mm512_mul_epu32(state_high_64, a_64);

    // compute modulus using optimized method for 2^31-1
    const auto m_64 = _mm512_set1_epi64(kModulus);
    auto compute_mod = [m_64](__m512i product) {
      const auto hi = _mm512_srli_epi64(product, 31);
      const auto lo = _mm512_and_epi64(product, m_64);
      auto sum = _mm512_add_epi64(hi, lo);

      const auto ge_mask = _mm512_cmp_epu64_mask(sum, m_64, _MM_CMPINT_GE);
      return _mm512_mask_sub_epi64(sum, ge_mask, sum, m_64);
    };

    // apply modulus to both halves
    const auto res_low = compute_mod(product_low);
    const auto res_high = compute_mod(product_high);

    // convert back to 32-bit and combine results
    const auto new_low = _mm512_cvtepi64_epi32(res_low);
    const auto new_high = _mm512_cvtepi64_epi32(res_high);

    state_ = _mm512_inserti32x8(_mm512_castsi256_si512(new_low), new_high, 1);
    return state_;
  }

  static constexpr std::uint32_t kMultiplier = 48271;
  static constexpr std::uint32_t kModulus = 0x7fffffff;

 private:
  static __m512i initState(std::uint32_t seed) noexcept {
    const auto indices =
        _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    auto seeds = _mm512_add_epi32(_mm512_set1_epi32(seed), indices);
    const auto m_vec = _mm512_set1_epi32(kModulus);
    auto lo = _mm512_and_epi32(seeds, m_vec);
    auto hi = _mm512_srli_epi32(seeds, 31);
    auto mod = _mm512_add_epi32(lo, hi);

    // handle potential overflow in modulus calculation
    const auto ge_mask = _mm512_cmp_epu32_mask(mod, m_vec, _MM_CMPINT_GE);
    mod = _mm512_mask_sub_epi32(mod, ge_mask, mod, m_vec);

    // remove zeroes
    const auto zero_mask = _mm512_cmpeq_epi32_mask(mod, _mm512_setzero_epi32());
    return _mm512_mask_add_epi32(mod, zero_mask, mod, _mm512_set1_epi32(1));
  }

 private:
  __m512i state_;
};

}  // namespace simd_random
