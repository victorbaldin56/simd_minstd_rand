#pragma once

#ifndef __AVX512F__
#error "AVX512 support is required"
#endif

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <random>

namespace simd_random {

class minstd_rand final {
 public:
  explicit minstd_rand(std::uint32_t seed = kDefaultSeed) noexcept
      : state_(initState(seed)) {}

  auto operator()() noexcept {
    auto res = state_;
    update();
    return res;
  }

  static constexpr std::uint32_t kVecMultiplier =
      1098894339;  // a^16 mod m where a=48271, m=2147483647
  static constexpr std::uint32_t kModulus = 0x7fffffff;
  static constexpr std::uint32_t kDefaultSeed = 1;

 private:
  void update() noexcept {
    // split state into two 256-bit chunks for processing
    const auto state_low = _mm512_extracti32x8_epi32(state_, 0);
    const auto state_high = _mm512_extracti32x8_epi32(state_, 1);

    // convert to 64-bit for multiplication
    const auto state_low_64 = _mm512_cvtepi32_epi64(state_low);
    const auto state_high_64 = _mm512_cvtepi32_epi64(state_high);
    const auto a64 = _mm512_set1_epi64(kVecMultiplier);
    auto product_low = _mm512_mul_epu32(state_low_64, a64);
    auto product_high = _mm512_mul_epu32(state_high_64, a64);

    // optimized modulus computation for 2^31-1
    const auto m64 = _mm512_set1_epi64(kModulus);
    auto compute_mod = [m64](__m512i product) {
      const auto hi = _mm512_srli_epi64(product, 31);
      const auto lo = _mm512_and_epi64(product, m64);
      auto sum = _mm512_add_epi64(hi, lo);

      const auto ge_mask = _mm512_cmp_epu64_mask(sum, m64, _MM_CMPINT_GE);
      return _mm512_mask_sub_epi64(sum, ge_mask, sum, m64);
    };

    // apply modulus to both halves
    const auto res_low = compute_mod(product_low);
    const auto res_high = compute_mod(product_high);

    // convert back to 32-bit and combine results
    const auto new_low = _mm512_cvtepi64_epi32(res_low);
    const auto new_high = _mm512_cvtepi64_epi32(res_high);

    state_ = _mm512_inserti32x8(_mm512_castsi256_si512(new_low), new_high, 1);
  }

  static __m512i initState(std::uint32_t seed) noexcept {
    std::minstd_rand rng{seed};
    auto s0 = rng();
    auto s1 = rng();
    auto s2 = rng();
    auto s3 = rng();
    auto s4 = rng();
    auto s5 = rng();
    auto s6 = rng();
    auto s7 = rng();
    auto s8 = rng();
    auto s9 = rng();
    auto s10 = rng();
    auto s11 = rng();
    auto s12 = rng();
    auto s13 = rng();
    auto s14 = rng();
    auto s15 = rng();
    return _mm512_setr_epi32(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11,
                             s12, s13, s14, s15);
  }

 private:
  __m512i state_;
};

}  // namespace simd_random
