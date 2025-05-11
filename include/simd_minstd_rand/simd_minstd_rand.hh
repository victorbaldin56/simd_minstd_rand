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
  minstd_rand(std::uint32_t seed) noexcept : state_(_mm512_set1_epi32(seed)) {}

  auto operator()() noexcept {
    auto vec_mask = _mm512_set1_epi32(0x7fffffff);
    auto vec_multiplier = _mm512_set1_epi32(1098894339);
    auto product_lo = _mm512_mullo_epi32(state_, vec_multiplier);
    auto product_hi = _mm512_mul_epi32(state_, vec_multiplier);

    auto x0 = _mm512_and_epi32(product_lo, vec_mask);
    auto temp = _mm512_or_epi32(_mm512_srli_epi32(product_lo, 31),
                                _mm512_slli_epi32(product_hi, 1));
    auto x1 = _mm512_and_epi32(temp, vec_mask);
    auto x2 = _mm512_srli_epi32(product_hi, 30);
    auto sum = _mm512_add_epi32(x0, x1);
    sum = _mm512_add_epi32(sum, x2);

    auto overflow = _mm512_cmpge_epu32_mask(sum, vec_mask);
    state_ = _mm512_mask_sub_epi32(sum, overflow, sum, vec_mask);

    return state_;
  }

 private:
  __m512i state_;
};

class uniform_distribution {
 public:
  constexpr uniform_distribution(float min, float max) noexcept
      : min_{min}, max_{max} {}

  template <typename Rand>
  auto operator()(Rand&& rng) const noexcept {
    __m512 rand_floats = _mm512_cvtepi32_ps(rng());
    rand_floats /= _mm512_set1_ps(
        static_cast<float>(std::numeric_limits<std::int32_t>::max()));

    auto range = _mm512_set1_ps(max_ - min_);
    auto offset = _mm512_set1_ps(min_);
    rand_floats = _mm512_fmadd_ps(rand_floats, range, offset);
    return rand_floats;
  }

 private:
  float min_, max_;
};
}  // namespace simd_random
