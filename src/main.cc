#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "omp.h"
#include "ompx.h"
#include "simd_minstd_rand/simd_minstd_rand.hh"

namespace {

constexpr std::size_t kNumPoints = 100000000;

float monteCarloPiNaive(std::size_t npoints) {
  std::minstd_rand rng;
  std::uniform_real_distribution<float> dist{-1.f, 1.f};
  std::size_t count_inside = 0;

  for (std::size_t i = 0; i < npoints; ++i) {
    float x = dist(rng);
    float y = dist(rng);
    if (x * x + y * y <= 1.f) {
      ++count_inside;
    }
  }

  return static_cast<float>(count_inside) * 4 / npoints;
}

float monteCarloPiParallel(std::size_t npoints) {
  assert((npoints & 0xf) == 0);

  constexpr simd_random::uniform_distribution kDist{-1.f, 1.f};
  std::size_t count_inside = 0;

#pragma omp parallel reduction(+ : count_inside)
  {
    int tid = omp_get_thread_num();
    std::uint32_t seed = 12345 + tid * 7919;
    simd_random::minstd_rand rng{seed};

#pragma omp for
    for (std::size_t i = 0; i < npoints; i += 16) {
      auto vec_x = kDist(rng);
      auto vec_y = kDist(rng);
      auto cmp_mask = _mm512_cmp_ps_mask(vec_x * vec_x + vec_y * vec_y,
                                         _mm512_set1_ps(1.f), _CMP_LE_OS);
      count_inside += _mm_popcnt_u32(cmp_mask);
    }
  }

  return static_cast<float>(count_inside) * 4 / npoints;
}
}  // namespace

int main() {
  auto res_naive = monteCarloPiNaive(kNumPoints);
  auto res_vec = monteCarloPiParallel(kNumPoints);
  std::cout << "Naive: " << res_naive << std::endl;
  std::cout << "Parallel: " << res_vec << std::endl;
  std::cout << "Reference: " << M_PI << std::endl;

  return 0;
}
