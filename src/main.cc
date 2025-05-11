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

template <typename Func>
void benchmark(Func f, std::size_t npoints, const std::string& label) {
  constexpr int kWarmups = 10;
  constexpr int kIters = 10;

  volatile float result_sink;

  for (int i = 0; i < kWarmups; ++i) result_sink = f(npoints);

  std::uint64_t total_cycles = 0;
  for (int i = 0; i < kIters; ++i) {
    auto start = __rdtsc();
    result_sink = f(npoints);
    auto end = __rdtsc();
    total_cycles += (end - start);
  }

  double avg_cycles = static_cast<double>(total_cycles) / kIters;
  double cpe = avg_cycles / npoints;
  std::cout << label << ": avg cycles = " << avg_cycles << ", CPE = " << cpe
            << '\n';
}

float monteCarloPiNaive(std::size_t npoints) {
  std::size_t count_inside = 0;

#pragma omp parallel reduction(+ : count_inside)
  {
    int tid = omp_get_thread_num();
    std::uint32_t seed = 12345 + tid * 7919;
    std::minstd_rand rng{seed};
    std::uniform_real_distribution<float> dist{-1.f, 1.f};

#pragma omp for
    for (std::size_t i = 0; i < npoints; ++i) {
      auto x = dist(rng);
      auto y = dist(rng);
      count_inside += (x * x + y * y <= 1.f);
    }
  }

  return static_cast<float>(count_inside) * 4 / npoints;
}

float monteCarloPiSimd(std::size_t npoints) {
  assert((npoints & 0xf) == 0);
  std::size_t count_inside = 0;

#pragma omp parallel reduction(+ : count_inside)
  {
    int tid = omp_get_thread_num();
    std::uint32_t seed = 12345 + tid * 7919;
    simd_random::minstd_rand rng{seed};
    simd_random::uniform_distribution dist{-1.f, 1.f};

#pragma omp for
    for (std::size_t i = 0; i < npoints; i += 16) {
      auto vec_x = dist(rng);
      auto vec_y = dist(rng);
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
  auto res_vec = monteCarloPiSimd(kNumPoints);
  std::cout << "std::minstd_rand: " << res_naive << std::endl;
  std::cout << "simd_random::minstd_rand: " << res_vec << std::endl;
  std::cout << "M_PI: " << M_PI << std::endl;

  benchmark(monteCarloPiNaive, kNumPoints, "std::minstd_rand");
  benchmark(monteCarloPiSimd, kNumPoints, "simd_random::minstd_rand");

  return 0;
}
