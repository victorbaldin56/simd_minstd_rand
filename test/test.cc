#include <algorithm>
#include <array>
#include <cstring>
#include <random>

#include "gtest/gtest.h"
#include "simd_minstd_rand/simd_minstd_rand.hh"

TEST(simd_minstd_rand, functional) {
  constexpr std::uint32_t kSeed = 1;
  simd_random::minstd_rand vec_rng{kSeed};
  std::minstd_rand rng{kSeed};

  constexpr std::size_t kIters = 100000000;
  for (std::size_t i = 0; i < kIters; i += 16) {
    // simulate vector rng with scalar reference
    std::array<std::uint32_t, 16> ss;
    std::generate(ss.begin(), ss.end(), [&] { return rng(); });
    auto v = vec_rng();
    ASSERT_EQ(std::memcmp(ss.data(), &v, 64), 0);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
