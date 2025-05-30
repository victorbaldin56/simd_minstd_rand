# Векторизация алгоритма LCG генерации случайных чисел

Данная работа была посвящена написанию векторного (SIMD) генератора псевдослучайных чисел с использованием
алгоритма LCG (Linear Congruent Engine) и измерению его производительности.

## Окружение

Использовалась машина с Intel(R) Core(TM) i5-11400F @ 2.60GHz, поддерживающим в том числе расширения AVX2 и AVX512.
Операционная система: Arch Linux kernel 6.13.8. Компилятор: Clang 19.

## Сборка проекта

Гарантируется корректная сборка проекта на x64-машинах с расширениями AVX2 и AVX512 с Linux с компилятором, указанным в окружении выше.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install conan
conan profile detect --force
conan install . --build=missing --output-folder=build -pr:a=linux_release.profile
cd build
cmake .. --preset conan-release
cmake --build . -j
```

## Реализация генератора

Версия, написанная для AVX512 на C++ с использованием векторных интринсиков:

```cpp
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
```

Параметры были взяты такие же как у [std::minstd_rand](https://en.cppreference.com/w/cpp/numeric/random/linear_congruential_engine).
В дальнейшим мы будем сравнивать именно с ним.

## Методика верификации и бенчаркинга

Для замеров производительности и подтверждения функциональной корректности используем бенчмарк,
вычисляющий число $\pi$ методом Монте Карло на количестве в $10^9$ случайных точек в квадрате
$[-1, 1] \times [-1, 1]$.

Сравнивать будем со скалярным генератором `std::minstd_rand`, как уже оговаривалось ранее.
Вычисления распараллелим при помощи библиотеки LLVM OpenMP в обоих случаях.

Скалярная версия:

```cpp
float monteCarloPiScalar(std::size_t npoints) {
  std::size_t count_inside = 0;

#pragma omp parallel reduction(+ : count_inside)
  {
    std::uint32_t tid = omp_get_thread_num();
    std::minstd_rand rng{tid};

#pragma omp for
    for (std::size_t i = 0; i < npoints; ++i) {
      auto x = convert(rng(), -1.f, 1.f);
      auto y = convert(rng(), -1.f, 1.f);
      count_inside += (x * x + y * y <= 1.f);
    }
  }

  return static_cast<float>(count_inside) * 4 / npoints;
}
```

Векторная:

```cpp
float monteCarloPiSimd(std::size_t npoints) {
  assert((npoints & 0xf) == 0);
  std::size_t count_inside = 0;

#pragma omp parallel reduction(+ : count_inside)
  {
    std::uint32_t tid = omp_get_thread_num();
    simd_random::minstd_rand rng{tid};

#pragma omp for
    for (std::size_t i = 0; i < npoints; i += 16) {
      auto vec_x = convert(rng(), -1.f, 1.f);
      auto vec_y = convert(rng(), -1.f, 1.f);
      auto cmp_mask = _mm512_cmp_ps_mask(vec_x * vec_x + vec_y * vec_y,
                                         _mm512_set1_ps(1.f), _CMP_LE_OS);
      count_inside += _mm_popcnt_u32(cmp_mask);
    }
  }

  return static_cast<float>(count_inside) * 4 / npoints;
}
```

Функция `convert()` имеет скалярную и веркторную реализации и
шкалирует полученный от генератора набор бит на нужный интервал.

Полный код бенчмарка можно увидеть в исходном файле. Сборка производилась
с флагами компиляции которые можно найти в [Conan профиле](linux_release.profile) Release сборки.
Полученные результаты:

```
std::minstd_rand: 3.14159
simd_random::minstd_rand: 3.14157
M_PI: 3.14159
std::minstd_rand: avg cycles = 1.43753e+10, CPE = 1.43753
simd_random::minstd_rand: avg cycles = 1.78411e+09, CPE = 0.178411
```

Превосходство векторизованной версии составило 8.1 раза.
Видно, что Clang не смог автоматически заинлайнить
и векторизовать алгоритм `std::minstd_rand`.

## Вывод

Генераторы случайных чисел являются примером алгоритма, векторизация
которого все еще зачастую требует вмешательства программиста.
