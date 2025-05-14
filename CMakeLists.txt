cmake_minimum_required(VERSION 3.14)
project(simd_minstd_rand)

find_package(OpenMP REQUIRED)

add_library(simd_rng INTERFACE)
target_include_directories(simd_rng INTERFACE include)
target_compile_options(simd_rng INTERFACE "-march=native")

add_executable(pi src/main.cc)
target_link_libraries(pi simd_rng llvm-openmp::llvm-openmp)

option(ENABLE_TESTS "Enable testing" ON)
if (${ENABLE_TESTS})
  find_package(GTest REQUIRED)
  include(GoogleTest)
  add_subdirectory(test)
endif()
