add_executable(minstd_rand_test test.cc)
target_link_libraries(minstd_rand_test simd_rng gtest::gtest)

gtest_discover_tests(minstd_rand_test)
