
// #include "algorithms/dbops/sort/sort_direct.hpp"
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

#include "algorithms/dbops/arithmetic/arithmetic.hpp"

template <class D>
using uniform_distribution = typename std::conditional_t<
  std::is_floating_point_v<D>, std::uniform_real_distribution<D>,
  typename std::conditional_t<std::is_integral_v<D>, std::uniform_int_distribution<D>, void>>;

template <typename T>
T get_random_val(size_t seed = 0) {
  if (seed == 0) {
    seed = 13371337;
  }
  std::mt19937_64 mt(seed);
  const size_t numeric_max_half = static_cast<size_t>(std::numeric_limits<T>::max()) / 2;
  const size_t min_bound = 0;

  uniform_distribution<T> dist(min_bound, numeric_max_half);
  return dist(mt);
}

template <class SimdStyle, bool positive_value, typename T = SimdStyle::base_type>
bool calc(const size_t elements, const size_t seed = 0) {
  using cpu_executor = tsl::executor<tsl::runtime::cpu>;
  cpu_executor exec;
  auto a = exec.allocate<T>(elements, 64);
  auto b = exec.allocate<T>(elements, 64);
  auto c = exec.allocate<T>(elements, 64);

  const T val = get_random_val<T>(seed) * (positive_value ? 1 : -1);
  for (size_t i = 0; i < elements; ++i) {
    a[i] = val;
    b[i] = val;
  }

  tuddbs::col_adder_t<SimdStyle> adder;
  adder(c, a, a + elements, b);

  const T expected_value = 2 * val;
  bool success = true;
  for (size_t i = 0; i < elements; ++i) {
    if (c[i] != expected_value) {
      std::cout << "Wrong value at index " << i << ". Is: " << +c[i] << " but should be: " << +expected_value
                << std::endl;
      success = false;
      break;
    }
  }

  exec.deallocate(c);
  exec.deallocate(b);
  exec.deallocate(a);

  return success;
}

template <class SimdStyle>
bool test(const size_t elements, const size_t seed = 0) {
  const bool pos_adder = calc<SimdStyle, true>(elements, seed);
  const bool neg_adder = calc<SimdStyle, false>(elements, seed);
  return pos_adder && neg_adder;
}

const static size_t element_base_count = 1024 * 1024;

template <class SimdStyle>
bool dispatch_type(const size_t elements) {
  std::cout << "  " << tsl::type_name<typename SimdStyle::base_type>() << "..." << std::endl;
  bool all_good = true;
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    all_good &= test<SimdStyle>(elements + i, seed);
  }
  return all_good;
}

TEST_CASE("Add 2 columns, scalar", "[scalar]") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  SECTION("ui8") { REQUIRE(dispatch_type<tsl::simd<uint8_t, tsl::scalar>>(element_base_count)); }
  SECTION("ui16") { REQUIRE(dispatch_type<tsl::simd<uint16_t, tsl::scalar>>(element_base_count)); }
  SECTION("ui32") { REQUIRE(dispatch_type<tsl::simd<uint32_t, tsl::scalar>>(element_base_count)); }
  SECTION("ui64") { REQUIRE(dispatch_type<tsl::simd<uint64_t, tsl::scalar>>(element_base_count)); }
  SECTION("i8") { REQUIRE(dispatch_type<tsl::simd<int8_t, tsl::scalar>>(element_base_count)); }
  SECTION("i16") { REQUIRE(dispatch_type<tsl::simd<int16_t, tsl::scalar>>(element_base_count)); }
  SECTION("i32") { REQUIRE(dispatch_type<tsl::simd<int32_t, tsl::scalar>>(element_base_count)); }
  SECTION("i64") { REQUIRE(dispatch_type<tsl::simd<int64_t, tsl::scalar>>(element_base_count)); }
  SECTION("f32") { REQUIRE(dispatch_type<tsl::simd<float, tsl::scalar>>(element_base_count)); }
  SECTION("f64") { REQUIRE(dispatch_type<tsl::simd<double, tsl::scalar>>(element_base_count)); }
}

#ifdef TSL_CONTAINS_SSE
TEST_CASE("Add 2 columns, sse", "[sse]") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  SECTION("ui8") { REQUIRE(dispatch_type<tsl::simd<uint8_t, tsl::sse>>(element_base_count)); }
  SECTION("ui16") { REQUIRE(dispatch_type<tsl::simd<uint16_t, tsl::sse>>(element_base_count)); }
  SECTION("ui32") { REQUIRE(dispatch_type<tsl::simd<uint32_t, tsl::sse>>(element_base_count)); }
  SECTION("ui64") { REQUIRE(dispatch_type<tsl::simd<uint64_t, tsl::sse>>(element_base_count)); }
  SECTION("i8") { REQUIRE(dispatch_type<tsl::simd<int8_t, tsl::sse>>(element_base_count)); }
  SECTION("i16") { REQUIRE(dispatch_type<tsl::simd<int16_t, tsl::sse>>(element_base_count)); }
  SECTION("i32") { REQUIRE(dispatch_type<tsl::simd<int32_t, tsl::sse>>(element_base_count)); }
  SECTION("i64") { REQUIRE(dispatch_type<tsl::simd<int64_t, tsl::sse>>(element_base_count)); }
  SECTION("f32") { REQUIRE(dispatch_type<tsl::simd<float, tsl::sse>>(element_base_count)); }
  SECTION("f64") { REQUIRE(dispatch_type<tsl::simd<double, tsl::sse>>(element_base_count)); }
}
#endif

#ifdef TSL_CONTAINS_AVX2
TEST_CASE("Add 2 columns, avx2", "[avx2]") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  SECTION("ui8") { REQUIRE(dispatch_type<tsl::simd<uint8_t, tsl::avx2>>(element_base_count)); }
  SECTION("ui16") { REQUIRE(dispatch_type<tsl::simd<uint16_t, tsl::avx2>>(element_base_count)); }
  SECTION("ui32") { REQUIRE(dispatch_type<tsl::simd<uint32_t, tsl::avx2>>(element_base_count)); }
  SECTION("ui64") { REQUIRE(dispatch_type<tsl::simd<uint64_t, tsl::avx2>>(element_base_count)); }
  SECTION("i8") { REQUIRE(dispatch_type<tsl::simd<int8_t, tsl::avx2>>(element_base_count)); }
  SECTION("i16") { REQUIRE(dispatch_type<tsl::simd<int16_t, tsl::avx2>>(element_base_count)); }
  SECTION("i32") { REQUIRE(dispatch_type<tsl::simd<int32_t, tsl::avx2>>(element_base_count)); }
  SECTION("i64") { REQUIRE(dispatch_type<tsl::simd<int64_t, tsl::avx2>>(element_base_count)); }
  SECTION("f32") { REQUIRE(dispatch_type<tsl::simd<float, tsl::avx2>>(element_base_count)); }
  SECTION("f64") { REQUIRE(dispatch_type<tsl::simd<double, tsl::avx2>>(element_base_count)); }
}
#endif

#ifdef TSL_CONTAINS_AVX512
TEST_CASE("Add 2 columns, avx512", "[avx512]") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  SECTION("ui8") { REQUIRE(dispatch_type<tsl::simd<uint8_t, tsl::avx512>>(element_base_count)); }
  SECTION("ui16") { REQUIRE(dispatch_type<tsl::simd<uint16_t, tsl::avx512>>(element_base_count)); }
  SECTION("ui32") { REQUIRE(dispatch_type<tsl::simd<uint32_t, tsl::avx512>>(element_base_count)); }
  SECTION("ui64") { REQUIRE(dispatch_type<tsl::simd<uint64_t, tsl::avx512>>(element_base_count)); }
  SECTION("i8") { REQUIRE(dispatch_type<tsl::simd<int8_t, tsl::avx512>>(element_base_count)); }
  SECTION("i16") { REQUIRE(dispatch_type<tsl::simd<int16_t, tsl::avx512>>(element_base_count)); }
  SECTION("i32") { REQUIRE(dispatch_type<tsl::simd<int32_t, tsl::avx512>>(element_base_count)); }
  SECTION("i64") { REQUIRE(dispatch_type<tsl::simd<int64_t, tsl::avx512>>(element_base_count)); }
  SECTION("f32") { REQUIRE(dispatch_type<tsl::simd<float, tsl::avx512>>(element_base_count)); }
  SECTION("f64") { REQUIRE(dispatch_type<tsl::simd<double, tsl::avx512>>(element_base_count)); }
}
#endif

#ifdef TSL_CONTAINS_NEON
TEST_CASE("Add 2 columns, neon", "[neon]") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  SECTION("ui8") { REQUIRE(dispatch_type<tsl::simd<uint8_t, tsl::neon>>(element_base_count)); }
  SECTION("ui16") { REQUIRE(dispatch_type<tsl::simd<uint16_t, tsl::neon>>(element_base_count)); }
  SECTION("ui32") { REQUIRE(dispatch_type<tsl::simd<uint32_t, tsl::neon>>(element_base_count)); }
  SECTION("ui64") { REQUIRE(dispatch_type<tsl::simd<uint64_t, tsl::neon>>(element_base_count)); }
  SECTION("i8") { REQUIRE(dispatch_type<tsl::simd<int8_t, tsl::neon>>(element_base_count)); }
  SECTION("i16") { REQUIRE(dispatch_type<tsl::simd<int16_t, tsl::neon>>(element_base_count)); }
  SECTION("i32") { REQUIRE(dispatch_type<tsl::simd<int32_t, tsl::neon>>(element_base_count)); }
  SECTION("i64") { REQUIRE(dispatch_type<tsl::simd<int64_t, tsl::neon>>(element_base_count)); }
  SECTION("f32") { REQUIRE(dispatch_type<tsl::simd<float, tsl::neon>>(element_base_count)); }
  SECTION("f64") { REQUIRE(dispatch_type<tsl::simd<double, tsl::neon>>(element_base_count)); }
}
#endif