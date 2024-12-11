
// #include "algorithms/dbops/sort/sort_direct.hpp"
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

// #include "algorithms/dbops/arithmetic/divide.hpp"
#include "algorithms/dbops/arithmetic/arithmetic.hpp"

template <class D>
using uniform_distribution = typename std::conditional_t<
  std::is_floating_point_v<D>, std::uniform_real_distribution<D>,
  typename std::conditional_t<std::is_integral_v<D>, std::uniform_int_distribution<D>, void>>;

template <typename T>
T get_random_root(size_t seed = 0) {
  if (seed == 0) {
    seed = 13371337;
  }
  std::mt19937_64 mt(seed);
  const size_t numeric_max = static_cast<size_t>(std::numeric_limits<T>::max());
  uniform_distribution<T> dist(1, static_cast<size_t>(std::sqrt(numeric_max)));

  const T val = dist(mt);
  if constexpr (std::is_signed_v<T>) {
    std::uniform_int_distribution<uint64_t> flip(0, 1);
    if (flip(mt)) {
      return val * -1;
    }
    return val;
  } else {
    return val;
  }
  return dist(mt);
}

template <class SimdStyle, typename T = SimdStyle::base_type>
bool test(const size_t elements, const size_t seed = 0) {
  // std::cout << "Running on " << tsl::type_name<T>() << " with "
  //           << tsl::type_name<typename SimdStyle::target_extension>()
  //           << ", Scalar Remainder: " << elements % SimdStyle::vector_element_count() << std::endl;

  using cpu_executor = tsl::executor<tsl::runtime::cpu>;
  cpu_executor exec;
  auto a = exec.allocate<T>(elements, 64);
  auto b = exec.allocate<T>(elements, 64);
  auto c = exec.allocate<T>(elements, 64);

  const T root = get_random_root<T>(seed);
  const T exp = root * root;
  for (size_t i = 0; i < elements; ++i) {
    a[i] = exp;
    b[i] = root;
  }

  using HS = tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::div>;
  using div_t = tuddbs::Arithmetic<SimdStyle, HS>;
  div_t divider;
  divider(c, a, a + elements, b);
  bool success = true;
  for (size_t i = 0; i < elements; ++i) {
    bool correct;
    if constexpr (std::is_floating_point_v<T>) {
      correct = (std::fabs(c[i] - root) <= std::numeric_limits<T>::epsilon());
    } else {
      correct = (c[i] == root);
    }

    if (!correct) {
      std::cout << "Wrong value at index " << i << ". Is: " << +c[i] << " but should be: " << +root << std::endl;
      success = false;
      break;
    }
  }

  exec.deallocate(c);
  exec.deallocate(b);
  exec.deallocate(a);

  return success;
}

const static size_t element_base_count = 1024 * 1024;

#ifdef TSL_CONTAINS_SSE
TEST_CASE("Divide 2 columns, ui8, sse", "[sse][ui8]") {
  using SimdStyle = tsl::simd<uint8_t, tsl::sse>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, ui16, sse", "[sse][ui16]") {
  using SimdStyle = tsl::simd<uint16_t, tsl::sse>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, ui32, sse", "[sse][ui32]") {
  using SimdStyle = tsl::simd<uint32_t, tsl::sse>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, ui64, sse", "[sse][ui64]") {
  using SimdStyle = tsl::simd<uint64_t, tsl::sse>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, i8, sse", "[sse][i8]") {
  using SimdStyle = tsl::simd<int8_t, tsl::sse>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, i16, sse", "[sse][i16]") {
  using SimdStyle = tsl::simd<int16_t, tsl::sse>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, i32, sse", "[sse][i32]") {
  using SimdStyle = tsl::simd<int32_t, tsl::sse>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, i64, sse", "[sse][i64]") {
  using SimdStyle = tsl::simd<int64_t, tsl::sse>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, f32, sse", "[sse][f32]") {
  using SimdStyle = tsl::simd<float, tsl::sse>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, f64, sse", "[sse][f64]") {
  using SimdStyle = tsl::simd<double, tsl::sse>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
#endif

#ifdef TSL_CONTAINS_AVX2
TEST_CASE("Divide 2 columns, ui8, avx2", "[avx2][ui8]") {
  using SimdStyle = tsl::simd<uint8_t, tsl::avx2>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, ui16, avx2", "[avx2][ui16]") {
  using SimdStyle = tsl::simd<uint16_t, tsl::avx2>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, ui32, avx2", "[avx2][ui32]") {
  using SimdStyle = tsl::simd<uint32_t, tsl::avx2>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, ui64, avx2", "[avx2][ui64]") {
  using SimdStyle = tsl::simd<uint64_t, tsl::avx2>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, i8, avx2", "[avx2][i8]") {
  using SimdStyle = tsl::simd<int8_t, tsl::avx2>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, i16, avx2", "[avx2][i16]") {
  using SimdStyle = tsl::simd<int16_t, tsl::avx2>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, i32, avx2", "[avx2][i32]") {
  using SimdStyle = tsl::simd<int32_t, tsl::avx2>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, i64, avx2", "[avx2][i64]") {
  using SimdStyle = tsl::simd<int64_t, tsl::avx2>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, f32, avx2", "[avx2][f32]") {
  using SimdStyle = tsl::simd<float, tsl::avx2>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, f64, avx2", "[avx2][f64]") {
  using SimdStyle = tsl::simd<double, tsl::avx2>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
#endif

#ifdef TSL_CONTAINS_AVX512
TEST_CASE("Divide 2 columns, ui8, avx512", "[avx512][ui8]") {
  using SimdStyle = tsl::simd<uint8_t, tsl::avx512>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, ui16, avx512", "[avx512][ui16]") {
  using SimdStyle = tsl::simd<uint16_t, tsl::avx512>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, ui32, avx512", "[avx512][ui32]") {
  using SimdStyle = tsl::simd<uint32_t, tsl::avx512>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, ui64, avx512", "[avx512][ui64]") {
  using SimdStyle = tsl::simd<uint64_t, tsl::avx512>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, i8, avx512", "[avx512][i8]") {
  using SimdStyle = tsl::simd<int8_t, tsl::avx512>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, i16, avx512", "[avx512][i16]") {
  using SimdStyle = tsl::simd<int16_t, tsl::avx512>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, i32, avx512", "[avx512][i32]") {
  using SimdStyle = tsl::simd<int32_t, tsl::avx512>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, i64, avx512", "[avx512][i64]") {
  using SimdStyle = tsl::simd<int64_t, tsl::avx512>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, f32, avx512", "[avx512][f32]") {
  using SimdStyle = tsl::simd<float, tsl::avx512>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, f64, avx512", "[avx512][f64]") {
  using SimdStyle = tsl::simd<double, tsl::avx512>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
#endif

#ifdef TSL_CONTAINS_NEON
TEST_CASE("Divide 2 columns, ui8, neon", "[neon][ui8]") {
  using SimdStyle = tsl::simd<uint8_t, tsl::neon>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, ui16, neon", "[neon][ui16]") {
  using SimdStyle = tsl::simd<uint16_t, tsl::neon>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, ui32, neon", "[neon][ui32]") {
  using SimdStyle = tsl::simd<uint32_t, tsl::neon>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, ui64, neon", "[neon][ui64]") {
  using SimdStyle = tsl::simd<uint64_t, tsl::neon>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, i8, neon", "[neon][i8]") {
  using SimdStyle = tsl::simd<int8_t, tsl::neon>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, i16, neon", "[neon][i16]") {
  using SimdStyle = tsl::simd<int16_t, tsl::neon>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, i32, neon", "[neon][i32]") {
  using SimdStyle = tsl::simd<int32_t, tsl::neon>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, i64, neon", "[neon][i64]") {
  using SimdStyle = tsl::simd<int64_t, tsl::neon>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, f32, neon", "[neon][f32]") {
  using SimdStyle = tsl::simd<float, tsl::neon>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
TEST_CASE("Divide 2 columns, f64, neon", "[neon][f64]") {
  using SimdStyle = tsl::simd<double, tsl::neon>;
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    REQUIRE(test<SimdStyle>(element_base_count + i, seed));
  }
}
#endif