
// #include "algorithms/dbops/sort/sort_direct.hpp"
#include <algorithm>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/interfaces/catch_interfaces_capture.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <chrono>
#include <cstddef>
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
  uniform_distribution<T> dist(1, 2);
  return dist(mt);
}

template <typename T>
bool approximate_equality(T a, T b) {
  if constexpr (std::is_floating_point_v<T>) {
    const T max_val = std::max(std::fabs(a), std::fabs(b));
    const T max_mult = std::max(static_cast<T>(1.0), max_val);
    const T ulps = std::numeric_limits<T>::epsilon() * max_mult;
    return std::fabs(a - b) <= ulps;
  } else {
    return a == b;
  }
}

template <class SimdStyle, bool positive_value, bool ptr_end, typename T = SimdStyle::base_type>
bool calc(const size_t elements, const size_t seed = 0) {
  using cpu_executor = tsl::executor<tsl::runtime::cpu>;
  cpu_executor exec;
  auto data = exec.allocate<T>(elements, 64);
  T result_sum;

  // To prevent strange behavior when calculating an average for float but store it in a double, we default to double as
  // result wrapper. This allows for easier approximate equality checking.
  using avg_t = std::conditional_t<std::is_floating_point_v<T>, T, double>;
  avg_t result_avg;

  T val = get_random_val<T>(seed);
  if constexpr (!positive_value) {
    val *= -1;
  }
  for (size_t i = 0; i < elements; ++i) {
    data[i] = val;
  }

  tuddbs::col_sum_t<SimdStyle> summation;
  if constexpr (ptr_end) {
    summation(&result_sum, data, data + elements);
  } else {
    summation(&result_sum, data, elements);
  }

  tuddbs::col_average_t<SimdStyle> averager;
  if constexpr (ptr_end) {
    averager(&result_avg, data, data + elements);
  } else {
    averager(&result_avg, data, elements);
  }

  const T expected_sum = val * elements;
  const avg_t expected_avg = val;
  bool success = true;

  // For float and double, we allow for approximate equality. For integer types this default to exact equality.
  if (!approximate_equality<T>(result_sum, expected_sum)) {
    std::cout << "Wrong sum. Is: " << +result_sum << " but should be: " << +expected_sum << std::endl;
    success = false;
  }
  if (!approximate_equality(result_avg, expected_avg)) {
    std::cout << (positive_value ? "[positive]" : "[nevative]") << " Wrong avg. Is: " << +result_avg
              << " but should be: " << +expected_avg << " my sum was " << +result_sum << "(expected " << +expected_sum
              << ") and elements: " << elements << std::endl;
    success = false;
  }

  exec.deallocate(data);

  return success;
}

template <class SimdStyle, bool positive_value, bool ptr_end, typename T = SimdStyle::base_type>
bool calc_with_valid_masks(const size_t elements, const size_t seed = 0) {
  using cpu_executor = tsl::executor<tsl::runtime::cpu>;
  cpu_executor exec;
  auto data = exec.allocate<T>(elements, 64);
  auto valid_masks = exec.allocate<typename SimdStyle::imask_type>(elements, 64);
  T result_sum;

  // To prevent strange behavior when calculating an average for float but store it in a double, we default to double as
  // result wrapper. This allows for easier approximate equality checking.
  using avg_t = std::conditional_t<std::is_floating_point_v<T>, T, double>;
  avg_t result_avg;

  T val = get_random_val<T>(seed);
  typename SimdStyle::imask_type mask = get_random_val<typename SimdStyle::imask_type>(seed);

  if constexpr (!positive_value) {
    val *= -1;
  }
  for (size_t i = 0; i < elements; ++i) {
    data[i] = val;
    valid_masks[i] = 0b10;  // mask;
  }

  size_t valid_count = 0;
  {
    auto mask_data = reinterpret_cast<typename SimdStyle::imask_type *>(valid_masks);
    size_t remainder = elements % SimdStyle::vector_element_count();
    size_t remainder_start = elements - remainder;
    size_t idx = 0;
    for (size_t i = 0; i < remainder_start; i += SimdStyle::vector_element_count(), ++idx) {
      // std::cout << "valid_count (old) = " << valid_count << std::endl;
      // std::cout << "mask_data[" << i << ":" << i + 8 << "] = " << std::bitset<8>(mask_data[idx]) << std::endl;
      valid_count += std::bitset<SimdStyle::vector_element_count()>(mask_data[idx]).count();
      // std::cout << "valid_count (new) = " << valid_count << std::endl;
    }
    if (remainder != 0) {
      // std::cout << "Remainder: " << std::endl;
      // std::cout << "mask_data[" << idx * 8 << ":" << (idx * 8) + 8 << "] = " << std::bitset<8>(mask_data[idx])
      // << std::endl;
      auto last_mask = std::bitset<SimdStyle::vector_element_count()>(mask_data[idx]);
      for (size_t i = 0; i < remainder; ++i) {
        // std::cout << "valid_count (old) = " << valid_count << std::endl;
        valid_count += last_mask[i];
        // std::cout << "valid_count (new) = " << valid_count << std::endl;
      }
    }
  }

  tuddbs::col_bm_sum_t<SimdStyle> summation;
  if constexpr (ptr_end) {
    summation(&result_sum, data, data + elements, valid_masks);
  } else {
    summation(&result_sum, data, elements, valid_masks);
  }

  // tuddbs::col_bm_average_t<SimdStyle> averager;
  // if constexpr (ptr_end) {
  //   averager(&result_avg, data, data + elements, valid_masks);
  // } else {
  //   averager(&result_avg, data, elements, valid_masks);
  // }

  const T expected_sum = val * valid_count;
  const avg_t expected_avg = val;
  bool success = true;

  // For float and double, we allow for approximate equality. For integer types this default to exact equality.
  if (!approximate_equality<T>(result_sum, expected_sum)) {
    std::cout << "Wrong sum. Is: " << +result_sum << " but should be: " << +expected_sum << std::endl;
    std::cout << "\t\tValue: " << +val << ". Elements: " << elements << ". Valid count: " << valid_count << std::endl;
    std::cout << "\t\tMASK value: " << std::bitset<sizeof(typename SimdStyle::imask_type) * CHAR_BIT>{mask}
              << std::endl;
    success = false;
  }
  // if (!approximate_equality(result_avg, expected_avg)) {
  //   std::cout << (positive_value ? "[positive]" : "[nevative]") << " Wrong avg. Is: " << +result_avg
  //             << " but should be: " << +expected_avg << " my sum was " << +result_sum << "(expected " <<
  //             +expected_sum
  //             << ") and elements: " << elements << std::endl;
  //   success = false;
  // }

  exec.deallocate(valid_masks);
  exec.deallocate(data);
  if (success) {
    std::cout << "Success" << std::endl;
  }
  return success;
}

template <class SimdStyle>
void test(const size_t elements, const size_t seed = 0) {
  constexpr bool use_positive_value = true;
  constexpr bool use_pointer_as_end = true;

  // REQUIRE(calc<SimdStyle, use_positive_value, use_pointer_as_end>(elements, seed));
  // REQUIRE(calc<SimdStyle, use_positive_value, !use_pointer_as_end>(elements, seed));

  REQUIRE(calc_with_valid_masks<SimdStyle, use_positive_value, use_pointer_as_end>(elements, seed));
  REQUIRE(calc_with_valid_masks<SimdStyle, use_positive_value, !use_pointer_as_end>(elements, seed));

  // Negative numbers are only tested for signed integral or floating point types
  if constexpr (std::is_signed_v<typename SimdStyle::base_type>) {
    // REQUIRE(calc<SimdStyle, !use_positive_value, use_pointer_as_end>(elements, seed));
    // REQUIRE(calc<SimdStyle, !use_positive_value, !use_pointer_as_end>(elements, seed));

    // REQUIRE(calc_with_valid_masks<SimdStyle, !use_positive_value, use_pointer_as_end>(elements, seed));
    // REQUIRE(calc_with_valid_masks<SimdStyle, !use_positive_value, !use_pointer_as_end>(elements, seed));
  }
}

const static size_t element_base_count = 1024 * 1024;

template <class SimdStyle>
bool dispatch_type() {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  constexpr bool is_i8 = std::is_same_v<typename SimdStyle::base_type, int8_t>;

  const size_t elements = 130;
  // std::min(element_base_count, static_cast<size_t>(std::numeric_limits<typename SimdStyle::base_type>::max() / 4));
  std::cout << "  " << tsl::type_name<typename SimdStyle::base_type>() << "..." << std::endl;
  bool all_good = true;

  // This is a general issue: We should always sum in [u]int64_t integral types and double for floating point values.
  // This requires an implementation of convert_up for all valid combinations and re-implementing the sum/average
  // accordingly.
  if constexpr (is_i8) {
    std::cout << "[WARNING] int8 is not necessarily processed with avx2 or avx512. We need to add a convert_up "
                 "implementation for all data types, such that we can perform the sum operation on the most largest "
                 "data type, i.e. 64bit int or double."
              << std::endl;
    const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    test<SimdStyle>(elements, seed);
  } else {
    for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
      const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
      test<SimdStyle>(elements + i, seed);
    }
  }
  return all_good;
}

TEMPLATE_TEST_CASE("Sum and Average, scalar", "[scalar]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::scalar>>(); }
}

#ifdef TSL_CONTAINS_SSE
TEMPLATE_TEST_CASE("Sum and Average, sse", "[sse]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t,
                   int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::sse>>(); }
}
#endif

#ifdef TSL_CONTAINS_AVX2
TEMPLATE_TEST_CASE("Sum and Average, avx2", "[avx2]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t,
                   int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::avx2>>(); }
}
#endif

#ifdef TSL_CONTAINS_AVX512
TEMPLATE_TEST_CASE("Sum and Average, avx512", "[avx512]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::avx512>>(); }
}
#endif

#ifdef TSL_CONTAINS_NEON
TEMPLATE_TEST_CASE("Sum and Average, neon", "[neon]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t,
                   int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::neon>>(); }
}
#endif
