
// #include "algorithms/dbops/sort/sort_direct.hpp"
#include "algorithms/dbops/arithmetic/arithmetic.hpp"

#include <algorithm>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

template <typename T>
struct TestData {
  T col1_value;
  T col2_value;
  T expected_result;

  TestData() = default;
  TestData(const TestData& other) = default;
  TestData& operator=(const TestData& other) = default;

  void print() const {
    std::cout << "C1: " << +col1_value << " C2: " << +col2_value << " Exp: " << +expected_result << std::endl;
  }
};

template <class D>
using uniform_distribution = typename std::conditional_t<
  std::is_floating_point_v<D>, std::uniform_real_distribution<D>,
  typename std::conditional_t<std::is_integral_v<D>, std::uniform_int_distribution<D>, void>>;

template <typename T>
T get_random_val(const T max_val, const size_t seed = 13371337) {
  std::mt19937_64 mt(seed);
  const size_t min_bound = 1;
  uniform_distribution<T> dist(min_bound, max_val);
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

template <class HintSet, class SimdStyle, bool make_negative, typename ret_t = typename SimdStyle::base_type>
ret_t getTestVal(const size_t seed = 13371337) {
  ret_t max_val;
  if constexpr (tuddbs::has_hint<HintSet, tuddbs::hints::arithmetic::add>) {
    max_val = std::numeric_limits<ret_t>::max() / 2;
  } else if constexpr (tuddbs::has_hint<HintSet, tuddbs::hints::arithmetic::sub>) {
    max_val = std::numeric_limits<ret_t>::max();
  } else if constexpr (tuddbs::has_hint<HintSet, tuddbs::hints::arithmetic::div>) {
    max_val = std::sqrt(std::numeric_limits<ret_t>::max());
  } else if constexpr (tuddbs::has_hint<HintSet, tuddbs::hints::arithmetic::mul>) {
    max_val = std::sqrt(std::numeric_limits<ret_t>::max());
  } else {
    throw std::runtime_error("getTestVal: No known arithmetic given. Implement me for: " + tsl::type_name<HintSet>());
  }
  const ret_t testval = get_random_val(max_val, seed);
  return make_negative ? (testval * -1) : testval;
}

template <class HintSet, typename T>
T getExpectedVal(const T testVal) {
  if (tuddbs::has_hint<HintSet, tuddbs::hints::arithmetic::add>) {
    return 2 * testVal;
  } else if constexpr (tuddbs::has_hint<HintSet, tuddbs::hints::arithmetic::sub>) {
    return 0;
  } else if constexpr (tuddbs::has_hint<HintSet, tuddbs::hints::arithmetic::div>) {
    return testVal * testVal;
  } else if constexpr (tuddbs::has_hint<HintSet, tuddbs::hints::arithmetic::mul>) {
    return testVal * testVal;
  } else {
    throw std::runtime_error("getExpectedVal: No known arithmetic given. Implement me for: " +
                             tsl::type_name<HintSet>());
  }
  return 0;
}

template <class HintSet, class SimdStyle, bool make_negative, typename T = typename SimdStyle::base_type>
TestData<T> getTestAndResultValue(const size_t seed) {
  const T testVal = getTestVal<HintSet, SimdStyle, make_negative>(seed);
  const T expected = getExpectedVal<HintSet>(testVal);
  TestData<T> values;
  if constexpr (tuddbs::has_hint<HintSet, tuddbs::hints::arithmetic::div>) {
    values.col1_value = expected;
    values.col2_value = testVal;
    values.expected_result = testVal;
  } else {
    values.col1_value = testVal;
    values.col2_value = testVal;
    values.expected_result = expected;
  }
  return values;
}

template <class arithmetic_t, class SimdStyle, typename T = SimdStyle::base_type>
bool calc(const T testval1, const T testval2, const T expected_result, const size_t elements) {
  using cpu_executor = tsl::executor<tsl::runtime::cpu>;
  cpu_executor exec;
  auto data1 = exec.allocate<T>(elements, 64);
  auto data2 = exec.allocate<T>(elements, 64);
  auto result = exec.allocate<T>(elements, 64);

  for (size_t i = 0; i < elements; ++i) {
    data1[i] = testval1;
    data2[i] = testval2;
  }

  arithmetic_t perform;
  perform(result, data1, data1 + elements, data2);

  bool success = true;
  for (size_t i = 0; i < elements; ++i) {
    if (!approximate_equality(result[i], expected_result)) {
      std::cout << "Wrong value at index " << i << ". Is: " << +result[i] << " but should be: " << +expected_result
                << std::endl;
      success = false;
      break;
    }
  }

  exec.deallocate(result);
  exec.deallocate(data2);
  exec.deallocate(data1);

  return success;
}

template <class HintSet, class arithmetic_t, class SimdStyle>
void test(const size_t elements, const size_t seed = 0) {
  using base_t = typename SimdStyle::base_type;
  using test_tuple_t = std::tuple<base_t, base_t, base_t>;
  constexpr bool make_negative = true;

  for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
    const TestData<base_t> testData_pos = getTestAndResultValue<HintSet, SimdStyle, !make_negative>(seed);
    REQUIRE(calc<arithmetic_t, SimdStyle>(testData_pos.col1_value, testData_pos.col2_value,
                                          testData_pos.expected_result, elements));

    if constexpr (std::is_signed_v<base_t>) {
      const TestData<base_t> testData_neg = getTestAndResultValue<HintSet, SimdStyle, make_negative>(seed);
      REQUIRE(calc<arithmetic_t, SimdStyle>(testData_neg.col1_value, testData_neg.col2_value,
                                            testData_neg.expected_result, elements));
    }
  }
}

const static size_t element_base_count = 1024 * 1024;

template <class SimdStyle, class HintSet>
void dispatch_type(const size_t elements) {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  const size_t seed = 12;
  using arithmetic_t = tuddbs::Arithmetic<SimdStyle, HintSet>;
  test<HintSet, arithmetic_t, SimdStyle>(elements, seed);
}

TEMPLATE_TEST_CASE("Add 2 columns, scalar", "[add][scalar]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  using HintSet = tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::add>;
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::scalar>, HintSet>(element_base_count); }
}
TEMPLATE_TEST_CASE("Subtract 2 columns, scalar", "[sub][scalar]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t,
                   int16_t, int32_t, int64_t, float, double) {
  using HintSet = tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::sub>;
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::scalar>, HintSet>(element_base_count); }
}
TEMPLATE_TEST_CASE("Divide 2 columns, scalar", "[div][scalar]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  using HintSet = tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::div>;
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::scalar>, HintSet>(element_base_count); }
}
TEMPLATE_TEST_CASE("Multiply 2 columns, scalar", "[mul][scalar]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t,
                   int16_t, int32_t, int64_t, float, double) {
  using HintSet = tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::mul>;
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::scalar>, HintSet>(element_base_count); }
}

#ifdef TSL_CONTAINS_SSE
TEMPLATE_TEST_CASE("Add 2 columns, sse", "[add][sse]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t,
                   int64_t, float, double) {
  using HintSet = tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::add>;
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::sse>, HintSet>(element_base_count); }
}
TEMPLATE_TEST_CASE("Subtract 2 columns, sse", "[sub][sse]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  using HintSet = tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::sub>;
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::sse>, HintSet>(element_base_count); }
}
TEMPLATE_TEST_CASE("Divide 2 columns, sse", "[div][sse]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  using HintSet = tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::div>;
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::sse>, HintSet>(element_base_count); }
}
TEMPLATE_TEST_CASE("Multiply 2 columns, sse", "[mul][sse]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  using HintSet = tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::mul>;
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::sse>, HintSet>(element_base_count); }
}
#endif

#ifdef TSL_CONTAINS_AVX2
TEMPLATE_TEST_CASE("Add 2 columns, avx2", "[add][avx2]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  using HintSet = tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::add>;
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::avx2>, HintSet>(element_base_count); }
}
TEMPLATE_TEST_CASE("Subtract 2 columns, avx2", "[sub][avx2]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  using HintSet = tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::sub>;
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::avx2>, HintSet>(element_base_count); }
}
TEMPLATE_TEST_CASE("Divide 2 columns, avx2", "[div][avx2]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  using HintSet = tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::div>;
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::avx2>, HintSet>(element_base_count); }
}
TEMPLATE_TEST_CASE("Multiply 2 columns, avx2", "[mul][avx2]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  using HintSet = tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::mul>;
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::avx2>, HintSet>(element_base_count); }
}
#endif

#ifdef TSL_CONTAINS_AVX512
TEMPLATE_TEST_CASE("Add 2 columns, avx512", "[add][avx512]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  using HintSet = tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::add>;
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::avx512>, HintSet>(element_base_count); }
}
TEMPLATE_TEST_CASE("Subtract 2 columns, avx512", "[sub][avx512]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t,
                   int16_t, int32_t, int64_t, float, double) {
  using HintSet = tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::sub>;
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::avx512>, HintSet>(element_base_count); }
}
TEMPLATE_TEST_CASE("Divide 2 columns, avx512", "[div][avx512]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  using HintSet = tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::div>;
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::avx512>, HintSet>(element_base_count); }
}
TEMPLATE_TEST_CASE("Multiply 2 columns, avx512", "[mul][avx512]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t,
                   int16_t, int32_t, int64_t, float, double) {
  using HintSet = tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::mul>;
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::avx512>, HintSet>(element_base_count); }
}
#endif

#ifdef TSL_CONTAINS_NEON
TEMPLATE_TEST_CASE("Add 2 columns, neon", "[add][neon]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  using HintSet = tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::add>;
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::neon>, HintSet>(element_base_count); }
}
TEMPLATE_TEST_CASE("Subtract 2 columns, neon", "[sub][neon]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  using HintSet = tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::sub>;
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::neon>, HintSet>(element_base_count); }
}
TEMPLATE_TEST_CASE("Divide 2 columns, neon", "[div][neon]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  using HintSet = tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::div>;
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::neon>, HintSet>(element_base_count); }
}
TEMPLATE_TEST_CASE("Multiply 2 columns, neon", "[mul][neon]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  using HintSet = tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::mul>;
  SECTION(tsl::type_name<TestType>()) { dispatch_type<tsl::simd<TestType, tsl::neon>, HintSet>(element_base_count); }
}
#endif
