#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/interfaces/catch_interfaces_capture.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>

#include "algorithms/dbops/filter/filter.hpp"
#include "simdops_testing.hpp"
#include "tsl.hpp"
#include "tslCPUrt.hpp"

using namespace tuddbs;

template <class D>
using uniform_distribution = typename std::conditional_t<
  std::is_floating_point_v<D>, std::uniform_real_distribution<D>,
  typename std::conditional_t<std::is_integral_v<D>, std::uniform_int_distribution<D>, void>>;

template <typename cmp_t>
struct cmp_eq {
  bool operator()(const cmp_t& a, const cmp_t& b) { return a == b; }
};

template <typename cmp_t>
struct cmp_neq {
  bool operator()(const cmp_t& a, const cmp_t& b) { return a != b; }
};

template <typename cmp_t>
struct cmp_lt {
  bool operator()(const cmp_t& a, const cmp_t& b) { return a < b; }
};

template <typename cmp_t>
struct cmp_gt {
  bool operator()(const cmp_t& a, const cmp_t& b) { return a > b; }
};

template <typename cmp_t>
struct cmp_le {
  bool operator()(const cmp_t& a, const cmp_t& b) { return a <= b; }
};

template <typename cmp_t>
struct cmp_ge {
  bool operator()(const cmp_t& a, const cmp_t& b) { return a >= b; }
};

template <typename cmp_t>
struct cmp_bwi {
  bool operator()(const cmp_t& v, const cmp_t& pred1, const cmp_t& pred2) { return ((v >= pred1) && (v <= pred2)); }
};

template <typename T, typename P>
void prepare_data(P* output, T* input, size_t element_count, P* positions, size_t selectivity, T predicate_true,
                  T predicate_false) {
  for (auto i = 0; i < element_count; ++i) {
    input[i] = predicate_false;
    output[i] = 0;
  }
  for (auto i = 0; i < (element_count * selectivity / 100); ++i) {
    input[positions[i]] = predicate_true;
  }
}

template <typename T, typename P>
void prepare_data_range(P* output, T* input, size_t element_count, P* positions, size_t selectivity, T predicate_true_1,
                        T predicate_true_2, T predicate_false) {
  for (auto i = 0; i < element_count; ++i) {
    input[i] = predicate_false;
    output[i] = 0;
  }

  std::mt19937 mt{std::random_device{}()};
  uniform_distribution<T> dist(predicate_true_1, predicate_true_2);
  for (auto i = 0; i < (element_count * selectivity / 100); ++i) {
    input[positions[i]] = dist(mt);
  }
}

template <typename P>
void init_positions(P* positions, size_t element_count) {
  for (auto i = 0; i < element_count; ++i) {
    positions[i] = i;
  }
  std::shuffle(positions, positions + element_count, std::mt19937{std::random_device{}()});
}

template <class Filter, class Cmp, typename DataType = typename Filter::SimdStyle::base_type>
bool do_select(Cmp comparator, const DataType predicate_true, const DataType predicate_false) {
  using cpu_executor = tsl::executor<tsl::runtime::cpu>;
  //** Bit width debug info **/
  // std::cerr << SimdStyle::vector_element_count() << std::endl;
  // std::cerr << std::bitset<64>((1 << SimdStyle::vector_element_count()) - 1) << std::endl;
  using PosType = typename Filter::ResultType;
  cpu_executor exec;
  bool all_good = true;
  size_t element_count = 256;
  auto input = exec.allocate<DataType>(element_count, 64);
  auto output = exec.allocate<PosType>(element_count, 64);
  auto output_reference = exec.allocate<PosType>(element_count, 64);
  auto positions = exec.allocate<PosType>(element_count, 64);
  init_positions(positions, element_count);

  for (size_t selectivity = 0; selectivity <= 100; ++selectivity) {
    prepare_data(output, input, element_count, positions, selectivity, predicate_true, predicate_false);
    memset(output_reference, 0, element_count * sizeof(PosType));

    auto ref_ptr = output_reference;
    for (size_t i = 0; i < element_count; ++i) {
      const auto res = comparator(input[i], predicate_true);
      if (res) {
        *ref_ptr = i;
        ref_ptr++;
      }
    }

    Filter simdops_scan(predicate_true);
    simdops_scan(output, input, input + element_count);

    REQUIRE(std::memcmp(output, output_reference, element_count * sizeof(PosType)) == 0);
  }
  return all_good;
}

template <class Filter, class Cmp, typename DataType = typename Filter::SimdStyle::base_type>
bool do_select_range(Cmp comparator, const DataType predicate_true_1, const DataType predicate_true_2,
                     const DataType predicate_false) {
  using cpu_executor = tsl::executor<tsl::runtime::cpu>;
  //** Bit width debug info **/
  // std::cerr << SimdStyle::vector_element_count() << std::endl;
  // std::cerr << std::bitset<64>((1 << SimdStyle::vector_element_count()) - 1) << std::endl;
  using PosType = typename Filter::ResultType;
  cpu_executor exec;
  bool all_good = true;
  size_t element_count = 256;
  auto input = exec.allocate<DataType>(element_count, 64);
  auto output = exec.allocate<PosType>(element_count, 64);
  auto output_reference = exec.allocate<PosType>(element_count, 64);
  auto positions = exec.allocate<PosType>(element_count, 64);
  init_positions(positions, element_count);

  for (size_t selectivity = 0; selectivity <= 100; ++selectivity) {
    prepare_data_range(output, input, element_count, positions, selectivity, predicate_true_1, predicate_true_2,
                       predicate_false);
    memset(output_reference, 0, element_count * sizeof(PosType));

    auto ref_ptr = output_reference;
    for (size_t i = 0; i < element_count; ++i) {
      const auto res = comparator(input[i], predicate_true_1, predicate_true_2);
      if (res) {
        *ref_ptr = i;
        ref_ptr++;
      }
    }

    Filter simdops_scan(predicate_true_1, predicate_true_2);
    simdops_scan(output, input, input + element_count);

    REQUIRE(std::memcmp(output, output_reference, element_count * sizeof(PosType)) == 0);
  }
  return all_good;
}

template <class SimdStyle, template <class, class> class TSLCmp, class CmpFun,
          typename cmp_t = typename SimdStyle::base_type>
bool dispatch_type(CmpFun cmp, const cmp_t pred_true = 1, const cmp_t pred_false = 0) {
  const size_t base_size = sizeof(typename SimdStyle::base_type);
  bool all_good = true;

  using HS = tuddbs::OperatorHintSet<hints::intermediate::position_list>;

  SECTION("PosType64") {
    if constexpr (base_size <= 8) {
      using Filter = tuddbs::Generic_Filter<SimdStyle, TSLCmp, HS, tsl::workaround, uint64_t>;
      all_good &= do_select<Filter>(cmp, pred_true, pred_false);
    }
  }
  SECTION("PosType32") {
    if constexpr (base_size <= 4) {
      using Filter = tuddbs::Generic_Filter<SimdStyle, TSLCmp, HS, tsl::workaround, uint32_t>;
      all_good &= do_select<Filter>(cmp, pred_true, pred_false);
    }
  }
  SECTION("PosType16") {
    if constexpr (base_size <= 2) {
      using Filter = tuddbs::Generic_Filter<SimdStyle, TSLCmp, HS, tsl::workaround, uint16_t>;
      all_good &= do_select<Filter>(cmp, pred_true, pred_false);
    }
  }
  SECTION("PosType8") {
    if constexpr (base_size == 1) {
      using Filter = tuddbs::Generic_Filter<SimdStyle, TSLCmp, HS, tsl::workaround, uint8_t>;
      all_good &= do_select<Filter>(cmp, pred_true, pred_false);
    }
  }
  return all_good;
}

template <class SimdStyle, template <class, class> class TSLCmp, class CmpFun,
          typename cmp_t = typename SimdStyle::base_type>
bool dispatch_type_range(CmpFun cmp, const cmp_t pred_true_1 = 1, const cmp_t pred_true_2 = 1,
                         const cmp_t pred_false = 0) {
  const size_t base_size = sizeof(typename SimdStyle::base_type);
  bool all_good = true;

  using HS = tuddbs::OperatorHintSet<hints::intermediate::position_list>;

  SECTION("PosType64") {
    if constexpr (base_size <= 8) {
      using Filter = tuddbs::Generic_Range_Filter<SimdStyle, TSLCmp, HS, tsl::workaround, uint64_t>;
      all_good &= do_select_range<Filter>(cmp, pred_true_1, pred_true_2, pred_false);
    }
  }
  SECTION("PosType32") {
    if constexpr (base_size <= 4) {
      using Filter = tuddbs::Generic_Range_Filter<SimdStyle, TSLCmp, HS, tsl::workaround, uint32_t>;
      all_good &= do_select_range<Filter>(cmp, pred_true_1, pred_true_2, pred_false);
    }
  }
  SECTION("PosType16") {
    if constexpr (base_size <= 2) {
      using Filter = tuddbs::Generic_Range_Filter<SimdStyle, TSLCmp, HS, tsl::workaround, uint16_t>;
      all_good &= do_select_range<Filter>(cmp, pred_true_1, pred_true_2, pred_false);
    }
  }
  SECTION("PosType8") {
    if constexpr (base_size == 1) {
      using Filter = tuddbs::Generic_Range_Filter<SimdStyle, TSLCmp, HS, tsl::workaround, uint8_t>;
      all_good &= do_select_range<Filter>(cmp, pred_true_1, pred_true_2, pred_false);
    }
  }
  return all_good;
}

TEMPLATE_TEST_CASE("Filter Equality, scalar", "[eq][scalar]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    dispatch_type<tsl::simd<TestType, tsl::scalar>, tsl::functors::equal>(cmp_eq<TestType>{});
  }
}

TEMPLATE_TEST_CASE("Filter Inequality, scalar", "[neq][scalar]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    dispatch_type<tsl::simd<TestType, tsl::scalar>, tsl::functors::nequal>(cmp_neq<TestType>{});
  }
}

TEMPLATE_TEST_CASE("Filter LessThan, scalar", "[lt][scalar]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    const TestType p1 = 4;
    const TestType p2 = 5;
    dispatch_type<tsl::simd<TestType, tsl::scalar>, tsl::functors::less_than>(cmp_lt<TestType>{}, p1, p2);
  }
}

TEMPLATE_TEST_CASE("Filter GreterThan, scalar", "[gt][scalar]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    const TestType p1 = 4;
    const TestType p2 = 5;
    dispatch_type<tsl::simd<TestType, tsl::scalar>, tsl::functors::greater_than>(cmp_gt<TestType>{}, p1, p2);
  }
}

TEMPLATE_TEST_CASE("Filter Lescalarqual, scalar", "[le][scalar]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t,
                   int16_t, int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    const TestType p1 = 4;
    const TestType p2 = 5;
    dispatch_type<tsl::simd<TestType, tsl::scalar>, tsl::functors::less_than_or_equal>(cmp_le<TestType>{}, p1, p2);
  }
}

TEMPLATE_TEST_CASE("Filter GreterEqual, scalar", "[ge][scalar]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    const TestType p1 = 4;
    const TestType p2 = 5;
    dispatch_type<tsl::simd<TestType, tsl::scalar>, tsl::functors::greater_than_or_equal>(cmp_ge<TestType>{}, p1, p2);
  }
}

TEMPLATE_TEST_CASE("Filter BWI, scalar", "[bwi][scalar]", uint32_t) {
  // TEMPLATE_TEST_CASE("Filter BWI, scalar", "[bwi][scalar]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
  //                    int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    const TestType p1 = 4;
    const TestType p2 = 5;
    const TestType p3 = 0;
    dispatch_type_range<tsl::simd<TestType, tsl::scalar>, tsl::functors::between_inclusive>(cmp_bwi<TestType>{}, p1, p2,
                                                                                            p3);
  }
}

#ifdef TSL_CONTAINS_SSE
TEMPLATE_TEST_CASE("Filter Equality, sse", "[eq][sse]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t,
                   int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    dispatch_type<tsl::simd<TestType, tsl::sse>, tsl::functors::equal>(cmp_eq<TestType>{});
  }
}

TEMPLATE_TEST_CASE("Filter Inequality, sse", "[neq][sse]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t) {
  SECTION(tsl::type_name<TestType>()) {
    dispatch_type<tsl::simd<TestType, tsl::sse>, tsl::functors::nequal>(cmp_neq<TestType>{});
  }
}

TEMPLATE_TEST_CASE("Filter LessThan, sse", "[lt][sse]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t,
                   int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    const TestType p1 = 4;
    const TestType p2 = 5;
    dispatch_type<tsl::simd<TestType, tsl::sse>, tsl::functors::less_than>(cmp_lt<TestType>{}, p1, p2);
  }
}

TEMPLATE_TEST_CASE("Filter GreterThan, sse", "[gt][sse]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    const TestType p1 = 4;
    const TestType p2 = 5;
    dispatch_type<tsl::simd<TestType, tsl::sse>, tsl::functors::greater_than>(cmp_gt<TestType>{}, p1, p2);
  }
}

TEMPLATE_TEST_CASE("Filter LessEqual, sse", "[le][sse]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    const TestType p1 = 4;
    const TestType p2 = 5;
    dispatch_type<tsl::simd<TestType, tsl::sse>, tsl::functors::less_than_or_equal>(cmp_le<TestType>{}, p1, p2);
  }
}

TEMPLATE_TEST_CASE("Filter GreterEqual, sse", "[ge][sse]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    const TestType p1 = 4;
    const TestType p2 = 5;
    dispatch_type<tsl::simd<TestType, tsl::sse>, tsl::functors::greater_than_or_equal>(cmp_ge<TestType>{}, p1, p2);
  }
}

TEMPLATE_TEST_CASE("Filter BWI, sse", "[bwi][sse]", uint32_t) {
  SECTION(tsl::type_name<TestType>()) {
    const TestType p1 = 4;
    const TestType p2 = 5;
    const TestType p3 = 0;
    dispatch_type_range<tsl::simd<TestType, tsl::sse>, tsl::functors::between_inclusive>(cmp_bwi<TestType>{}, p1, p2,
                                                                                         p3);
  }
}
#endif

#ifdef TSL_CONTAINS_AVX2
TEMPLATE_TEST_CASE("Filter Equality, avx2", "[eq][avx2]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    dispatch_type<tsl::simd<TestType, tsl::avx2>, tsl::functors::equal>(cmp_eq<TestType>{});
  }
}

TEMPLATE_TEST_CASE("Filter Inequality, avx2", "[neq][avx2]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    dispatch_type<tsl::simd<TestType, tsl::avx2>, tsl::functors::nequal>(cmp_neq<TestType>{});
  }
}

TEMPLATE_TEST_CASE("Filter LessThan, avx2", "[lt][avx2]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    const TestType p1 = 4;
    const TestType p2 = 5;
    dispatch_type<tsl::simd<TestType, tsl::avx2>, tsl::functors::less_than>(cmp_lt<TestType>{}, p1, p2);
  }
}

TEMPLATE_TEST_CASE("Filter GreterThan, avx2", "[gt][avx2]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    const TestType p1 = 4;
    const TestType p2 = 5;
    dispatch_type<tsl::simd<TestType, tsl::avx2>, tsl::functors::greater_than>(cmp_gt<TestType>{}, p1, p2);
  }
}

TEMPLATE_TEST_CASE("Filter LessEqual, avx2", "[le][avx2]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    const TestType p1 = 4;
    const TestType p2 = 5;
    dispatch_type<tsl::simd<TestType, tsl::avx2>, tsl::functors::less_than_or_equal>(cmp_le<TestType>{}, p1, p2);
  }
}

TEMPLATE_TEST_CASE("Filter GreterEqual, avx2", "[ge][avx2]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    const TestType p1 = 4;
    const TestType p2 = 5;
    dispatch_type<tsl::simd<TestType, tsl::avx2>, tsl::functors::greater_than_or_equal>(cmp_ge<TestType>{}, p1, p2);
  }
}

TEMPLATE_TEST_CASE("Filter BWI, avx2", "[bwi][avx2]", uint32_t) {
  SECTION(tsl::type_name<TestType>()) {
    const TestType p1 = 4;
    const TestType p2 = 5;
    const TestType p3 = 0;
    dispatch_type_range<tsl::simd<TestType, tsl::avx2>, tsl::functors::between_inclusive>(cmp_bwi<TestType>{}, p1, p2,
                                                                                          p3);
  }
}
#endif

#ifdef TSL_CONTAINS_AVX512
TEMPLATE_TEST_CASE("Filter Equality, avx512", "[eq][avx512]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    dispatch_type<tsl::simd<TestType, tsl::avx512>, tsl::functors::equal>(cmp_eq<TestType>{});
  }
}

TEMPLATE_TEST_CASE("Filter Inequality, avx512", "[neq][avx512]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    dispatch_type<tsl::simd<TestType, tsl::avx512>, tsl::functors::nequal>(cmp_neq<TestType>{});
  }
}

TEMPLATE_TEST_CASE("Filter LessThan, avx512", "[lt][avx512]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    const TestType p1 = 4;
    const TestType p2 = 5;
    dispatch_type<tsl::simd<TestType, tsl::avx512>, tsl::functors::less_than>(cmp_lt<TestType>{}, p1, p2);
  }
}

TEMPLATE_TEST_CASE("Filter GreterThan, avx512", "[gt][avx512]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    const TestType p1 = 4;
    const TestType p2 = 5;
    dispatch_type<tsl::simd<TestType, tsl::avx512>, tsl::functors::greater_than>(cmp_gt<TestType>{}, p1, p2);
  }
}

TEMPLATE_TEST_CASE("Filter LessEqual, avx512", "[le][avx512]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    const TestType p1 = 4;
    const TestType p2 = 5;
    dispatch_type<tsl::simd<TestType, tsl::avx512>, tsl::functors::less_than_or_equal>(cmp_le<TestType>{}, p1, p2);
  }
}

TEMPLATE_TEST_CASE("Filter GreterEqual, avx512", "[ge][avx512]", uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
                   int32_t, int64_t, float, double) {
  SECTION(tsl::type_name<TestType>()) {
    const TestType p1 = 4;
    const TestType p2 = 5;
    dispatch_type<tsl::simd<TestType, tsl::avx512>, tsl::functors::greater_than_or_equal>(cmp_ge<TestType>{}, p1, p2);
  }
}

TEMPLATE_TEST_CASE("Filter BWI, avx512", "[bwi][avx512]", uint32_t) {
  SECTION(tsl::type_name<TestType>()) {
    const TestType p1 = 4;
    const TestType p2 = 5;
    const TestType p3 = 0;
    dispatch_type_range<tsl::simd<TestType, tsl::avx512>, tsl::functors::between_inclusive>(cmp_bwi<TestType>{}, p1, p2,
                                                                                            p3);
  }
}
#endif