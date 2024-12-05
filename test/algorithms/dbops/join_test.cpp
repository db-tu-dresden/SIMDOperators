#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#include "algorithms/dbops/dbops_hints.hpp"
#include "algorithms/dbops/join/hash_join.hpp"
#include "algorithms/utils/hashing.hpp"
#include "datastructures/column.hpp"

#define ELEMENT_MIN(a, b) ((a) < (b) ? (a) : (b))
#define DATA_ELEMENT_COUNT_A (1 << 20)
#define DATA_ELEMENT_COUNT_B (DATA_ELEMENT_COUNT_A << 3)
#define GLOBAL_GROUP_COUNT DATA_ELEMENT_COUNT_A
#define HASH_BUCKET_COUNT (GLOBAL_GROUP_COUNT << 1)
#define DATA_GENERATION_AMOUNT (HASH_BUCKET_COUNT << 2)
#define MAX_PARALLELISM_DEGREE 32
#define BENCHMARK_ITERATIONS 3

namespace tuddbs {
  static uint64_t bench_seed{0};

  static uint64_t get_bench_seed() {
    if (bench_seed == 0) {
      bench_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    return bench_seed;
  }
}  // namespace tuddbs

template <typename T>
auto join_alloc_fn(size_t i) -> T * {
  return static_cast<T *>(_mm_malloc(i * sizeof(T), 64));
}

template <typename T>
auto join_dealloc_fn(T *ptr) {
  _mm_free(ptr);
}

template <typename T>
void random_data_generator(tuddbs::InMemoryColumn<T> &build_column, tuddbs::InMemoryColumn<T> &probe_column,
                           const T min_value = 0, const size_t run = 0) {
  size_t seed = tuddbs::get_bench_seed();
  std::cout << "(Seed: " << seed << ", Run: " << run << " ) " << std::flush;

  std::uniform_int_distribution<size_t> dist_b(0, build_column.count() - 1);
  std::mt19937 mt(seed + run);

  std::iota(build_column.begin(), build_column.end(), min_value);
  std::shuffle(build_column.begin(), build_column.end(), mt);
  for (auto it = probe_column.begin(); it != probe_column.end(); ++it) {
    (*it) = build_column.get_value(dist_b(mt));
  }
}

template <typename T>
auto verify(tuddbs::InMemoryColumn<size_t> &build_column_result, tuddbs::InMemoryColumn<size_t> &probe_column_result,
            size_t const result_count, tuddbs::InMemoryColumn<T> const &build_column,
            tuddbs::InMemoryColumn<T> const &probe_column) {
  std::unordered_map<T, size_t> stl_hashmap;
  stl_hashmap.reserve(build_column.count());
  tuddbs::InMemoryColumn<size_t> reference_build_column_result(probe_column.count(), join_alloc_fn<size_t>,
                                                               join_dealloc_fn<size_t>);
  tuddbs::InMemoryColumn<size_t> reference_probe_column_result(probe_column.count(), join_alloc_fn<size_t>,
                                                               join_dealloc_fn<size_t>);

  const auto t_start = std::chrono::high_resolution_clock::now();
  for (auto it = build_column.cbegin(); it != build_column.cend(); ++it) {
    stl_hashmap[*it] = it - build_column.cbegin();
  }
  auto reference_build_column_result_it = reference_build_column_result.begin();
  auto reference_probe_column_result_it = reference_probe_column_result.begin();
  auto count = 0;
  for (auto it = probe_column.cbegin(); it != probe_column.cend(); ++it) {
    if (auto search = stl_hashmap.find(*it); search != stl_hashmap.end()) {
      *reference_build_column_result_it = search->second;
      *reference_probe_column_result_it = it - probe_column.cbegin();
      ++reference_build_column_result_it;
      ++reference_probe_column_result_it;
    }

    ++count;
  }

  const auto t_end = std::chrono::high_resolution_clock::now();
  auto const reference_result_count = reference_build_column_result_it - reference_build_column_result.begin();

  REQUIRE((reference_build_column_result_it - reference_build_column_result.begin()) == result_count);

  std::vector<std::pair<size_t, size_t>> result_combined;
  for (size_t i = 0; i < result_count; i++) {
    result_combined.emplace_back(build_column_result.get_value(i), probe_column_result.get_value(i));
  }
  std::sort(result_combined.begin(), result_combined.end(),
            [](const std::pair<size_t, size_t> &x, const std::pair<size_t, size_t> &y) {
              return x.first < y.first || (x.first == y.first && x.second < y.second);
            });

  std::vector<std::pair<size_t, size_t>> reference_result_combined;
  for (size_t i = 0; i < reference_result_count; i++) {
    reference_result_combined.emplace_back(reference_build_column_result.get_value(i),
                                           reference_probe_column_result.get_value(i));
  }
  std::sort(reference_result_combined.begin(), reference_result_combined.end(),
            [](const std::pair<size_t, size_t> &x, const std::pair<size_t, size_t> &y) {
              return x.first < y.first || (x.first == y.first && x.second < y.second);
            });

  auto const min_result_count =
    std::min(reference_build_column_result_it - reference_build_column_result.begin(), result_count);
  auto result_it = result_combined.begin();
  auto reference_result_it = reference_result_combined.begin();

  bool equal = std::equal(result_combined.begin(), result_combined.end(), reference_result_combined.begin());
  if (!equal) {
    auto i = 0;
    auto error_count = 0;
    while (error_count < 100) {
      if (result_it->first != reference_result_it->first || result_it->second != reference_result_it->second) {
        std::cout << "Mismatch at index " << i << ": " << +(result_it->first) << " " << +(result_it->second)
                  << " != " << +(reference_result_it->first) << " " << +(reference_result_it->second) << std::endl;
        ++error_count;
      }
      ++result_it;
      ++reference_result_it;
      ++i;
    }
  }
  REQUIRE(equal);
  return std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
}

template <typename T, typename Extension, bool ContainsZero>
auto test_join() {
  std::cout << "Testing " << tsl::type_name<Extension>() << " with " << tsl::type_name<T>()
            << (ContainsZero ? " with zero key" : "") << std::endl;
  using SimdExt = tsl::simd<T, Extension>;

  using ValueT =
    std::conditional_t<std::is_floating_point_v<T>, std::conditional_t<std::is_same_v<T, float>, uint32_t, uint64_t>,
                       std::make_unsigned_t<T>>;

  using hash_hint_set = std::conditional_t<
    ContainsZero,
    tuddbs::OperatorHintSet<tuddbs::hints::hashing::linear_displacement, tuddbs::hints::hashing::size_exp_2,
                            tuddbs::hints::hash_join::keys_may_contain_empty_indicator,
                            tuddbs::hints::hash_join::global_first_occurence_required>,
    tuddbs::OperatorHintSet<tuddbs::hints::hashing::linear_displacement, tuddbs::hints::hashing::size_exp_2,
                            tuddbs::hints::hash_join::global_first_occurence_required>>;

  using hash_join_t = typename tuddbs::Hash_Join<SimdExt, ValueT, hash_hint_set>;

  auto const build_element_count = (ContainsZero)
                                     ? std::clamp((long unsigned int)DATA_ELEMENT_COUNT_A, (long unsigned int)0,
                                                  (long unsigned int)std::numeric_limits<T>::max())
                                     : std::clamp((long unsigned int)DATA_ELEMENT_COUNT_A, (long unsigned int)1,
                                                  (long unsigned int)std::numeric_limits<T>::max());

  auto const hash_bucket_count =
    1ULL << ((sizeof(T) * CHAR_BIT) - std::countl_zero<std::make_unsigned_t<T>>(build_element_count));

  tuddbs::InMemoryColumn<T> build_column(build_element_count, join_alloc_fn<T>, join_dealloc_fn<T>);
  tuddbs::InMemoryColumn<T> probe_column(DATA_ELEMENT_COUNT_B, join_alloc_fn<T>, join_dealloc_fn<T>);

  tuddbs::InMemoryColumn<T> hash_map_key_column(hash_bucket_count, join_alloc_fn<T>, join_dealloc_fn<T>);
  tuddbs::InMemoryColumn<ValueT> hash_map_value_column(hash_bucket_count, join_alloc_fn<ValueT>,
                                                       join_dealloc_fn<ValueT>);
  tuddbs::InMemoryColumn<T> hash_map_used_column(hash_bucket_count, join_alloc_fn<T>, join_dealloc_fn<T>);

  tuddbs::InMemoryColumn<size_t> hash_join_result_build_column(DATA_ELEMENT_COUNT_B, join_alloc_fn<size_t>,
                                                               join_dealloc_fn<size_t>);
  tuddbs::InMemoryColumn<size_t> hash_join_result_probe_column(DATA_ELEMENT_COUNT_B, join_alloc_fn<size_t>,
                                                               join_dealloc_fn<size_t>);

  double timings_sum = 0.0;
  double reference_timings_sum = 0.0;
  std::cout << "[INFO] Benchmark iteration " << std::flush;
  for (size_t benchIt = 0; benchIt < BENCHMARK_ITERATIONS; ++benchIt) {
    std::cout << benchIt + 1 << "... " << std::flush;
    random_data_generator<T>(build_column, probe_column, (ContainsZero) ? 0 : 1, benchIt);
    typename hash_join_t::builder_t builder(hash_map_key_column.begin(), hash_map_used_column.begin(),
                                            hash_map_value_column.begin(), hash_bucket_count);
    typename hash_join_t::prober_t prober(hash_map_key_column.begin(), hash_map_used_column.begin(),
                                          hash_map_value_column.begin(), hash_bucket_count);

    const auto t_start = std::chrono::high_resolution_clock::now();
    builder(build_column.cbegin(), build_column.cend());
    size_t join_result_count = prober(hash_join_result_build_column.begin(), hash_join_result_probe_column.begin(),
                                      probe_column.cbegin(), probe_column.cend());
    const auto t_end = std::chrono::high_resolution_clock::now();

    timings_sum += std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
    reference_timings_sum += verify<T>(hash_join_result_build_column, hash_join_result_probe_column, join_result_count,
                                       build_column, probe_column);
  }
  std::cout << "done" << std::endl;
  std::cout << "Average execution time          : " << std::fixed << std::setprecision(2)
            << timings_sum / BENCHMARK_ITERATIONS << "us" << std::endl;
  std::cout << "Average reference execution time: " << std::fixed << std::setprecision(2)
            << reference_timings_sum / BENCHMARK_ITERATIONS << "us" << std::endl;
  std::cout << "Speedup: " << reference_timings_sum / timings_sum << "x" << std::endl;
}

#ifdef TSL_CONTAINS_AVX512
TEST_CASE("Join for uint64_t", "[cpu][join][uint64_t][avx512]") { test_join<uint64_t, tsl::avx512, false>(); }
TEST_CASE("Join for uint64_t", "[cpu][join][uint64_t][avx512][zero_key]") { test_join<uint64_t, tsl::avx512, true>(); }

TEST_CASE("Join for uint32_t", "[cpu][join][uint32_t][avx512]") { test_join<uint32_t, tsl::avx512, false>(); }
TEST_CASE("Join for uint32_t", "[cpu][join][uint32_t][avx512][zero_key]") { test_join<uint32_t, tsl::avx512, true>(); }

TEST_CASE("Join for uint16_t", "[cpu][join][uint16_t][avx512]") { test_join<uint16_t, tsl::avx512, false>(); }

TEST_CASE("Join for uint16_t", "[cpu][join][uint16_t][avx512][zero_key]") { test_join<uint16_t, tsl::avx512, true>(); }

TEST_CASE("Join for uint8_t", "[cpu][join][uint8_t][avx512]") { test_join<uint8_t, tsl::avx512, false>(); }
TEST_CASE("Join for uint8_t", "[cpu][join][uint8_t][avx512][zero_key]") { test_join<uint8_t, tsl::avx512, true>(); }

TEST_CASE("Join for int64_t", "[cpu][join][int64_t][avx512]") { test_join<int64_t, tsl::avx512, false>(); }
TEST_CASE("Join for int64_t", "[cpu][join][int64_t][avx512][zero_key]") { test_join<int64_t, tsl::avx512, true>(); }

TEST_CASE("Join for int32_t", "[cpu][join][int32_t][avx512]") { test_join<int32_t, tsl::avx512, false>(); }
TEST_CASE("Join for int32_t", "[cpu][join][int32_t][avx512][zero_key]") { test_join<int32_t, tsl::avx512, true>(); }

TEST_CASE("Join for int16_t", "[cpu][join][int16_t][avx512]") { test_join<int16_t, tsl::avx512, false>(); }
TEST_CASE("Join for int16_t", "[cpu][join][int16_t][avx512][zero_key]") { test_join<int16_t, tsl::avx512, true>(); }

TEST_CASE("Join for int8_t", "[cpu][join][int8_t][avx512]") { test_join<int8_t, tsl::avx512, false>(); }
TEST_CASE("Join for int8_t", "[cpu][join][int8_t][avx512][zero_key]") { test_join<int8_t, tsl::avx512, true>(); }
#endif
#ifdef TSL_CONTAINS_AVX2
TEST_CASE("Join for uint64_t", "[cpu][join][uint64_t][avx2]") { test_join<uint64_t, tsl::avx2, false>(); }
TEST_CASE("Join for uint64_t", "[cpu][join][uint64_t][avx2][zero_key]") { test_join<uint64_t, tsl::avx2, true>(); }

TEST_CASE("Join for uint32_t", "[cpu][join][uint32_t][avx2]") { test_join<uint32_t, tsl::avx2, false>(); }
TEST_CASE("Join for uint32_t", "[cpu][join][uint32_t][avx2][zero_key]") { test_join<uint32_t, tsl::avx2, true>(); }

TEST_CASE("Join for uint16_t", "[cpu][join][uint16_t][avx2]") { test_join<uint16_t, tsl::avx2, false>(); }
TEST_CASE("Join for uint16_t", "[cpu][join][uint16_t][avx2][zero_key]") { test_join<uint16_t, tsl::avx2, true>(); }

TEST_CASE("Join for uint8_t", "[cpu][join][uint8_t][avx2]") { test_join<uint8_t, tsl::avx2, false>(); }
TEST_CASE("Join for uint8_t", "[cpu][join][uint8_t][avx2][zero_key]") { test_join<uint8_t, tsl::avx2, true>(); }

TEST_CASE("Join for int64_t", "[cpu][join][int64_t][avx2]") { test_join<int64_t, tsl::avx2, false>(); }
TEST_CASE("Join for int64_t", "[cpu][join][int64_t][avx2][zero_key]") { test_join<int64_t, tsl::avx2, true>(); }

TEST_CASE("Join for int32_t", "[cpu][join][int32_t][avx2]") { test_join<int32_t, tsl::avx2, false>(); }
TEST_CASE("Join for int32_t", "[cpu][join][int32_t][avx2][zero_key]") { test_join<int32_t, tsl::avx2, true>(); }

TEST_CASE("Join for int16_t", "[cpu][join][int16_t][avx2]") { test_join<int16_t, tsl::avx2, false>(); }
TEST_CASE("Join for int16_t", "[cpu][join][int16_t][avx2][zero_key]") { test_join<int16_t, tsl::avx2, true>(); }

TEST_CASE("Join for int8_t", "[cpu][join][int8_t][avx2]") { test_join<int8_t, tsl::avx2, false>(); }
TEST_CASE("Join for int8_t", "[cpu][join][int8_t][avx2][zero_key]") { test_join<int8_t, tsl::avx2, true>(); }
#endif

#ifdef TSL_CONTAINS_SSE
TEST_CASE("Join for uint64_t", "[cpu][join][uint64_t][sse]") { test_join<uint64_t, tsl::sse, false>(); }
TEST_CASE("Join for uint64_t", "[cpu][join][uint64_t][sse][zero_key]") { test_join<uint64_t, tsl::sse, true>(); }

TEST_CASE("Join for uint32_t", "[cpu][join][uint32_t][sse]") { test_join<uint32_t, tsl::sse, false>(); }
TEST_CASE("Join for uint32_t", "[cpu][join][uint32_t][sse][zero_key]") { test_join<uint32_t, tsl::sse, true>(); }

TEST_CASE("Join for uint16_t", "[cpu][join][uint16_t][sse]") { test_join<uint16_t, tsl::sse, false>(); }
TEST_CASE("Join for uint16_t", "[cpu][join][uint16_t][sse][zero_key]") { test_join<uint16_t, tsl::sse, true>(); }

TEST_CASE("Join for uint8_t", "[cpu][join][uint8_t][sse]") { test_join<uint8_t, tsl::sse, false>(); }
TEST_CASE("Join for uint8_t", "[cpu][join][uint8_t][sse][zero_key]") { test_join<uint8_t, tsl::sse, true>(); }

TEST_CASE("Join for int64_t", "[cpu][join][int64_t][sse]") { test_join<int64_t, tsl::sse, false>(); }
TEST_CASE("Join for int64_t", "[cpu][join][int64_t][sse][zero_key]") { test_join<int64_t, tsl::sse, true>(); }

TEST_CASE("Join for int32_t", "[cpu][join][int32_t][sse]") { test_join<int32_t, tsl::sse, false>(); }
TEST_CASE("Join for int32_t", "[cpu][join][int32_t][sse][zero_key]") { test_join<int32_t, tsl::sse, true>(); }

TEST_CASE("Join for int16_t", "[cpu][join][int16_t][sse]") { test_join<int16_t, tsl::sse, false>(); }
TEST_CASE("Join for int16_t", "[cpu][join][int16_t][sse][zero_key]") { test_join<int16_t, tsl::sse, true>(); }

TEST_CASE("Join for int8_t", "[cpu][join][int8_t][sse]") { test_join<int8_t, tsl::sse, false>(); }
TEST_CASE("Join for int8_t", "[cpu][join][int8_t][sse][zero_key]") { test_join<int8_t, tsl::sse, true>(); }
#endif

TEST_CASE("Join for uint64_t", "[cpu][join][uint64_t][scalar]") { test_join<uint64_t, tsl::scalar, false>(); }
TEST_CASE("Join for uint64_t", "[cpu][join][uint64_t][scalar][zero_key]") { test_join<uint64_t, tsl::scalar, true>(); }

TEST_CASE("Join for uint32_t", "[cpu][join][uint32_t][scalar]") { test_join<uint32_t, tsl::scalar, false>(); }
TEST_CASE("Join for uint32_t", "[cpu][join][uint32_t][scalar][zero_key]") { test_join<uint32_t, tsl::scalar, true>(); }

TEST_CASE("Join for uint16_t", "[cpu][join][uint16_t][scalar]") { test_join<uint16_t, tsl::scalar, false>(); }
TEST_CASE("Join for uint16_t", "[cpu][join][uint16_t][scalar][zero_key]") { test_join<uint16_t, tsl::scalar, true>(); }

TEST_CASE("Join for uint8_t", "[cpu][join][uint8_t][scalar]") { test_join<uint8_t, tsl::scalar, false>(); }
TEST_CASE("Join for uint8_t", "[cpu][join][uint8_t][scalar][zero_key]") { test_join<uint8_t, tsl::scalar, true>(); }

TEST_CASE("Join for int64_t", "[cpu][join][int64_t][scalar]") { test_join<int64_t, tsl::scalar, false>(); }
TEST_CASE("Join for int64_t", "[cpu][join][int64_t][scalar][zero_key]") { test_join<int64_t, tsl::scalar, true>(); }

TEST_CASE("Join for int32_t", "[cpu][join][int32_t][scalar]") { test_join<int32_t, tsl::scalar, false>(); }
TEST_CASE("Join for int32_t", "[cpu][join][int32_t][scalar][zero_key]") { test_join<int32_t, tsl::scalar, true>(); }

TEST_CASE("Join for int16_t", "[cpu][join][int16_t][scalar]") { test_join<int16_t, tsl::scalar, false>(); }
TEST_CASE("Join for int16_t", "[cpu][join][int16_t][scalar][zero_key]") { test_join<int16_t, tsl::scalar, true>(); }

TEST_CASE("Join for int8_t", "[cpu][join][int8_t][scalar]") { test_join<int8_t, tsl::scalar, false>(); }
TEST_CASE("Join for int8_t", "[cpu][join][int8_t][scalar][zero_key]") { test_join<int8_t, tsl::scalar, true>(); }
