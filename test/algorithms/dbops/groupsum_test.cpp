#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <sstream>
#include <thread>
#include <tuple>
#include <valarray>
#include <vector>

#include "algorithms/dbops/dbops_hints.hpp"
#include "algorithms/dbops/groupby/groupby.hpp"
#include "algorithms/utils/hashing.hpp"
#include "datastructures/column.hpp"
#include "static/utils/type_concepts.hpp"

#define DATA_ELEMENT_COUNT (1 << 22)
#define GLOBAL_GROUP_COUNT (1 << 15)
#define HASH_BUCKET_COUNT (2 * GLOBAL_GROUP_COUNT)
#define MAX_PARALLELISM_DEGREE 32
#define BENCHMARK_ITERATIONS 3

namespace tuddbs {
  static uint64_t bench_seed{1708006188442894170};

  static uint64_t get_bench_seed() {
    if (bench_seed == 0) {
      bench_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    return bench_seed;
  }

  static std::map<size_t, double> bench_timings;
  static void print_timings(const size_t normalizer) {
    std::stringstream ss;
    for (auto it = bench_timings.begin(); it != bench_timings.end(); ++it) {
      ss << std::setw(2) << it->first << ": " << std::fixed << std::setprecision(2) << std::setw(12)
         << it->second / normalizer << std::endl;
    }
    std::cout << ss.str() << std::endl;
  }
}  // namespace tuddbs

void fill(auto key_column, auto key_column_end, auto value_column, bool contain_zero = false) {
  size_t seed = tuddbs::get_bench_seed();
  std::mt19937 mt(seed);
  // std::cerr << "Seed: " << seed << std::endl;
  std::uniform_int_distribution<> dist;
  if (contain_zero) {
    dist = std::uniform_int_distribution<>(0, GLOBAL_GROUP_COUNT);
  } else {
    dist = std::uniform_int_distribution<>(1, GLOBAL_GROUP_COUNT);
  }
  for (; key_column != key_column_end; ++key_column, ++value_column) {
    *key_column = dist(mt);
    *value_column = *key_column & 15;
  }
}

template <typename RangeTupleT>
struct GroupBySumScalar : Catch::Matchers::MatcherGenericBase {
  std::map<uint64_t, uint64_t> m_group_sums;
  std::map<uint64_t, std::vector<uint64_t>> m_group_values;
  GroupBySumScalar(RangeTupleT data_iterators) : m_group_sums() {
    auto key_it = std::get<0>(data_iterators);
    auto key_end = std::get<1>(data_iterators);
    auto value_it = std::get<2>(data_iterators);
    for (; key_it != key_end; ++key_it, ++value_it) {
      auto bucket = m_group_sums.find(*key_it);
      if (bucket == m_group_sums.end()) {
        m_group_sums[*key_it] = *value_it;
        m_group_values.emplace(*key_it, std::vector<uint64_t>());
      } else {
        bucket->second += *value_it;
      }
      m_group_values[*key_it].push_back(*value_it);
    }
  }

  template <typename KeyValueTupleT>
  bool match(KeyValueTupleT const &result) const {
    auto keys_it = std::get<0>(result);
    auto keys_end = std::get<1>(result);
    auto values_it = std::get<2>(result);

    std::map<uint64_t, uint64_t> result_group_sums;

    for (; keys_it != keys_end; ++keys_it, ++values_it) {
      auto bucket = result_group_sums.find(*keys_it);
      if (bucket == result_group_sums.end()) {
        result_group_sums[*keys_it] = *values_it;
      } else {
        std::cerr << "Duplicate key " << *keys_it << " found in simdops-groupby result" << std::endl;
        return false;
      }
    }

    if (result_group_sums.size() != m_group_sums.size()) {
      std::cerr << "simdops-groupby result has " << result_group_sums.size() << " groups, but scalar result has "
                << m_group_sums.size() << " groups" << std::endl;
      return false;
    }

    if (std::equal(m_group_sums.cbegin(), m_group_sums.cend(), result_group_sums.cbegin())) {
      return true;
    } else {
      for (auto it = m_group_sums.cbegin(); it != m_group_sums.cend(); ++it) {
        auto key = it->first;
        auto bucket = result_group_sums.find(key);
        if (bucket == result_group_sums.end()) {
          std::cerr << "Key " << key << " not found in simdops-groupby result" << std::endl;
          return false;
        }
        if (bucket->second != it->second) {
          std::cerr << "Mismatch at Key " << key << " with scalar sum " << it->second << " and simdops-groupby sum "
                    << bucket->second << std::endl;
          std::cerr << "Corresponding data: " << std::endl;
          auto scalar_entries_it = m_group_values.find(key);
          auto entries = scalar_entries_it->second;
          bool first = true;
          for (auto entry : entries) {
            if (first) {
              first = false;
            } else {
              std::cerr << ", ";
            }
            std::cerr << entry << " ";
          }
          std::cerr << std::endl;
          return false;
        }
      }
    }
    return true;
  }

  std::string describe() const override { return "Result of grouping does not equal to scalar variant."; }
};

template <typename RangeTupleT>
auto GroupBySumMatcher(RangeTupleT const &data) -> GroupBySumScalar<RangeTupleT> {
  return GroupBySumScalar<RangeTupleT>{data};
}

template <typename Key, typename Val = Key>
struct group_aggregate_data_set_t {
  tuddbs::InMemoryColumn<Key> keys_sink;
  tuddbs::InMemoryColumn<Val> value_sink;

  explicit group_aggregate_data_set_t(const size_t element_count, bool contain_zero, auto allocator, auto deleter)
    : keys_sink(element_count, allocator, deleter), value_sink(element_count, allocator, deleter) {
    fill(keys_sink.begin(), keys_sink.end(), value_sink.begin(), contain_zero);
  }

  ~group_aggregate_data_set_t() {}
};

template <typename Key, typename Val = Key>
struct group_aggregate_map_t {
  tuddbs::InMemoryColumn<Key> keys_sink;
  tuddbs::InMemoryColumn<Val> value_sink;

  explicit group_aggregate_map_t(const size_t map_count, auto allocator, auto deleter)
    : keys_sink(map_count, allocator, deleter), value_sink(map_count, allocator, deleter) {}

  ~group_aggregate_map_t() {}
};

template <typename Key, typename Val = Key>
struct group_aggregate_result_set_t {
  tuddbs::InMemoryColumn<Key> keys_sink;
  tuddbs::InMemoryColumn<Val> value_sink;

  explicit group_aggregate_result_set_t(const size_t map_count, auto allocator, auto deleter)
    : keys_sink(map_count, allocator, deleter), value_sink(map_count, allocator, deleter) {}

  ~group_aggregate_result_set_t() {}
};

TEST_CASE("GroupBy-Sum for uint64_t with sse, single thread", "[cpu][groupby_sum][uint64_t][sse][sequential]") {
  std::cout << "[sse] uint64_t with single thread" << std::endl;
  using key_t = uint64_t;
  using value_t = uint64_t;
  using extension = tsl::sse;
  using namespace tuddbs;

  auto group_allocator = []<tsl::TSLArithmetic T>(size_t i) -> T * {
    return reinterpret_cast<T *>(_mm_malloc(i * sizeof(T), 64));
  };
  auto group_deleter = []<tsl::TSLArithmetic T>(T *ptr) { _mm_free(ptr); };

  SECTION("Data does not contain ZERO") {
    group_aggregate_data_set_t<key_t, value_t> data_columns(DATA_ELEMENT_COUNT, false, group_allocator, group_deleter);
    group_aggregate_map_t<key_t, value_t> intermediate_map(HASH_BUCKET_COUNT, group_allocator, group_deleter);

    GroupAggregate_Sum<tsl::simd<key_t, extension>, value_t,
                       OperatorHintSet<hints::hashing::size_exp_2, hints::hashing::linear_displacement>>::builder_t
      builder(intermediate_map.keys_sink.begin(), intermediate_map.value_sink.begin(), HASH_BUCKET_COUNT);
    builder(data_columns.keys_sink.cbegin(), data_columns.keys_sink.cend(), data_columns.value_sink.cbegin());

    auto const key_count = builder.distinct_key_count();
    group_aggregate_result_set_t<key_t, value_t> result_columns(key_count, group_allocator, group_deleter);

    GroupAggregate_Sum<tsl::simd<key_t, extension>, value_t,
                       OperatorHintSet<hints::hashing::size_exp_2, hints::hashing::linear_displacement>>::grouper_t
      grouper(intermediate_map.keys_sink.begin(), intermediate_map.value_sink.begin(), HASH_BUCKET_COUNT);
    grouper(result_columns.keys_sink.begin(), result_columns.value_sink.begin());
    REQUIRE_THAT(std::make_tuple(result_columns.keys_sink.cbegin(), result_columns.keys_sink.cend(),
                                 result_columns.value_sink.cbegin()),
                 GroupBySumMatcher(std::make_tuple(data_columns.keys_sink.cbegin(), data_columns.keys_sink.cend(),
                                                   data_columns.value_sink.cbegin())));
  }
  SECTION("Data does contain ZERO") {
    group_aggregate_data_set_t<key_t, value_t> data_columns(DATA_ELEMENT_COUNT, true, group_allocator, group_deleter);
    group_aggregate_map_t<key_t, value_t> intermediate_map(HASH_BUCKET_COUNT, group_allocator, group_deleter);

    GroupAggregate_Sum<tsl::simd<key_t, extension>, value_t,
                       OperatorHintSet<hints::hashing::size_exp_2, hints::hashing::linear_displacement,
                                       hints::hashing::keys_may_contain_zero>>::builder_t
      builder(intermediate_map.keys_sink.begin(), intermediate_map.value_sink.begin(), HASH_BUCKET_COUNT);
    builder(data_columns.keys_sink.cbegin(), data_columns.keys_sink.cend(), data_columns.value_sink.cbegin());

    auto const key_count = builder.distinct_key_count();
    group_aggregate_result_set_t<key_t, value_t> result_columns(key_count, group_allocator, group_deleter);

    GroupAggregate_Sum<tsl::simd<key_t, extension>, value_t,
                       OperatorHintSet<hints::hashing::size_exp_2, hints::hashing::linear_displacement,
                                       hints::hashing::keys_may_contain_zero>>::grouper_t
      grouper(intermediate_map.keys_sink.begin(), intermediate_map.value_sink.begin(), HASH_BUCKET_COUNT);
    grouper(result_columns.keys_sink.begin(), result_columns.value_sink.begin());
    REQUIRE_THAT(std::make_tuple(result_columns.keys_sink.cbegin(), result_columns.keys_sink.cend(),
                                 result_columns.value_sink.cbegin()),
                 GroupBySumMatcher(std::make_tuple(data_columns.keys_sink.cbegin(), data_columns.keys_sink.cend(),
                                                   data_columns.value_sink.cbegin())));
  }
}
TEST_CASE("GroupBy-Sum for uint64_t with avx, single thread", "[cpu][groupby_sum][uint64_t][avx][sequential]") {
  std::cout << "[avx] uint64_t with single thread" << std::endl;
  using key_t = uint64_t;
  using value_t = uint64_t;
  using extension = tsl::avx2;
  using namespace tuddbs;

  auto group_allocator = []<tsl::TSLArithmetic T>(size_t i) -> T * {
    return reinterpret_cast<T *>(_mm_malloc(i * sizeof(T), 64));
  };
  auto group_deleter = []<tsl::TSLArithmetic T>(T *ptr) { _mm_free(ptr); };

  SECTION("Data does not contain ZERO") {
    group_aggregate_data_set_t<key_t, value_t> data_columns(DATA_ELEMENT_COUNT, false, group_allocator, group_deleter);
    group_aggregate_map_t<key_t, value_t> intermediate_map(HASH_BUCKET_COUNT, group_allocator, group_deleter);

    GroupAggregate_Sum<tsl::simd<key_t, extension>, value_t,
                       OperatorHintSet<hints::hashing::size_exp_2, hints::hashing::linear_displacement>>::builder_t
      builder(intermediate_map.keys_sink.begin(), intermediate_map.value_sink.begin(), HASH_BUCKET_COUNT);
    builder(data_columns.keys_sink.cbegin(), data_columns.keys_sink.cend(), data_columns.value_sink.cbegin());

    auto const key_count = builder.distinct_key_count();
    group_aggregate_result_set_t<key_t, value_t> result_columns(key_count, group_allocator, group_deleter);

    GroupAggregate_Sum<tsl::simd<key_t, extension>, value_t,
                       OperatorHintSet<hints::hashing::size_exp_2, hints::hashing::linear_displacement>>::grouper_t
      grouper(intermediate_map.keys_sink.begin(), intermediate_map.value_sink.begin(), HASH_BUCKET_COUNT);
    grouper(result_columns.keys_sink.begin(), result_columns.value_sink.begin());
    REQUIRE_THAT(std::make_tuple(result_columns.keys_sink.cbegin(), result_columns.keys_sink.cend(),
                                 result_columns.value_sink.cbegin()),
                 GroupBySumMatcher(std::make_tuple(data_columns.keys_sink.cbegin(), data_columns.keys_sink.cend(),
                                                   data_columns.value_sink.cbegin())));
  }
  SECTION("Data does contain ZERO") {
    group_aggregate_data_set_t<key_t, value_t> data_columns(DATA_ELEMENT_COUNT, true, group_allocator, group_deleter);
    group_aggregate_map_t<key_t, value_t> intermediate_map(HASH_BUCKET_COUNT, group_allocator, group_deleter);

    GroupAggregate_Sum<tsl::simd<key_t, extension>, value_t,
                       OperatorHintSet<hints::hashing::size_exp_2, hints::hashing::linear_displacement,
                                       hints::hashing::keys_may_contain_zero>>::builder_t
      builder(intermediate_map.keys_sink.begin(), intermediate_map.value_sink.begin(), HASH_BUCKET_COUNT);
    builder(data_columns.keys_sink.cbegin(), data_columns.keys_sink.cend(), data_columns.value_sink.cbegin());

    auto const key_count = builder.distinct_key_count();
    group_aggregate_result_set_t<key_t, value_t> result_columns(key_count, group_allocator, group_deleter);

    GroupAggregate_Sum<tsl::simd<key_t, extension>, value_t,
                       OperatorHintSet<hints::hashing::size_exp_2, hints::hashing::linear_displacement,
                                       hints::hashing::keys_may_contain_zero>>::grouper_t
      grouper(intermediate_map.keys_sink.begin(), intermediate_map.value_sink.begin(), HASH_BUCKET_COUNT);
    grouper(result_columns.keys_sink.begin(), result_columns.value_sink.begin());
    REQUIRE_THAT(std::make_tuple(result_columns.keys_sink.cbegin(), result_columns.keys_sink.cend(),
                                 result_columns.value_sink.cbegin()),
                 GroupBySumMatcher(std::make_tuple(data_columns.keys_sink.cbegin(), data_columns.keys_sink.cend(),
                                                   data_columns.value_sink.cbegin())));
  }
}