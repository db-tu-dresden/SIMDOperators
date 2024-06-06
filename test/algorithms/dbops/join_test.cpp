#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
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
#define BENCHMARK_ITERATIONS 5

namespace tuddbs {
  static uint64_t bench_seed{0};

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

template <typename T>
struct hash_join_column_set_t {
  tuddbs::InMemoryColumn<T> key_sink;
  tuddbs::InMemoryColumn<T> value_sink;
  tuddbs::InMemoryColumn<T> used_sink;

  explicit hash_join_column_set_t(const size_t map_count, auto allocator, auto deleter)
    : key_sink(map_count, allocator, deleter),
      value_sink(map_count, allocator, deleter),
      used_sink(map_count, allocator, deleter) {}

  ~hash_join_column_set_t() {}
};

template <typename T>
T random_data_generator(T *raw_data, const size_t raw_data_amount, tuddbs::InMemoryColumn<T> &column_a,
                        const size_t column_a_size, tuddbs::InMemoryColumn<T> &column_b, size_t column_b_size,
                        size_t hash_bucket_count, const T start_at = 0, const size_t run = 0) {
  size_t seed = tuddbs::get_bench_seed();
  std::mt19937 mt(seed + run);
  std::uniform_int_distribution<> dist_b(0, hash_bucket_count - 1);

  std::iota(raw_data, raw_data + raw_data_amount, start_at);
  std::shuffle(raw_data, raw_data + raw_data_amount, mt);
  std::sort(raw_data, raw_data + hash_bucket_count);

  T max_findable = raw_data[column_a_size - 1];
  std::copy(raw_data, raw_data + column_a_size, column_a.begin());

  for (auto it = column_b.begin(); it != column_b.end(); ++it) {
    (*it) = raw_data[dist_b(mt)];
  }
  return max_findable;
}

template <typename T>
void result_checker(bool &all_found, bool &wrong_result, tuddbs::InMemoryColumn<T> &column_a,
                    const size_t column_a_size, tuddbs::InMemoryColumn<T> &column_b, size_t column_b_size,
                    T *join_result_a, T *join_result_b, size_t result_size, T max_findable_value) {
  for (size_t i = 0; i < result_size; i++) {
    if (*(column_a.begin() + join_result_a[i]) != *(column_b.begin() + join_result_b[i])) {
      std::cerr << "Result Mismatch at " << i << "\tKey A " << *(column_a.begin() + join_result_a[i]) << "\tKey B "
                << *(column_b.begin() + join_result_b[i]) << std::endl;
      wrong_result = true;
    }
  }

  size_t local_join_result = 0;
  size_t missing_result = 0;
  auto it = column_b.cbegin();
  for (; it != column_b.cend(); ++it) {
    const size_t curr_pos = it - column_b.cbegin();
    if (*it <= max_findable_value) {  // data should be in the table
      if (local_join_result >= result_size) {
        std::cerr << "Missing Result Total: " << ++missing_result << std::endl;
        wrong_result = true;
      } else if (curr_pos != join_result_b[local_join_result]) {  // BUT we didn't find the key
        all_found = false;
        std::cerr << *it << " didn't produce a join partner or position list out of order. Result position: "
                  << join_result_b[local_join_result] << " <>  is position: " << curr_pos << std::endl;
      } else {  // we found it in the result
        local_join_result++;
      }
    } else {  // data is not in the table
      if (join_result_b[local_join_result] == curr_pos && local_join_result < result_size) {  // BUT we found a match
        wrong_result = true;
        std::cerr << "Found Element in Table that shouldn't be in the Table at result index: " << local_join_result
                  << " Key A: " << *(column_a.begin() + join_result_a[local_join_result])
                  << " Key B: " << *(column_b.begin() + join_result_b[local_join_result]) << std::endl;
        local_join_result++;
      }
    }
  }
}

// #### std::unordered_map implementation
TEST_CASE("Join Default for uint64_t", "[cpu][groupby][uint64_t][stl-seq]") {
  std::cout << "[stl-seq] uint64_t Sequential" << std::endl;
  using base_t = uint64_t;

  // data allocator
  auto join_allocator = [](size_t i) -> base_t * {
    return reinterpret_cast<base_t *>(_mm_malloc(i * sizeof(base_t), 64));
  };
  auto join_deleter = [](base_t *ptr) { _mm_free(ptr); };

  // Columns

  base_t *raw_data = join_allocator(DATA_GENERATION_AMOUNT);
  tuddbs::InMemoryColumn<base_t> column_to_join_a(DATA_ELEMENT_COUNT_A, join_allocator, join_deleter);
  tuddbs::InMemoryColumn<base_t> column_to_join_b(DATA_ELEMENT_COUNT_B, join_allocator, join_deleter);

  // ResultArrays.
  for (size_t benchIt = 0; benchIt < BENCHMARK_ITERATIONS; ++benchIt) {
    base_t max_findable =
      random_data_generator<base_t>(raw_data, DATA_GENERATION_AMOUNT, column_to_join_a, DATA_ELEMENT_COUNT_A,
                                    column_to_join_b, DATA_ELEMENT_COUNT_B, HASH_BUCKET_COUNT, 0, benchIt);
    std::unordered_map<base_t, base_t> hash_mapuh;
    hash_mapuh.reserve(HASH_BUCKET_COUNT);
    base_t *join_result_id_a = join_allocator(DATA_ELEMENT_COUNT_B);
    base_t *join_result_id_b = join_allocator(DATA_ELEMENT_COUNT_B);
    const auto t_start = std::chrono::high_resolution_clock::now();

    // build phase
    {
      auto it = column_to_join_a.cbegin();
      for (; it != column_to_join_a.cend(); ++it) {
        if (!hash_mapuh.contains(*it)) {
          const size_t curr_pos = it - column_to_join_a.cbegin();
          hash_mapuh[*it] = curr_pos;
        }
      }
    }

    // probe phase
    size_t join_result = 0;
    {
      auto it = column_to_join_b.cbegin();
      for (; it != column_to_join_b.cend(); ++it) {
        if (hash_mapuh.contains(*it)) {
          const size_t curr_pos = it - column_to_join_b.cbegin();
          join_result_id_a[join_result] = hash_mapuh[*it];
          join_result_id_b[join_result] = curr_pos;
          ++join_result;
        }
      }
    }

    const auto t_end = std::chrono::high_resolution_clock::now();

    bool all_found = true;
    bool wrong_result = false;

    result_checker<base_t>(all_found, wrong_result, column_to_join_a, DATA_ELEMENT_COUNT_A, column_to_join_b,
                           DATA_ELEMENT_COUNT_B, join_result_id_a, join_result_id_b, join_result, max_findable);

    REQUIRE(all_found);
    REQUIRE(!wrong_result);

    const auto bench_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
    if (tuddbs::bench_timings.contains(1)) {
      tuddbs::bench_timings[1] += bench_us;
    } else {
      tuddbs::bench_timings[1] = bench_us;
    }

    join_deleter(join_result_id_a);
    join_deleter(join_result_id_b);
  }

  join_deleter(raw_data);
  tuddbs::print_timings(BENCHMARK_ITERATIONS);
  tuddbs::bench_timings.clear();
}

// #### Sequential execution
TEST_CASE("Hash Join for uint64_t with avx2 with key may contain zero",
          "[cpu][hash_join][uint64_t][zero_key][avx2-seq]") {
  std::cout << "[avx2-seq][zero_key] uint64_t Sequential" << std::endl;
  using base_t = uint64_t;
  using namespace tuddbs;
  using hash_join_t = Hash_Join<tsl::simd<uint64_t, tsl::avx2>, size_t,
                                OperatorHintSet<hints::hashing::linear_displacement, hints::hashing::size_exp_2,
                                                hints::hash_join::keys_may_contain_empty_indicator,
                                                hints::hash_join::global_first_occurence_required>>;
  using hash_join_state_t = hash_join_column_set_t<base_t>;
  // data allocator
  auto join_allocator = [](size_t i) -> base_t * {
    return reinterpret_cast<base_t *>(_mm_malloc(i * sizeof(base_t), 64));
  };
  auto join_deleter = [](base_t *ptr) { _mm_free(ptr); };

  // Columns

  base_t *raw_data = join_allocator(DATA_GENERATION_AMOUNT);
  tuddbs::InMemoryColumn<base_t> column_to_join_a(DATA_ELEMENT_COUNT_A, join_allocator, join_deleter);
  tuddbs::InMemoryColumn<base_t> column_to_join_b(DATA_ELEMENT_COUNT_B, join_allocator, join_deleter);

  // ResultArrays.
  for (size_t benchIt = 0; benchIt < BENCHMARK_ITERATIONS; ++benchIt) {
    base_t max_findable =
      random_data_generator<base_t>(raw_data, DATA_GENERATION_AMOUNT, column_to_join_a, DATA_ELEMENT_COUNT_A,
                                    column_to_join_b, DATA_ELEMENT_COUNT_B, HASH_BUCKET_COUNT, 0, benchIt);
    base_t *join_result_id_a = join_allocator(DATA_ELEMENT_COUNT_B);
    base_t *join_result_id_b = join_allocator(DATA_ELEMENT_COUNT_B);
    hash_join_state_t hash_join_columns(HASH_BUCKET_COUNT, join_allocator, join_deleter);

    hash_join_t::builder_t builder(hash_join_columns.key_sink.begin(), hash_join_columns.used_sink.begin(),
                                   hash_join_columns.value_sink.begin(), HASH_BUCKET_COUNT);
    hash_join_t::prober_t prober(hash_join_columns.key_sink.begin(), hash_join_columns.used_sink.begin(),
                                 hash_join_columns.value_sink.begin(), HASH_BUCKET_COUNT);
    const auto t_start = std::chrono::high_resolution_clock::now();
    // build phase
    builder(column_to_join_a.cbegin(), column_to_join_a.cend());

    // probe phase
    size_t join_result = 0;
    join_result = prober(join_result_id_a, join_result_id_b, column_to_join_b.cbegin(), column_to_join_b.cend());

    const auto t_end = std::chrono::high_resolution_clock::now();

    bool all_found = true;
    bool wrong_result = false;

    result_checker<base_t>(all_found, wrong_result, column_to_join_a, DATA_ELEMENT_COUNT_A, column_to_join_b,
                           DATA_ELEMENT_COUNT_B, join_result_id_a, join_result_id_b, join_result, max_findable);

    REQUIRE(all_found);
    REQUIRE(!wrong_result);

    const auto bench_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
    if (tuddbs::bench_timings.contains(1)) {
      tuddbs::bench_timings[1] += bench_us;
    } else {
      tuddbs::bench_timings[1] = bench_us;
    }

    join_deleter(join_result_id_a);
    join_deleter(join_result_id_b);
  }

  join_deleter(raw_data);
  tuddbs::print_timings(BENCHMARK_ITERATIONS);
  tuddbs::bench_timings.clear();
}

TEST_CASE("Hash Join for uint64_t with avx2 without key may contain zero", "[cpu][hash_join][uint64_t][avx2-seq]") {
  std::cout << "[avx2-seq] uint64_t Sequential" << std::endl;
  using base_t = uint64_t;
  using namespace tuddbs;
  using hash_join_t = Hash_Join<tsl::simd<uint64_t, tsl::avx2>, size_t,
                                OperatorHintSet<hints::hashing::linear_displacement, hints::hashing::size_exp_2,
                                                hints::hash_join::global_first_occurence_required>>;
  using hash_join_state_t = hash_join_column_set_t<base_t>;
  // data allocator
  auto join_allocator = [](size_t i) -> base_t * {
    return reinterpret_cast<base_t *>(_mm_malloc(i * sizeof(base_t), 64));
  };
  auto join_deleter = [](base_t *ptr) { _mm_free(ptr); };

  // Columns

  base_t *raw_data = join_allocator(DATA_GENERATION_AMOUNT);
  tuddbs::InMemoryColumn<base_t> column_to_join_a(DATA_ELEMENT_COUNT_A, join_allocator, join_deleter);
  tuddbs::InMemoryColumn<base_t> column_to_join_b(DATA_ELEMENT_COUNT_B, join_allocator, join_deleter);

  // ResultArrays.
  for (size_t benchIt = 0; benchIt < BENCHMARK_ITERATIONS; ++benchIt) {
    base_t max_findable =
      random_data_generator<base_t>(raw_data, DATA_GENERATION_AMOUNT, column_to_join_a, DATA_ELEMENT_COUNT_A,
                                    column_to_join_b, DATA_ELEMENT_COUNT_B, HASH_BUCKET_COUNT, 1, benchIt);
    base_t *join_result_id_a = join_allocator(DATA_ELEMENT_COUNT_B);
    base_t *join_result_id_b = join_allocator(DATA_ELEMENT_COUNT_B);
    hash_join_state_t hash_join_columns(HASH_BUCKET_COUNT, join_allocator, join_deleter);

    hash_join_t::builder_t builder(hash_join_columns.key_sink.begin(), hash_join_columns.used_sink.begin(),
                                   hash_join_columns.value_sink.begin(), HASH_BUCKET_COUNT);
    hash_join_t::prober_t prober(hash_join_columns.key_sink.begin(), hash_join_columns.used_sink.begin(),
                                 hash_join_columns.value_sink.begin(), HASH_BUCKET_COUNT);
    const auto t_start = std::chrono::high_resolution_clock::now();
    // build phase
    builder(column_to_join_a.cbegin(), column_to_join_a.cend());

    // probe phase
    size_t join_result = 0;
    join_result = prober(join_result_id_a, join_result_id_b, column_to_join_b.cbegin(), column_to_join_b.cend());

    const auto t_end = std::chrono::high_resolution_clock::now();

    bool all_found = true;
    bool wrong_result = false;

    result_checker<base_t>(all_found, wrong_result, column_to_join_a, DATA_ELEMENT_COUNT_A, column_to_join_b,
                           DATA_ELEMENT_COUNT_B, join_result_id_a, join_result_id_b, join_result, max_findable);

    REQUIRE(all_found);
    REQUIRE(!wrong_result);

    const auto bench_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
    if (tuddbs::bench_timings.contains(1)) {
      tuddbs::bench_timings[1] += bench_us;
    } else {
      tuddbs::bench_timings[1] = bench_us;
    }

    join_deleter(join_result_id_a);
    join_deleter(join_result_id_b);
  }

  join_deleter(raw_data);
  tuddbs::print_timings(BENCHMARK_ITERATIONS);
  tuddbs::bench_timings.clear();
}

// #### Sequential Merge
TEST_CASE("Hash Join for uint64_t with avx2 simple Merge", "[cpu][hash_join][simple_merge][uint64_t][avx2]") {
  std::cout << "[avx2] uint64_t with simple Merge (Sequential)" << std::endl;
  using base_t = uint64_t;
  using namespace tuddbs;
  using hash_join_t = Hash_Join<tsl::simd<uint64_t, tsl::avx2>, size_t,
                                OperatorHintSet<hints::hashing::linear_displacement, hints::hashing::size_exp_2,
                                                hints::hash_join::global_first_occurence_required>>;
  using hash_join_state_t = hash_join_column_set_t<base_t>;
  // data allocator
  auto join_allocator = [](size_t i) -> base_t * {
    return reinterpret_cast<base_t *>(_mm_malloc(i * sizeof(base_t), 64));
  };
  auto join_deleter = [](base_t *ptr) { _mm_free(ptr); };

  // Columns

  base_t *raw_data = join_allocator(DATA_GENERATION_AMOUNT);
  tuddbs::InMemoryColumn<base_t> column_to_join_a(DATA_ELEMENT_COUNT_A, join_allocator, join_deleter);
  tuddbs::InMemoryColumn<base_t> column_to_join_b(DATA_ELEMENT_COUNT_B, join_allocator, join_deleter);

  // ResultArrays.
  for (size_t benchIt = 0; benchIt < BENCHMARK_ITERATIONS; ++benchIt) {
    base_t max_findable =
      random_data_generator<base_t>(raw_data, DATA_GENERATION_AMOUNT, column_to_join_a, DATA_ELEMENT_COUNT_A,
                                    column_to_join_b, DATA_ELEMENT_COUNT_B, HASH_BUCKET_COUNT, 1, benchIt);
    base_t *join_result_id_a = join_allocator(DATA_ELEMENT_COUNT_B);
    base_t *join_result_id_b = join_allocator(DATA_ELEMENT_COUNT_B);
    hash_join_state_t hash_join_columns_1(HASH_BUCKET_COUNT, join_allocator, join_deleter);
    hash_join_state_t hash_join_columns_2(HASH_BUCKET_COUNT, join_allocator, join_deleter);

    hash_join_t::builder_t builder1(hash_join_columns_1.key_sink.begin(), hash_join_columns_1.used_sink.begin(),
                                    hash_join_columns_1.value_sink.begin(), HASH_BUCKET_COUNT);
    hash_join_t::prober_t prober1(hash_join_columns_1.key_sink.begin(), hash_join_columns_1.used_sink.begin(),
                                  hash_join_columns_1.value_sink.begin(), HASH_BUCKET_COUNT);

    hash_join_t::builder_t builder2(hash_join_columns_2.key_sink.begin(), hash_join_columns_2.used_sink.begin(),
                                    hash_join_columns_2.value_sink.begin(), HASH_BUCKET_COUNT);

    const auto t_start = std::chrono::high_resolution_clock::now();

    // build phase
    builder1(column_to_join_a.cbegin(), column_to_join_a.cbegin() + DATA_ELEMENT_COUNT_A / 2);
    builder2(column_to_join_a.cbegin() + DATA_ELEMENT_COUNT_A / 2, column_to_join_a.cend(), DATA_ELEMENT_COUNT_A / 2);

    // simple merge
    builder1.merge(builder2);

    // probe phase
    size_t join_result = 0;
    join_result = prober1(join_result_id_a, join_result_id_b, column_to_join_b.cbegin(), column_to_join_b.cend());

    const auto t_end = std::chrono::high_resolution_clock::now();

    bool all_found = true;
    bool wrong_result = false;

    result_checker<base_t>(all_found, wrong_result, column_to_join_a, DATA_ELEMENT_COUNT_A, column_to_join_b,
                           DATA_ELEMENT_COUNT_B, join_result_id_a, join_result_id_b, join_result, max_findable);

    REQUIRE(all_found);
    REQUIRE(!wrong_result);

    const auto bench_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
    if (tuddbs::bench_timings.contains(1)) {
      tuddbs::bench_timings[1] += bench_us;
    } else {
      tuddbs::bench_timings[1] = bench_us;
    }

    join_deleter(join_result_id_a);
    join_deleter(join_result_id_b);
  }

  join_deleter(raw_data);
  tuddbs::print_timings(BENCHMARK_ITERATIONS);
  tuddbs::bench_timings.clear();
}
