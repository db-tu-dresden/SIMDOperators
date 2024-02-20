#include "algorithms/dbops/groupby/groupby.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include "algorithms/dbops/dbops_hints.hpp"
#include "algorithms/utils/hashing.hpp"
#include "datastructures/column.hpp"

#define DATA_ELEMENT_COUNT (1 << 22)
#define GLOBAL_GROUP_COUNT (1 << 15)
#define HASH_BUCKET_COUNT (2 * GLOBAL_GROUP_COUNT)
#define MAX_PARALLELISM_DEGREE 32
#define BENCHMARK_ITERATIONS 3

template <typename Range>
struct GroupByOrigRecreator : Catch::Matchers::MatcherGenericBase {
  Range const &m_orig_column;
  GroupByOrigRecreator(Range const &orig_column) : m_orig_column(orig_column) {}

  template <typename OtherRange>
  bool match(OtherRange const &other) const {
    auto orig_column_it = m_orig_column.cbegin();
    auto gids_it = other.gids.cbegin();
    auto gexts_it = other.gext_sink.cbegin();
    for (size_t i = 0; i < m_orig_column.count(); ++i) {
      auto gid = gids_it[i];
      auto gext = gexts_it[gid];
      if (orig_column_it[i] != orig_column_it[gext]) {
        std::cerr << "Mismatch at index " << i << " with orig " << orig_column_it[i] << " and recreated "
                  << orig_column_it[gext] << " (GID: " << gid << ", GEXT: " << gext << ")" << std::endl;
        return false;
      }
    }
    return true;
  }

  std::string describe() const override { return "Original column could NOT be recreated from groupby result"; }
};

template <typename Range>
auto GroupByRecreate(const Range &range) -> GroupByOrigRecreator<Range> {
  return GroupByOrigRecreator<Range>{range};
}

template <typename T>
struct group_column_set_t {
  tuddbs::InMemoryColumn<T> gids;
  tuddbs::InMemoryColumn<T> map_key_sink;
  tuddbs::InMemoryColumn<T> map_gid_sink;
  tuddbs::InMemoryColumn<T> gext_sink;

  explicit group_column_set_t(const size_t map_count, auto allocator, auto deleter)
    : gids(),
      map_key_sink(map_count, allocator, deleter),
      map_gid_sink(map_count, allocator, deleter),
      gext_sink(map_count, allocator, deleter) {}

  ~group_column_set_t() {}

  void allocate_gid_column(size_t element_count, auto allocator, auto deleter) {
    gids = tuddbs::InMemoryColumn<T>(element_count, allocator, deleter);
  }
};

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

// #### Sequential Merge
TEST_CASE("GroupBy for uint64_t with sse", "[cpu][groupby][uint64_t][sse]") {
  std::cout << "[sse] uint64_t with Cascading Merge" << std::endl;
  using base_t = uint64_t;
  using namespace tuddbs;
  using group_t =
    Group<tsl::simd<uint64_t, tsl::sse>,
          OperatorHintSet<hints::hashing::linear_displacement, hints::hashing::size_exp_2,
                          hints::hashing::keys_may_contain_zero, hints::grouping::global_first_occurence_required>>;
  using group_state_t = group_column_set_t<base_t>;

  auto group_allocator = [](size_t i) -> base_t * {
    return reinterpret_cast<base_t *>(_mm_malloc(i * sizeof(base_t), 64));
  };
  auto group_deleter = [](base_t *ptr) { _mm_free(ptr); };

  auto parallel_build_groups = [](InMemoryColumn<base_t> *data_col, group_t::builder_t *builder, size_t start_offset,
                                  size_t elements, size_t tid) -> void {
    auto data_start = data_col->cbegin(start_offset);
    auto data_end = data_start + elements;
    (*builder)(data_start, data_end, start_offset);
  };

  InMemoryColumn<base_t> column_to_group(DATA_ELEMENT_COUNT, group_allocator, group_deleter);
  // auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  size_t seed = tuddbs::get_bench_seed();
  std::mt19937 mt(seed);
  // std::cerr << "Seed: " << seed << std::endl;
  std::uniform_int_distribution<> dist(0, GLOBAL_GROUP_COUNT);

  for (auto it = column_to_group.begin(); it != column_to_group.end(); ++it) {
    (*it) = dist(mt);
  }

  // We currently require element_count to be divisible by parallelism_degree. Best case its a power of 2.
  for (size_t parallelism_degree = 1; parallelism_degree <= MAX_PARALLELISM_DEGREE; parallelism_degree *= 2) {
    for (size_t benchIt = 0; benchIt < BENCHMARK_ITERATIONS; ++benchIt) {
      const auto t_start = std::chrono::high_resolution_clock::now();
      // std::cout << "=== Running with " << parallelism_degree << " Thread(s) ===" << std::endl;
      const size_t elements_per_thread = DATA_ELEMENT_COUNT / parallelism_degree;
      const size_t map_count_per_thread = 2 * elements_per_thread;

      std::vector<std::thread> pool;
      std::vector<group_t::builder_t *> builders;
      std::vector<group_state_t *> builder_states;

      group_state_t group_columns(HASH_BUCKET_COUNT, group_allocator, group_deleter);
      group_t::builder_t builder(group_columns.map_key_sink.begin(), group_columns.map_gid_sink.begin(),
                                 group_columns.gext_sink.begin(), HASH_BUCKET_COUNT);

      // First grouper has a large-enough state to hold all groups later on.
      pool.emplace_back(parallel_build_groups, &column_to_group, &builder, 0, elements_per_thread, 0);

      for (size_t i = 1; i < parallelism_degree; ++i) {
        builder_states.push_back(new group_state_t(map_count_per_thread, group_allocator, group_deleter));
        const auto state = builder_states.back();

        builders.push_back(new group_t::builder_t(state->map_key_sink.begin(), state->map_gid_sink.begin(),
                                                  state->gext_sink.begin(), map_count_per_thread));

        pool.emplace_back(parallel_build_groups, &column_to_group, builders.back(), i * elements_per_thread,
                          elements_per_thread, i);
      }

      for (auto &t : pool) {
        t.join();
      }
      pool.clear();

      for (auto parallel_builder : builders) {
        builder.merge(*parallel_builder);
      }

      group_t::grouper_t grouper(group_columns.map_key_sink.begin(), group_columns.map_gid_sink.begin(),
                                 group_columns.gext_sink.begin(), HASH_BUCKET_COUNT);

      group_columns.allocate_gid_column(DATA_ELEMENT_COUNT, group_allocator, group_deleter);
      grouper(group_columns.gids.begin(), column_to_group.cbegin(), column_to_group.cend());

      REQUIRE_THAT(group_columns, GroupByRecreate(column_to_group));

      for (auto state : builder_states) {
        delete state;
      }
      const auto t_end = std::chrono::high_resolution_clock::now();
      const auto bench_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
      if (tuddbs::bench_timings.contains(parallelism_degree)) {
        tuddbs::bench_timings[parallelism_degree] += bench_us;
      } else {
        tuddbs::bench_timings[parallelism_degree] = bench_us;
      }
    }
  }
  tuddbs::print_timings(BENCHMARK_ITERATIONS);
  tuddbs::bench_timings.clear();
}

TEST_CASE("GroupBy for uint64_t with avx", "[cpu][groupby][uint64_t][avx2]") {
  std::cout << "[avx2] uint64_t with Cascading Merge" << std::endl;

  using base_t = uint64_t;
  using namespace tuddbs;
  using group_t =
    Group<tsl::simd<uint64_t, tsl::avx2>,
          OperatorHintSet<hints::hashing::linear_displacement, hints::hashing::size_exp_2,
                          hints::hashing::keys_may_contain_zero, hints::grouping::global_first_occurence_required>>;
  using group_state_t = group_column_set_t<base_t>;

  auto group_allocator = [](size_t i) -> base_t * {
    return reinterpret_cast<base_t *>(_mm_malloc(i * sizeof(base_t), 64));
  };
  auto group_deleter = [](base_t *ptr) { _mm_free(ptr); };

  auto parallel_build_groups = [](InMemoryColumn<base_t> *data_col, group_t::builder_t *builder, size_t start_offset,
                                  size_t elements, size_t tid) -> void {
    auto data_start = data_col->cbegin(start_offset);
    auto data_end = data_start + elements;
    (*builder)(data_start, data_end, start_offset);
  };

  InMemoryColumn<base_t> column_to_group(DATA_ELEMENT_COUNT, group_allocator, group_deleter);
  size_t seed = tuddbs::get_bench_seed();
  std::mt19937 mt(seed);
  // std::cerr << "Seed: " << seed << std::endl;
  std::uniform_int_distribution<> dist(0, GLOBAL_GROUP_COUNT);

  for (auto it = column_to_group.begin(); it != column_to_group.end(); ++it) {
    (*it) = dist(mt);
  }

  // We currently require element_count to be divisible by parallelism_degree. Best case its a power of 2.
  for (size_t parallelism_degree = 1; parallelism_degree <= MAX_PARALLELISM_DEGREE; parallelism_degree *= 2) {
    for (size_t benchIt = 0; benchIt < BENCHMARK_ITERATIONS; ++benchIt) {
      const auto t_start = std::chrono::high_resolution_clock::now();
      // std::cout << "=== Running with " << parallelism_degree << " Thread(s) ===" << std::endl;
      const size_t elements_per_thread = DATA_ELEMENT_COUNT / parallelism_degree;
      const size_t map_count_per_thread = 2 * elements_per_thread;

      std::vector<std::thread> pool;
      std::vector<group_t::builder_t *> builders;
      std::vector<group_state_t *> builder_states;

      group_state_t group_columns(HASH_BUCKET_COUNT, group_allocator, group_deleter);
      group_t::builder_t builder(group_columns.map_key_sink.begin(), group_columns.map_gid_sink.begin(),
                                 group_columns.gext_sink.begin(), HASH_BUCKET_COUNT);

      // First grouper has a large-enough state to hold all groups later on.
      pool.emplace_back(parallel_build_groups, &column_to_group, &builder, 0, elements_per_thread, 0);

      for (size_t i = 1; i < parallelism_degree; ++i) {
        builder_states.push_back(new group_state_t(map_count_per_thread, group_allocator, group_deleter));
        const auto state = builder_states.back();

        builders.push_back(new group_t::builder_t(state->map_key_sink.begin(), state->map_gid_sink.begin(),
                                                  state->gext_sink.begin(), map_count_per_thread));

        pool.emplace_back(parallel_build_groups, &column_to_group, builders.back(), i * elements_per_thread,
                          elements_per_thread, i);
      }

      for (auto &t : pool) {
        t.join();
      }
      pool.clear();

      for (auto parallel_builder : builders) {
        builder.merge(*parallel_builder);
      }

      group_t::grouper_t grouper(group_columns.map_key_sink.begin(), group_columns.map_gid_sink.begin(),
                                 group_columns.gext_sink.begin(), HASH_BUCKET_COUNT);

      group_columns.allocate_gid_column(DATA_ELEMENT_COUNT, group_allocator, group_deleter);
      grouper(group_columns.gids.begin(), column_to_group.cbegin(), column_to_group.cend());

      REQUIRE_THAT(group_columns, GroupByRecreate(column_to_group));

      for (auto state : builder_states) {
        delete state;
      }
      const auto t_end = std::chrono::high_resolution_clock::now();
      const auto bench_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
      if (tuddbs::bench_timings.contains(parallelism_degree)) {
        tuddbs::bench_timings[parallelism_degree] += bench_us;
      } else {
        tuddbs::bench_timings[parallelism_degree] = bench_us;
      }
    }
  }
  tuddbs::print_timings(BENCHMARK_ITERATIONS);
  tuddbs::bench_timings.clear();
}

// #### Merge Tree - Sequential
TEST_CASE("GroupBy for uint64_t with sse / Merge Tree - Sequential", "[cpu][groupby-tree-seq][uint64_t][sse]") {
  std::cout << "[sse] uint64_t with Tree-Merge (Sequential)" << std::endl;
  using base_t = uint64_t;
  using namespace tuddbs;
  using group_t =
    Group<tsl::simd<uint64_t, tsl::sse>,
          OperatorHintSet<hints::hashing::linear_displacement, hints::hashing::size_exp_2,
                          hints::grouping::global_first_occurence_required, hints::hashing::keys_may_contain_zero>>;
  using group_state_t = group_column_set_t<base_t>;

  auto group_allocator = [](size_t i) -> base_t * {
    return reinterpret_cast<base_t *>(_mm_malloc(i * sizeof(base_t), 64));
  };
  auto group_deleter = [](base_t *ptr) { _mm_free(ptr); };

  auto parallel_build_groups = [](InMemoryColumn<base_t> *data_col, group_t::builder_t *builder, size_t start_offset,
                                  size_t elements, size_t tid) -> void {
    auto data_start = data_col->cbegin(start_offset);
    auto data_end = data_start + elements;
    (*builder)(data_start, data_end, start_offset);
  };

  InMemoryColumn<base_t> column_to_group(DATA_ELEMENT_COUNT, group_allocator, group_deleter);
  // auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  size_t seed = tuddbs::get_bench_seed();
  std::mt19937 mt(seed);
  // std::cerr << "Seed: " << seed << std::endl;
  std::uniform_int_distribution<> dist(0, GLOBAL_GROUP_COUNT);

  for (auto it = column_to_group.begin(); it != column_to_group.end(); ++it) {
    (*it) = dist(mt);
  }

  // We currently require element_count to be divisible by parallelism_degree. Best case its a power of 2.
  for (size_t parallelism_degree = 1; parallelism_degree <= MAX_PARALLELISM_DEGREE; parallelism_degree *= 2) {
    for (size_t benchIt = 0; benchIt < BENCHMARK_ITERATIONS; ++benchIt) {
      const auto t_start = std::chrono::high_resolution_clock::now();
      // std::cout << "=== Running with " << parallelism_degree << " Thread(s) ===" << std::endl;
      const size_t elements_per_thread = DATA_ELEMENT_COUNT / parallelism_degree;
      const size_t map_count_per_thread = 2 * elements_per_thread;

      std::vector<std::thread> pool;
      std::vector<group_t::builder_t *> builders;
      std::vector<group_state_t *> builder_states;

      for (size_t i = 0; i < parallelism_degree; ++i) {
        builder_states.push_back(new group_state_t(map_count_per_thread, group_allocator, group_deleter));
        const auto state = builder_states.back();

        builders.push_back(new group_t::builder_t(state->map_key_sink.begin(), state->map_gid_sink.begin(),
                                                  state->gext_sink.begin(), map_count_per_thread));

        pool.emplace_back(parallel_build_groups, &column_to_group, builders.back(), i * elements_per_thread,
                          elements_per_thread, i);
      }

      for (auto &t : pool) {
        t.join();
      }
      pool.clear();

      auto next_pow2 = [](uint64_t x) -> uint64_t { return x == 1 ? 1 : 1 << (64 - __builtin_clzl(x - 1)); };

      group_state_t *final_merge_state = nullptr;
      size_t final_map_count = 0;

      // Tree merge
      if (parallelism_degree > 1) {
        using builder_vec_t = std::vector<group_t::builder_t *>;
        using state_vec_t = std::vector<group_state_t *>;
        size_t stages = parallelism_degree / 2;

        state_vec_t states_empty;
        builder_vec_t builders_empty;

        state_vec_t merge_states_last = builder_states;
        builder_states.clear();
        state_vec_t merge_states_current = states_empty;

        builder_vec_t builders_last = builders;
        builder_vec_t builders_current = builders_empty;

        while (stages > 0) {
          for (size_t i = 0; i < merge_states_last.size(); i += 2) {
            const size_t max_distinct_values =
              next_pow2(builders_last[i]->distinct_key_count() + builders_last[i + 1]->distinct_key_count());

            const size_t current_map_count = next_pow2(2 * max_distinct_values);

            auto merge_state = new group_state_t(current_map_count, group_allocator, group_deleter);
            auto merge_builder =
              new group_t::builder_t(merge_state->map_key_sink.begin(), merge_state->map_gid_sink.begin(),
                                     merge_state->gext_sink.begin(), current_map_count);

            merge_builder->merge(*(builders_last[i]));
            merge_builder->merge(*(builders_last[i + 1]));

            builders_current.push_back(merge_builder);
            merge_states_current.push_back(merge_state);
            final_map_count = std::max(final_map_count, current_map_count);
          }
          for (auto state : merge_states_last) {
            delete state;
          }
          for (auto builder : builders_last) {
            delete builder;
          }
          merge_states_last.clear();
          builders_last.clear();

          builders_current.swap(builders_last);
          merge_states_current.swap(merge_states_last);

          // Advance stage counter
          stages /= 2;
        }
        final_merge_state = merge_states_last[0];
        builder_states.push_back(final_merge_state);
      } else {
        final_merge_state = builder_states[0];
        final_map_count = map_count_per_thread;
      }

      group_t::grouper_t grouper(final_merge_state->map_key_sink.begin(), final_merge_state->map_gid_sink.begin(),
                                 final_merge_state->gext_sink.begin(), final_map_count);

      final_merge_state->allocate_gid_column(DATA_ELEMENT_COUNT, group_allocator, group_deleter);
      grouper(final_merge_state->gids.begin(), column_to_group.cbegin(), column_to_group.cend());

      REQUIRE_THAT(*final_merge_state, GroupByRecreate(column_to_group));

      for (auto state : builder_states) {
        delete state;
      }
      const auto t_end = std::chrono::high_resolution_clock::now();
      const auto bench_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
      if (tuddbs::bench_timings.contains(parallelism_degree)) {
        tuddbs::bench_timings[parallelism_degree] += bench_us;
      } else {
        tuddbs::bench_timings[parallelism_degree] = bench_us;
      }
    }
  }
  tuddbs::print_timings(BENCHMARK_ITERATIONS);
  tuddbs::bench_timings.clear();
}

TEST_CASE("GroupBy for uint64_t with avx2 / Merge Tree - Sequential", "[cpu][groupby-tree-seq][uint64_t][avx2]") {
  std::cout << "[avx2] uint64_t with Tree-Merge (Sequential)" << std::endl;
  using base_t = uint64_t;
  using namespace tuddbs;
  using group_t =
    Group<tsl::simd<uint64_t, tsl::avx2>,
          OperatorHintSet<hints::hashing::linear_displacement, hints::hashing::size_exp_2,
                          hints::grouping::global_first_occurence_required, hints::hashing::keys_may_contain_zero>>;
  using group_state_t = group_column_set_t<base_t>;

  auto group_allocator = [](size_t i) -> base_t * {
    return reinterpret_cast<base_t *>(_mm_malloc(i * sizeof(base_t), 64));
  };
  auto group_deleter = [](base_t *ptr) { _mm_free(ptr); };

  auto parallel_build_groups = [](InMemoryColumn<base_t> *data_col, group_t::builder_t *builder, size_t start_offset,
                                  size_t elements, size_t tid) -> void {
    auto data_start = data_col->cbegin(start_offset);
    auto data_end = data_start + elements;
    (*builder)(data_start, data_end, start_offset);
  };

  InMemoryColumn<base_t> column_to_group(DATA_ELEMENT_COUNT, group_allocator, group_deleter);
  // auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  size_t seed = tuddbs::get_bench_seed();
  std::mt19937 mt(seed);
  // std::cerr << "Seed: " << seed << std::endl;
  std::uniform_int_distribution<> dist(0, GLOBAL_GROUP_COUNT);

  for (auto it = column_to_group.begin(); it != column_to_group.end(); ++it) {
    (*it) = dist(mt);
  }

  // We currently require element_count to be divisible by parallelism_degree. Best case its a power of 2.
  for (size_t parallelism_degree = 1; parallelism_degree <= MAX_PARALLELISM_DEGREE; parallelism_degree *= 2) {
    for (size_t benchIt = 0; benchIt < BENCHMARK_ITERATIONS; ++benchIt) {
      const auto t_start = std::chrono::high_resolution_clock::now();
      // std::cout << "=== Running with " << parallelism_degree << " Thread(s) ===" << std::endl;
      const size_t elements_per_thread = DATA_ELEMENT_COUNT / parallelism_degree;
      const size_t map_count_per_thread = 2 * elements_per_thread;

      std::vector<std::thread> pool;
      std::vector<group_t::builder_t *> builders;
      std::vector<group_state_t *> builder_states;

      for (size_t i = 0; i < parallelism_degree; ++i) {
        builder_states.push_back(new group_state_t(map_count_per_thread, group_allocator, group_deleter));
        const auto state = builder_states.back();

        builders.push_back(new group_t::builder_t(state->map_key_sink.begin(), state->map_gid_sink.begin(),
                                                  state->gext_sink.begin(), map_count_per_thread));

        pool.emplace_back(parallel_build_groups, &column_to_group, builders.back(), i * elements_per_thread,
                          elements_per_thread, i);
      }

      for (auto &t : pool) {
        t.join();
      }
      pool.clear();

      auto next_pow2 = [](uint64_t x) -> uint64_t { return x == 1 ? 1 : 1 << (64 - __builtin_clzl(x - 1)); };

      group_state_t *final_merge_state = nullptr;
      size_t final_map_count = 0;

      // Tree merge
      if (parallelism_degree > 1) {
        using builder_vec_t = std::vector<group_t::builder_t *>;
        using state_vec_t = std::vector<group_state_t *>;
        size_t stages = parallelism_degree / 2;

        state_vec_t states_empty;
        builder_vec_t builders_empty;

        state_vec_t merge_states_last = builder_states;
        builder_states.clear();
        state_vec_t merge_states_current = states_empty;

        builder_vec_t builders_last = builders;
        builder_vec_t builders_current = builders_empty;

        while (stages > 0) {
          for (size_t i = 0; i < merge_states_last.size(); i += 2) {
            const size_t max_distinct_values =
              next_pow2(builders_last[i]->distinct_key_count() + builders_last[i + 1]->distinct_key_count());

            const size_t current_map_count = next_pow2(2 * max_distinct_values);

            auto merge_state = new group_state_t(current_map_count, group_allocator, group_deleter);
            auto merge_builder =
              new group_t::builder_t(merge_state->map_key_sink.begin(), merge_state->map_gid_sink.begin(),
                                     merge_state->gext_sink.begin(), current_map_count);

            merge_builder->merge(*(builders_last[i]));
            merge_builder->merge(*(builders_last[i + 1]));

            builders_current.push_back(merge_builder);
            merge_states_current.push_back(merge_state);
            final_map_count = std::max(final_map_count, current_map_count);
          }
          for (auto state : merge_states_last) {
            delete state;
          }
          for (auto builder : builders_last) {
            delete builder;
          }
          merge_states_last.clear();
          builders_last.clear();

          builders_current.swap(builders_last);
          merge_states_current.swap(merge_states_last);

          // Advance stage counter
          stages /= 2;
        }
        final_merge_state = merge_states_last[0];
        builder_states.push_back(final_merge_state);
      } else {
        final_merge_state = builder_states[0];
        final_map_count = map_count_per_thread;
      }

      group_t::grouper_t grouper(final_merge_state->map_key_sink.begin(), final_merge_state->map_gid_sink.begin(),
                                 final_merge_state->gext_sink.begin(), final_map_count);

      final_merge_state->allocate_gid_column(DATA_ELEMENT_COUNT, group_allocator, group_deleter);
      grouper(final_merge_state->gids.begin(), column_to_group.cbegin(), column_to_group.cend());

      REQUIRE_THAT(*final_merge_state, GroupByRecreate(column_to_group));

      for (auto state : builder_states) {
        delete state;
      }
      const auto t_end = std::chrono::high_resolution_clock::now();
      const auto bench_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
      if (tuddbs::bench_timings.contains(parallelism_degree)) {
        tuddbs::bench_timings[parallelism_degree] += bench_us;
      } else {
        tuddbs::bench_timings[parallelism_degree] = bench_us;
      }
    }
  }
  tuddbs::print_timings(BENCHMARK_ITERATIONS);
  tuddbs::bench_timings.clear();
}

// Merge Tree - Parallelized
TEST_CASE("GroupBy for uint64_t with sse / Merge Tree - Parallel", "[cpu][groupby-tree-par][uint64_t][sse]") {
  std::cout << "[sse] uint64_t with Tree-Merge (Parallel)" << std::endl;
  using base_t = uint64_t;
  using namespace tuddbs;
  using group_t =
    Group<tsl::simd<uint64_t, tsl::sse>,
          OperatorHintSet<hints::hashing::linear_displacement, hints::hashing::size_exp_2,
                          hints::grouping::global_first_occurence_required, hints::hashing::keys_may_contain_zero>>;
  using group_state_t = group_column_set_t<base_t>;

  auto group_allocator = [](size_t i) -> base_t * {
    return reinterpret_cast<base_t *>(_mm_malloc(i * sizeof(base_t), 64));
  };
  auto group_deleter = [](base_t *ptr) { _mm_free(ptr); };

  auto parallel_build_groups = [](InMemoryColumn<base_t> *data_col, group_t::builder_t *builder, size_t start_offset,
                                  size_t elements, size_t tid) -> void {
    auto data_start = data_col->cbegin(start_offset);
    auto data_end = data_start + elements;
    (*builder)(data_start, data_end, start_offset);
  };

  InMemoryColumn<base_t> column_to_group(DATA_ELEMENT_COUNT, group_allocator, group_deleter);
  // auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  size_t seed = tuddbs::get_bench_seed();
  std::mt19937 mt(seed);
  // std::cerr << "Seed: " << seed << std::endl;
  std::uniform_int_distribution<> dist(0, GLOBAL_GROUP_COUNT);

  for (auto it = column_to_group.begin(); it != column_to_group.end(); ++it) {
    (*it) = dist(mt);
  }

  // We currently require element_count to be divisible by parallelism_degree. Best case its a power of 2.
  for (size_t parallelism_degree = 1; parallelism_degree <= MAX_PARALLELISM_DEGREE; parallelism_degree *= 2) {
    for (size_t benchIt = 0; benchIt < BENCHMARK_ITERATIONS; ++benchIt) {
      const auto t_start = std::chrono::high_resolution_clock::now();
      // std::cout << "=== Running with " << parallelism_degree << " Thread(s) ===" << std::endl;
      const size_t elements_per_thread = DATA_ELEMENT_COUNT / parallelism_degree;
      const size_t map_count_per_thread = 2 * elements_per_thread;

      std::vector<std::thread> pool;
      std::vector<group_t::builder_t *> builders;
      std::vector<group_state_t *> builder_states;

      for (size_t i = 0; i < parallelism_degree; ++i) {
        builder_states.push_back(new group_state_t(map_count_per_thread, group_allocator, group_deleter));
        const auto state = builder_states.back();

        builders.push_back(new group_t::builder_t(state->map_key_sink.begin(), state->map_gid_sink.begin(),
                                                  state->gext_sink.begin(), map_count_per_thread));

        pool.emplace_back(parallel_build_groups, &column_to_group, builders.back(), i * elements_per_thread,
                          elements_per_thread, i);
      }

      for (auto &t : pool) {
        t.join();
      }
      pool.clear();

      auto next_pow2 = [](uint64_t x) -> uint64_t { return x == 1 ? 1 : 1 << (64 - __builtin_clzl(x - 1)); };

      group_state_t *final_merge_state = nullptr;
      size_t final_map_count = 0;

      std::mutex state_mutex;
      std::mutex builder_mutex;
      using builder_vec_t = std::vector<group_t::builder_t *>;
      using state_vec_t = std::vector<group_state_t *>;
      auto do_parallel_merge = [&state_mutex, &builder_mutex, next_pow2, group_allocator, group_deleter](
                                 group_t::builder_t *b1, group_t::builder_t *b2, state_vec_t *merge_states_current,
                                 builder_vec_t *builders_current, size_t *largest_map_count) -> void {
        const size_t max_distinct_values = next_pow2(b1->distinct_key_count() + b2->distinct_key_count());
        const size_t current_map_count = next_pow2(2 * max_distinct_values);
        auto merge_state = new group_state_t(current_map_count, group_allocator, group_deleter);
        auto merge_builder =
          new group_t::builder_t(merge_state->map_key_sink.begin(), merge_state->map_gid_sink.begin(),
                                 merge_state->gext_sink.begin(), current_map_count);

        merge_builder->merge(*b1);
        merge_builder->merge(*b2);

        {
          std::lock_guard lk(state_mutex);
          merge_states_current->push_back(merge_state);
        }
        {
          std::lock_guard lk(builder_mutex);
          builders_current->push_back(merge_builder);
          *largest_map_count = std::max(*largest_map_count, current_map_count);
        }
      };

      // Tree merge
      if (parallelism_degree > 1) {
        size_t stages = parallelism_degree / 2;

        state_vec_t states_empty;
        builder_vec_t builders_empty;

        state_vec_t merge_states_last = builder_states;
        builder_states.clear();
        state_vec_t merge_states_current = states_empty;

        builder_vec_t builders_last = builders;
        builder_vec_t builders_current = builders_empty;

        std::vector<std::thread> merge_pool;
        while (stages > 0) {
          for (size_t i = 0; i < merge_states_last.size(); i += 2) {
            merge_pool.emplace_back(do_parallel_merge, builders_last[i], builders_last[i + 1], &merge_states_current,
                                    &builders_current, &final_map_count);
          }
          for (auto &t : merge_pool) {
            t.join();
          }
          merge_pool.clear();
          for (auto state : merge_states_last) {
            delete state;
          }
          for (auto builder : builders_last) {
            delete builder;
          }
          merge_states_last.clear();
          builders_last.clear();

          builders_current.swap(builders_last);
          merge_states_current.swap(merge_states_last);

          // Advance stage counter
          stages /= 2;
        }
        final_merge_state = merge_states_last[0];
        builder_states.push_back(final_merge_state);
      } else {
        final_merge_state = builder_states[0];
        final_map_count = map_count_per_thread;
      }

      group_t::grouper_t grouper(final_merge_state->map_key_sink.begin(), final_merge_state->map_gid_sink.begin(),
                                 final_merge_state->gext_sink.begin(), final_map_count);

      final_merge_state->allocate_gid_column(DATA_ELEMENT_COUNT, group_allocator, group_deleter);
      grouper(final_merge_state->gids.begin(), column_to_group.cbegin(), column_to_group.cend());

      REQUIRE_THAT(*final_merge_state, GroupByRecreate(column_to_group));

      for (auto state : builder_states) {
        delete state;
      }
      const auto t_end = std::chrono::high_resolution_clock::now();
      const auto bench_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
      if (tuddbs::bench_timings.contains(parallelism_degree)) {
        tuddbs::bench_timings[parallelism_degree] += bench_us;
      } else {
        tuddbs::bench_timings[parallelism_degree] = bench_us;
      }
    }
  }
  tuddbs::print_timings(BENCHMARK_ITERATIONS);
  tuddbs::bench_timings.clear();
}

TEST_CASE("GroupBy for uint64_t with avx2 / Merge Tree - Parallel", "[cpu][groupby-tree-par][uint64_t][avx2]") {
  std::cout << "[avx2] uint64_t with Tree-Merge (Parallel)" << std::endl;
  using base_t = uint64_t;
  using namespace tuddbs;
  using group_t =
    Group<tsl::simd<uint64_t, tsl::sse>,
          OperatorHintSet<hints::hashing::linear_displacement, hints::hashing::size_exp_2,
                          hints::grouping::global_first_occurence_required, hints::hashing::keys_may_contain_zero>>;
  using group_state_t = group_column_set_t<base_t>;

  auto group_allocator = [](size_t i) -> base_t * {
    return reinterpret_cast<base_t *>(_mm_malloc(i * sizeof(base_t), 64));
  };
  auto group_deleter = [](base_t *ptr) { _mm_free(ptr); };

  auto parallel_build_groups = [](InMemoryColumn<base_t> *data_col, group_t::builder_t *builder, size_t start_offset,
                                  size_t elements, size_t tid) -> void {
    auto data_start = data_col->cbegin(start_offset);
    auto data_end = data_start + elements;
    (*builder)(data_start, data_end, start_offset);
  };

  InMemoryColumn<base_t> column_to_group(DATA_ELEMENT_COUNT, group_allocator, group_deleter);
  // auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  size_t seed = tuddbs::get_bench_seed();
  std::mt19937 mt(seed);
  // std::cerr << "Seed: " << seed << std::endl;
  std::uniform_int_distribution<> dist(0, GLOBAL_GROUP_COUNT);

  for (auto it = column_to_group.begin(); it != column_to_group.end(); ++it) {
    (*it) = dist(mt);
  }

  // We currently require element_count to be divisible by parallelism_degree. Best case its a power of 2.
  for (size_t parallelism_degree = 1; parallelism_degree <= MAX_PARALLELISM_DEGREE; parallelism_degree *= 2) {
    for (size_t benchIt = 0; benchIt < BENCHMARK_ITERATIONS; ++benchIt) {
      const auto t_start = std::chrono::high_resolution_clock::now();
      // std::cout << "=== Running with " << parallelism_degree << " Thread(s) ===" << std::endl;
      const size_t elements_per_thread = DATA_ELEMENT_COUNT / parallelism_degree;
      const size_t map_count_per_thread = 2 * elements_per_thread;

      std::vector<std::thread> pool;
      std::vector<group_t::builder_t *> builders;
      std::vector<group_state_t *> builder_states;

      if (parallelism_degree == 1) {
        group_state_t *group_columns = new group_state_t(HASH_BUCKET_COUNT, group_allocator, group_deleter);
        group_t::builder_t *builder =
          new group_t::builder_t(group_columns->map_key_sink.begin(), group_columns->map_gid_sink.begin(),
                                 group_columns->gext_sink.begin(), HASH_BUCKET_COUNT);

        // First grouper has a large-enough state to hold all groups later on.
        builders.push_back(builder);
        builder_states.push_back(group_columns);
        pool.emplace_back(parallel_build_groups, &column_to_group, builder, 0, elements_per_thread, 0);
      } else {
        for (size_t i = 0; i < parallelism_degree; ++i) {
          group_state_t *group_columns = new group_state_t(map_count_per_thread, group_allocator, group_deleter);
          group_t::builder_t *builder =
            new group_t::builder_t(group_columns->map_key_sink.begin(), group_columns->map_gid_sink.begin(),
                                   group_columns->gext_sink.begin(), map_count_per_thread);

          builders.push_back(builder);
          builder_states.push_back(group_columns);
          pool.emplace_back(parallel_build_groups, &column_to_group, builders.back(), i * elements_per_thread,
                            elements_per_thread, i);
        }
      }

      for (auto &t : pool) {
        t.join();
      }
      pool.clear();

      auto next_pow2 = [](uint64_t x) -> uint64_t { return x == 1 ? 1 : 1 << (64 - __builtin_clzl(x - 1)); };

      group_state_t *final_merge_state = nullptr;
      size_t final_map_count = 0;

      using builder_vec_t = std::vector<group_t::builder_t *>;
      using state_vec_t = std::vector<group_state_t *>;
      auto do_parallel_merge = [next_pow2, group_allocator, group_deleter](
                                 group_t::builder_t *b1, group_t::builder_t *b2, state_vec_t *merge_states_current,
                                 builder_vec_t *builders_current, std::vector<size_t> *largest_map_count,
                                 size_t tid) -> void {
        const size_t max_distinct_values = next_pow2(b1->distinct_key_count() + b2->distinct_key_count());
        const size_t current_map_count = next_pow2(2 * max_distinct_values);
        auto merge_state = new group_state_t(current_map_count, group_allocator, group_deleter);
        auto merge_builder =
          new group_t::builder_t(merge_state->map_key_sink.begin(), merge_state->map_gid_sink.begin(),
                                 merge_state->gext_sink.begin(), current_map_count);

        merge_builder->merge(*b1);
        merge_builder->merge(*b2);

        merge_states_current->at(tid) = merge_state;
        builders_current->at(tid) = merge_builder;
        largest_map_count->at(tid) = current_map_count;
      };

      // Tree merge
      if (parallelism_degree > 1) {
        size_t stages = parallelism_degree / 2;

        state_vec_t states_empty;
        builder_vec_t builders_empty;

        state_vec_t merge_states_last = builder_states;
        builder_states.clear();
        state_vec_t merge_states_current = states_empty;

        builder_vec_t builders_last = builders;
        builder_vec_t builders_current = builders_empty;

        std::vector<size_t> thread_max_values;
        thread_max_values.resize(stages);
        std::vector<std::thread> merge_pool;

        while (stages > 0) {
          builders_current.resize(stages);
          merge_states_current.resize(stages);
          size_t tid = 0;
          for (size_t i = 0; i < merge_states_last.size(); i += 2) {
            merge_pool.emplace_back(do_parallel_merge, builders_last[i], builders_last[i + 1], &merge_states_current,
                                    &builders_current, &thread_max_values, tid++);
          }
          for (auto &t : merge_pool) {
            t.join();
          }
          merge_pool.clear();
          for (auto state : merge_states_last) {
            delete state;
          }
          for (auto builder : builders_last) {
            delete builder;
          }
          merge_states_last.clear();
          builders_last.clear();

          builders_current.swap(builders_last);
          merge_states_current.swap(merge_states_last);

          final_map_count =
            std::max(final_map_count, *std::max_element(thread_max_values.begin(), thread_max_values.begin() + stages));
          // Advance stage counter
          stages /= 2;
        }
        final_merge_state = merge_states_last[0];
        builder_states.push_back(final_merge_state);
      } else {
        final_merge_state = builder_states[0];
        final_map_count = map_count_per_thread;
      }

      group_t::grouper_t grouper(final_merge_state->map_key_sink.begin(), final_merge_state->map_gid_sink.begin(),
                                 final_merge_state->gext_sink.begin(), final_map_count);

      final_merge_state->allocate_gid_column(DATA_ELEMENT_COUNT, group_allocator, group_deleter);
      grouper(final_merge_state->gids.begin(), column_to_group.cbegin(), column_to_group.cend());

      REQUIRE_THAT(*final_merge_state, GroupByRecreate(column_to_group));

      for (auto state : builder_states) {
        delete state;
      }
      const auto t_end = std::chrono::high_resolution_clock::now();
      const auto bench_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
      if (tuddbs::bench_timings.contains(parallelism_degree)) {
        tuddbs::bench_timings[parallelism_degree] += bench_us;
      } else {
        tuddbs::bench_timings[parallelism_degree] = bench_us;
      }
    }
  }
  tuddbs::print_timings(BENCHMARK_ITERATIONS);
  tuddbs::bench_timings.clear();
}