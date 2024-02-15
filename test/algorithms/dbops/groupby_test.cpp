#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

#include "algorithms/dbops/group.hpp"
#include "algorithms/dbops/hashing.hpp"
#include "algorithms/dbops/simdops.hpp"
#include "datastructures/column.hpp"

template <typename T>
struct group_column_set_t {
  tuddbs::InMemoryColumn<T> gids;
  tuddbs::InMemoryColumn<T> map_key_sink;
  tuddbs::InMemoryColumn<T> map_gid_sink;
  tuddbs::InMemoryColumn<T> gext_sink;

  explicit group_column_set_t(const size_t element_count, const size_t map_count)
    : gids(element_count), map_key_sink(map_count), map_gid_sink(map_count), gext_sink(map_count) {}

  explicit group_column_set_t(const size_t element_count, const size_t map_count, auto allocator, auto deleter)
    : gids(element_count, allocator, deleter),
      map_key_sink(map_count, allocator, deleter),
      map_gid_sink(map_count, allocator, deleter),
      gext_sink(map_count, allocator, deleter) {}
};

TEST_CASE("GroupBy for uint64_t with sse", "[cpu][groupby][uint64_t][sse]") {
  using base_t = uint64_t;
  using namespace tuddbs;
  using group_t = Group<tsl::simd<uint64_t, tsl::sse>,
                        OperatorHintSet<hints::hashing::linear_displacement, hints::hashing::size_exp_2,
                                        hints::grouping::global_first_occurence_required>>;
  using group_state_t = group_column_set_t<base_t>;

  const size_t element_count = 1UL << 20;
  const size_t map_count = 1UL << 24;

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

  InMemoryColumn<base_t> column_to_group(element_count, group_allocator, group_deleter);
  std::mt19937 mt(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<> dist(1, map_count);

  for (auto it = column_to_group.begin(); it != column_to_group.end(); ++it) {
    (*it) = dist(mt);
  }

  // We currently require element_count to be divisible by parallelism_degree. Best case its a power of 2.
  const size_t max_parallelism_degree = 16;

  for (size_t parallelism_degree = 1; parallelism_degree <= max_parallelism_degree; parallelism_degree *= 2) {
    std::cout << "=== Running with " << parallelism_degree << " Thread(s) ===" << std::endl;
    const size_t elements_per_thread = element_count / parallelism_degree;
    const size_t map_count_per_thread = 2 * elements_per_thread;

    std::vector<std::thread> pool;
    std::vector<group_t::builder_t *> builders;
    std::vector<group_state_t *> builder_states;

    group_state_t group_columns(element_count, map_count, group_allocator, group_deleter);
    group_t::builder_t builder(group_columns.map_key_sink.begin(), group_columns.map_gid_sink.begin(),
                               group_columns.gext_sink.begin(), map_count);

    // First grouper has a large-enough state to hold all groups later on.
    pool.emplace_back(parallel_build_groups, &column_to_group, &builder, 0, elements_per_thread, 0);

    for (size_t i = 1; i < parallelism_degree; ++i) {
      builder_states.push_back(
        new group_state_t(elements_per_thread, map_count_per_thread, group_allocator, group_deleter));
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
                               group_columns.gext_sink.begin(), map_count);

    grouper(group_columns.gids.begin(), column_to_group.cbegin(), column_to_group.cend());

    for (size_t i = 0; i < element_count; ++i) {
      REQUIRE(column_to_group.cbegin()[group_columns.gext_sink.cbegin()[group_columns.gids.cbegin()[i]]] ==
              column_to_group.cbegin()[i]);
    }

    for (auto state : builder_states) {
      delete state;
    }
  }
}

TEST_CASE("GroupBy for uint64_t with avx2", "[cpu][groupby][uint64_t][avx2]") {
  using base_t = uint64_t;
  using namespace tuddbs;
  using group_t = Group<tsl::simd<uint64_t, tsl::avx2>,
                        OperatorHintSet<hints::hashing::linear_displacement, hints::hashing::size_exp_2,
                                        hints::grouping::global_first_occurence_required>>;
  using group_state_t = group_column_set_t<base_t>;

  const size_t element_count = 1UL << 20;
  const size_t map_count = 1UL << 24;

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

  InMemoryColumn<base_t> column_to_group(element_count, group_allocator, group_deleter);
  std::mt19937 mt(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<> dist(1, map_count);

  for (auto it = column_to_group.begin(); it != column_to_group.end(); ++it) {
    (*it) = dist(mt);
  }

  // We currently require element_count to be divisible by parallelism_degree. Best case its a power of 2.
  const size_t max_parallelism_degree = 16;

  for (size_t parallelism_degree = 1; parallelism_degree <= max_parallelism_degree; parallelism_degree *= 2) {
    std::cout << "=== Running with " << parallelism_degree << " Thread(s) ===" << std::endl;
    const size_t elements_per_thread = element_count / parallelism_degree;
    const size_t map_count_per_thread = 2 * elements_per_thread;

    std::vector<std::thread> pool;
    std::vector<group_t::builder_t *> builders;
    std::vector<group_state_t *> builder_states;

    group_state_t group_columns(element_count, map_count, group_allocator, group_deleter);
    group_t::builder_t builder(group_columns.map_key_sink.begin(), group_columns.map_gid_sink.begin(),
                               group_columns.gext_sink.begin(), map_count);

    // First grouper has a large-enough state to hold all groups later on.
    pool.emplace_back(parallel_build_groups, &column_to_group, &builder, 0, elements_per_thread, 0);

    for (size_t i = 1; i < parallelism_degree; ++i) {
      builder_states.push_back(
        new group_state_t(elements_per_thread, map_count_per_thread, group_allocator, group_deleter));
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
                               group_columns.gext_sink.begin(), map_count);

    grouper(group_columns.gids.begin(), column_to_group.cbegin(), column_to_group.cend());

    for (size_t i = 0; i < element_count; ++i) {
      REQUIRE(column_to_group.cbegin()[group_columns.gext_sink.cbegin()[group_columns.gids.cbegin()[i]]] ==
              column_to_group.cbegin()[i]);
    }

    for (auto state : builder_states) {
      delete state;
    }
  }
}
