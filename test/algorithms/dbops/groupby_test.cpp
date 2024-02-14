#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>

#include "algorithms/dbops/group.hpp"
#include "algorithms/dbops/hashing.hpp"
#include "algorithms/dbops/simdops.hpp"
#include "datastructures/column.hpp"

TEST_CASE("GroupBy for uint64_t with avx2", "[cpu][groupby][uint64_t][avx2]") {
  using namespace tuddbs;
  using group_t = Group<tsl::simd<uint64_t, tsl::avx2>,
                        OperatorHintSet<hints::hashing::linear_displacement, hints::hashing::size_exp_2,
                                        hints::grouping::global_first_occurence_required>>;

  const size_t element_count = 1 << 30;
  const size_t map_count = 1 << 31;

  InMemoryColumn<uint64_t> column_to_group(
    element_count, [](size_t i) { return reinterpret_cast<uint64_t*>(_mm_malloc(i * sizeof(uint64_t), 64)); },
    [](uint64_t* ptr) { _mm_free(ptr); });

  InMemoryColumn<uint64_t> gids(
    element_count, [](size_t i) { return reinterpret_cast<uint64_t*>(_mm_malloc(i * sizeof(uint64_t), 64)); },
    [](uint64_t* ptr) { _mm_free(ptr); });

  InMemoryColumn<uint64_t> map_key_sink(
    map_count, [](size_t i) { return reinterpret_cast<uint64_t*>(_mm_malloc(i * sizeof(uint64_t), 64)); },
    [](uint64_t* ptr) { _mm_free(ptr); });

  InMemoryColumn<uint64_t> map_gid_sink(
    map_count, [](size_t i) { return reinterpret_cast<uint64_t*>(_mm_malloc(i * sizeof(uint64_t), 64)); },
    [](uint64_t* ptr) { _mm_free(ptr); });

  InMemoryColumn<uint64_t> gext_sink(
    map_count, [](size_t i) { return reinterpret_cast<uint64_t*>(_mm_malloc(i * sizeof(uint64_t), 64)); },
    [](uint64_t* ptr) { _mm_free(ptr); });

  group_t::builder_t builder(map_key_sink.begin(), map_gid_sink.begin(), gext_sink.begin(), map_count);

  builder(column_to_group.cbegin(), column_to_group.cend());

  group_t::grouper_t grouper(map_key_sink.begin(), map_gid_sink.begin(), gext_sink.begin(), map_count);

  grouper(gids.begin(), column_to_group.cbegin(), column_to_group.cend());

  for (size_t i = 0; i < element_count; ++i) {
    REQUIRE(column_to_group[gext_sink[gids[i]]] == column_to_group[i]);
  }
}