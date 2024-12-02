
// #include "algorithms/dbops/sort/sort_direct.hpp"
#include <algorithm>
#include <cassert>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

#include "algorithms/dbops/sort/sort.hpp"
#include "algorithms/dbops/sort/sort_by_clusters.hpp"
#include "algorithms/dbops/sort/sort_utils.hpp"

void fill(std::vector<uint64_t>& vec, const size_t seed, const size_t lo, const size_t hi, const size_t count) {
  std::mt19937 mt(seed);
  std::uniform_int_distribution dist(lo, hi);
  vec.resize(count);
  auto gen = [&mt, &dist] { return dist(mt); };
  std::generate(vec.begin(), vec.end(), gen);
}

int main() {
  std::vector<uint64_t> base, data_arr, col2, col3;

  const size_t elements = 128;
  fill(data_arr, 13371337, 1, 5, elements);
  fill(col2, 13371338, 3, 7, elements);
  fill(col3, 13371339, 2, 9, elements);
  base.insert(base.begin(), data_arr.begin(), data_arr.end());

  std::vector<uint64_t> idx;
  for (size_t i = 0; i < data_arr.size(); ++i) {
    idx.push_back(i);
  }

  using SimdStyle = tsl::simd<uint64_t, tsl::avx512>;
  using IndexStyle = tsl::simd<uint64_t, tsl::avx512>;
  using HS_INPL = tuddbs::OperatorHintSet<tuddbs::hints::sort::indirect_inplace>;
  using HS_GATH = tuddbs::OperatorHintSet<tuddbs::hints::sort::indirect_gather>;
  using cluster_sort_t =
    tuddbs::ClusteringSingleColumnSortIndirectInplace<SimdStyle, IndexStyle, tuddbs::TSL_SORT_ORDER::ASC, HS_INPL>;
  cluster_sort_t clusterer(data_arr.data(), idx.data());
  clusterer(0, elements);
  auto clusters = clusterer.getClusters();

  tuddbs::ClusterSortIndirect<SimdStyle, IndexStyle, HS_GATH> mcol_sorter(idx.data(), &clusters);

  mcol_sorter(col2.data(), tuddbs::TSL_SORT_ORDER::ASC);
  mcol_sorter(col3.data(), tuddbs::TSL_SORT_ORDER::DESC);

  auto print_arr = [](auto& vec, auto& idxs) -> void {
    for (auto i : idxs) {
      std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
  };

  print_arr(base, idx);
  print_arr(col2, idx);
  print_arr(col3, idx);

  return 0;
}