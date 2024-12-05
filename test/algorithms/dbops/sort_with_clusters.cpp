
// #include "algorithms/dbops/sort/sort_direct.hpp"
#include <algorithm>
#include <cassert>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "algorithms/dbops/sort/sort.hpp"
#include "algorithms/dbops/sort/sort_by_clusters.hpp"
#include "algorithms/dbops/sort/sort_utils.hpp"
#include "static/utils/type_helper.hpp"

using TupleVec = std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>;

template <typename T>
void fill(std::vector<T>& vec, const size_t seed, const size_t lo, const size_t hi, const size_t count) {
  std::mt19937 mt(seed);
  std::uniform_int_distribution dist(lo, hi);
  vec.resize(count);
  auto gen = [&mt, &dist] { return dist(mt); };
  std::generate(vec.begin(), vec.end(), gen);
}

void print_arr_by_index(auto& vec, auto& idxs) {
  for (auto i : idxs) {
    std::cout << vec[i] << " ";
  }
  std::cout << std::endl;
}

template <class SorterT, class SimdStyle, class IndexStyle, class HintSet>
TupleVec sort(const size_t elements = 64, const size_t seed = 13371337) {
  using T = typename SimdStyle::base_type;
  using IndexType = typename IndexStyle::base_type;
  std::vector<T> base, data_arr, col2, col3;

  fill(data_arr, seed, 1, 5, elements);
  fill(col2, seed + 1, 3, 7, elements);
  fill(col3, seed + 2, 2, 9, elements);
  base.insert(base.begin(), data_arr.begin(), data_arr.end());

  std::vector<IndexType> idx;
  for (size_t i = 0; i < data_arr.size(); ++i) {
    idx.push_back(i);
  }

  SorterT clusterer(data_arr.data(), idx.data());
  clusterer(0, elements);
  auto clusters = clusterer.getClusters();

  tuddbs::ClusterSortIndirect<SimdStyle, IndexStyle, HintSet> mcol_sorter(idx.data(), &clusters);

  mcol_sorter(col2.data(), tuddbs::TSL_SORT_ORDER::ASC);
  mcol_sorter(col3.data(), tuddbs::TSL_SORT_ORDER::DESC);

  // print_arr_by_index(base, idx);
  // print_arr_by_index(col2, idx);
  // print_arr_by_index(col3, idx);

  TupleVec res;
  res.reserve(elements);
  for (size_t i = 0; i < elements; ++i) {
    res.emplace_back(std::make_tuple(base[idx[i]], col2[idx[i]], col3[idx[i]]));
  }
  return res;
}

template <typename T, typename IndexType, tuddbs::TSL_SORT_ORDER order>
void sort_scalar(T* data, IndexType* idx, size_t elementcount) {
  const auto customCompIndirect_LT = [&data](const T& lhs, const T& rhs) -> bool {
    if (order == tuddbs::TSL_SORT_ORDER::ASC) {
      return data[lhs] < data[rhs];
    } else {
      return data[lhs] > data[rhs];
    }
  };

  std::sort(idx, idx + elementcount, customCompIndirect_LT);
}

template <typename T, typename IndexType>
TupleVec sort_with_std(const size_t elements = 64, const size_t seed = 13371337) {
  std::vector<T> base, data_arr, col2, col3;

  fill(data_arr, seed, 1, 5, elements);
  fill(col2, seed + 1, 3, 7, elements);
  fill(col3, seed + 2, 2, 9, elements);
  base.insert(base.begin(), data_arr.begin(), data_arr.end());

  std::vector<IndexType> idx;
  for (size_t i = 0; i < data_arr.size(); ++i) {
    idx.push_back(i);
  }

  auto refine = [elements](std::deque<tuddbs::Cluster>& clusters, T* data, IndexType* idx,
                           tuddbs::TSL_SORT_ORDER order) -> void {
    while (!clusters.empty()) {
      tuddbs::Cluster& c = clusters.front();
      clusters.pop_front();
      if (order == tuddbs::TSL_SORT_ORDER::ASC) {
        sort_scalar<T, IndexType, tuddbs::TSL_SORT_ORDER::ASC>(data, idx + c.start, c.len);
      } else {
        sort_scalar<T, IndexType, tuddbs::TSL_SORT_ORDER::DESC>(data, idx + c.start, c.len);
      }
    }
  };

  std::deque<tuddbs::Cluster> clusters;
  sort_scalar<T, IndexType, tuddbs::TSL_SORT_ORDER::ASC>(data_arr.data(), idx.data(), elements);
  tuddbs::gather_sort::detect_cluster(clusters, data_arr.data(), idx.data(), 0, elements);
  refine(clusters, col2.data(), idx.data(), tuddbs::TSL_SORT_ORDER::ASC);
  tuddbs::gather_sort::detect_cluster(clusters, col2.data(), idx.data(), 0, elements);
  refine(clusters, col3.data(), idx.data(), tuddbs::TSL_SORT_ORDER::DESC);

  // print_arr_by_index(base, idx);
  // print_arr_by_index(col2, idx);
  // print_arr_by_index(col3, idx);

  TupleVec res;
  res.reserve(elements);
  for (size_t i = 0; i < elements; ++i) {
    res.emplace_back(std::make_tuple(base[idx[i]], col2[idx[i]], col3[idx[i]]));
  }
  return res;
}

template <class SimdStyle, class IndexStyle>
void test() {
  using T = typename SimdStyle::base_type;
  using IndexType = typename IndexStyle::base_type;
  using HS_INTAIL =
    tuddbs::OperatorHintSet<tuddbs::hints::sort::indirect_inplace, tuddbs::hints::sort::tail_clustering>;
  using HS_INLEAF =
    tuddbs::OperatorHintSet<tuddbs::hints::sort::indirect_inplace, tuddbs::hints::sort::leaf_clustering>;
  using HS_GATH = tuddbs::OperatorHintSet<tuddbs::hints::sort::indirect_gather>;
  using HS_GATHTAIL =
    tuddbs::OperatorHintSet<tuddbs::hints::sort::indirect_gather, tuddbs::hints::sort::tail_clustering>;
  using HS_GATHLEAF =
    tuddbs::OperatorHintSet<tuddbs::hints::sort::indirect_gather, tuddbs::hints::sort::leaf_clustering>;

  using cluster_proxy_inplace_leaf =
    tuddbs::ClusteringSingleColumnSort<SimdStyle, tuddbs::TSL_SORT_ORDER::ASC, HS_INLEAF, IndexStyle>;
  using cluster_proxy_inplace_tail =
    tuddbs::ClusteringSingleColumnSort<SimdStyle, tuddbs::TSL_SORT_ORDER::ASC, HS_INTAIL, IndexStyle>;

  using cluster_proxy_gather_leaf =
    tuddbs::ClusteringSingleColumnSort<SimdStyle, tuddbs::TSL_SORT_ORDER::ASC, HS_GATHLEAF, IndexStyle>;
  using cluster_proxy_gather_tail =
    tuddbs::ClusteringSingleColumnSort<SimdStyle, tuddbs::TSL_SORT_ORDER::ASC, HS_GATHTAIL, IndexStyle>;

  const size_t elements = std::clamp((size_t)1024, (size_t)std::numeric_limits<IndexType>::min(),
                                     (size_t)std::numeric_limits<IndexType>::max());
  const size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::cout << "Running Multi-Column Sort Test with 3 columns, " << elements << " elements, seed " << seed
            << " [d: " << tsl::type_name<T>() << ", i: " << tsl::type_name<IndexType>() << "]..." << std::endl;
  std::cout << "\t> Running std::sort..." << std::endl;
  const auto std_res = sort_with_std<T, IndexType>(elements, seed);

  std::cout << "\t> Running Inplace, Cluster on Leaf..." << std::endl;
  const auto inplace_leaf_res =
    sort<typename cluster_proxy_inplace_leaf::sorter_t, SimdStyle, IndexStyle, HS_GATH>(elements, seed);
  REQUIRE(std::equal(std_res.begin(), std_res.end(), inplace_leaf_res.begin()));

  std::cout << "\t> Running Inplace, Cluster on Tail..." << std::endl;
  const auto inplace_tail_res =
    sort<typename cluster_proxy_inplace_tail::sorter_t, SimdStyle, IndexStyle, HS_GATH>(elements, seed);
  REQUIRE(std::equal(std_res.begin(), std_res.end(), inplace_tail_res.begin()));

  std::cout << "\t> Running Gather, Cluster on Leaf..." << std::endl;
  const auto gather_leaf_res =
    sort<typename cluster_proxy_gather_leaf::sorter_t, SimdStyle, IndexStyle, HS_GATH>(elements, seed);
  REQUIRE(std::equal(std_res.begin(), std_res.end(), gather_leaf_res.begin()));

  std::cout << "\t> Running Gather, Cluster on Tail..." << std::endl;
  const auto gather_tail_res =
    sort<typename cluster_proxy_gather_tail::sorter_t, SimdStyle, IndexStyle, HS_GATH>(elements, seed);
  REQUIRE(std::equal(std_res.begin(), std_res.end(), gather_tail_res.begin()));
}

#ifdef TSL_CONTAINS_AVX2
TEST_CASE("Cluster Sort 3 Columns", "[3col][avx2][d:uint64_t][i:uint64_t]") {
  test<tsl::simd<uint64_t, tsl::avx2>, tsl::simd<uint64_t, tsl::avx2>>();
}
TEST_CASE("Cluster Sort 3 Columns", "[3col][avx2][d:uint32_t][i:uint64_t]") {
  test<tsl::simd<uint32_t, tsl::avx2>, tsl::simd<uint64_t, tsl::avx2>>();
}
TEST_CASE("Cluster Sort 3 Columns", "[3col][avx2][d:uint32_t][i:uint32_t]") {
  test<tsl::simd<uint32_t, tsl::avx2>, tsl::simd<uint32_t, tsl::avx2>>();
}
TEST_CASE("Cluster Sort 3 Columns", "[3col][avx2][d:uint16_t][i:uint64_t]") {
  test<tsl::simd<uint16_t, tsl::avx2>, tsl::simd<uint64_t, tsl::avx2>>();
}
TEST_CASE("Cluster Sort 3 Columns", "[3col][avx2][d:uint16_t][i:uint32_t]") {
  test<tsl::simd<uint16_t, tsl::avx2>, tsl::simd<uint32_t, tsl::avx2>>();
}
TEST_CASE("Cluster Sort 3 Columns", "[3col][avx2][d:uint16_t][i:uint16_t]") {
  test<tsl::simd<uint16_t, tsl::avx2>, tsl::simd<uint16_t, tsl::avx2>>();
}

TEST_CASE("Cluster Sort 3 Columns", "[3col][avx2][d:uint8_t][i:uint64_t]") {
  test<tsl::simd<uint8_t, tsl::avx2>, tsl::simd<uint64_t, tsl::avx2>>();
}
TEST_CASE("Cluster Sort 3 Columns", "[3col][avx2][d:uint8_t][i:uint32_t]") {
  test<tsl::simd<uint8_t, tsl::avx2>, tsl::simd<uint32_t, tsl::avx2>>();
}
TEST_CASE("Cluster Sort 3 Columns", "[3col][avx2][d:uint8_t][i:uint16_t]") {
  test<tsl::simd<uint8_t, tsl::avx2>, tsl::simd<uint16_t, tsl::avx2>>();
}
TEST_CASE("Cluster Sort 3 Columns", "[3col][avx2][d:uint8_t][i:uint8_t]") {
  test<tsl::simd<uint8_t, tsl::avx2>, tsl::simd<uint8_t, tsl::avx2>>();
}

#endif