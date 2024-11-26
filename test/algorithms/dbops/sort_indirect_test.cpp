
#include "algorithms/dbops/sort/sort_indirect.hpp"

#include <algorithm>
#include <cassert>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

#include "algorithms/dbops/sort/sort_utils.hpp"
#include "algorithms/utils/hinting.hpp"

const static size_t global_max_seeds = 2;
const static size_t global_max_repetitions = 3;

template <class D>
using uniform_distribution = typename std::conditional_t<
  std::is_floating_point_v<D>, std::uniform_real_distribution<D>,
  typename std::conditional_t<std::is_integral_v<D>, std::uniform_int_distribution<D>, void>>;

template <typename T>
void fill(T* data, const size_t count, size_t seed = 0) {
  if (seed == 0) {
    seed = 13371337;
  }
  std::mt19937_64 mt(seed);
  const size_t numeric_max = static_cast<size_t>(std::numeric_limits<T>::max());
  ssize_t min_bound = 0;
  if constexpr (std::is_signed_v<T>) {
    min_bound = -50;
  }
  const size_t dst_bound = 1024;
  const T max_val = static_cast<T>(std::clamp(dst_bound, static_cast<size_t>(50), numeric_max));
  uniform_distribution<T> dist(min_bound, max_val);

  for (size_t i = 0; i < count; ++i) {
    data[i] = dist(mt);
  }
}

// template <class SimdStyle, typename T = typename SimdStyle::base_type>
// ssize_t run_tsl(T* base, T* data, T* reference, const size_t array_element_count, const size_t array_size_B,
//                 const size_t seed, const size_t runs) {
//   size_t ret = 0;
//   for (size_t rep = 0; rep < runs; ++rep) {
//     // Reset the state to the original unsorted data
//     memcpy(data, base, array_size_B);

//     tuddbs::Sort<SimdStyle, tuddbs::TSL_SORT_ORDER::ASC> sorter(data);
//     sorter(0, array_element_count);

//     // Compare our permutation to the std::sort reference and print debug output, if it does not match.
//     ret |= memcmp(data, reference, array_size_B);
//     REQUIRE(ret == 0);
//   }
//   // Return the sanity information. 0 means its sane
//   return static_cast<ssize_t>(ret);
// }

// template <typename T>
// void run_std(T* base, T* data, const size_t array_element_count, const size_t array_size_B, const size_t runs) {
//   const std::string tab("\t");
//   double bwdh_total = 0.0;
//   for (size_t rep = 0; rep < runs; ++rep) {
//     // Reset the state to the original unsorted data
//     memcpy(data, base, array_size_B);

//     // Do the sort
//     std::sort(data, data + array_element_count);
//   }
// }

// template <class SimdStyle, typename T = SimdStyle::base_type>
// void dispatch_type(const size_t max_seeds, const size_t max_repetitions) {
//   std::cout << "Running " << tsl::type_name<SimdStyle>() << "..." << std::endl;
//   // Increase the tested data size from 512 Byte to 128 MiB (last measured boundary)
//   std::vector<size_t> sizes{256, 16 * 1024, 16 * 1024 * 1024, 128ul * 1024ul * 1024ul};
//   for (auto size_B : sizes) {
//     std::cout << "\t" << size_B << " Byte" << std::endl;
//     const size_t array_size_B = size_B;
//     const size_t array_element_count = array_size_B / sizeof(T);

//     // TSL Executor for common allocation structure
//     using cpu_executor = tsl::executor<tsl::runtime::cpu>;
//     cpu_executor exec;

//     // Allocate the array for the to-be-sorted data (base), std::sort baseline (arr_1) and the vectorized execution
//     auto base = exec.allocate<T>(array_element_count, 64);
//     auto arr_1 = exec.allocate<T>(array_element_count, 64);
//     auto arr_2 = exec.allocate<T>(array_element_count, 64);

//     ssize_t sanity = 0;

//     // Benchmark loop
//     std::cout << "\tSeed ";
//     for (size_t run = 0; run < max_seeds; ++run) {
//       std::cout << run + 1 << "..." << std::flush;
//       // Draw the current timestamp as seed
//       const auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

//       // Populte the to-be-sortred data once
//       fill(base, array_element_count, seed);

//       // Perform the sorting using std::sort
//       run_std(base, arr_1, array_element_count, array_size_B, max_repetitions);

//       // Template-based dispatching for the TSL processing styles
//       sanity |= run_tsl<SimdStyle>(base, arr_2, arr_1, array_element_count, array_size_B, seed, max_repetitions);

//       // Check if the sorted output equals the permutation of std::sort, print some fancy debug stuff if not
//       REQUIRE(sanity == 0);
//     }
//     std::cout << std::endl;
//     // Proper cleanup
//     exec.deallocate(arr_2);
//     exec.deallocate(arr_1);
//     exec.deallocate(base);
//   }
// }

// #ifdef TSL_CONTAINS_SSE
// TEST_CASE("Direct Sort, SSE, signed 8 bit", "[sort_direct][sse][int8]") {
//   dispatch_type<tsl::simd<int8_t, tsl::sse>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, SSE, signed 16 bit", "[sort_direct][sse][int16]") {
//   dispatch_type<tsl::simd<int16_t, tsl::sse>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, SSE, signed 32 bit", "[sort_direct][sse][int32]") {
//   dispatch_type<tsl::simd<int32_t, tsl::sse>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, SSE, signed 64 bit", "[sort_direct][sse][int64]") {
//   dispatch_type<tsl::simd<int64_t, tsl::sse>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, SSE, unsigned 8 bit", "[sort_direct][sse][uint8]") {
//   dispatch_type<tsl::simd<uint8_t, tsl::sse>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, SSE, unsigned 16 bit", "[sort_direct][sse][uint16]") {
//   dispatch_type<tsl::simd<uint16_t, tsl::sse>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, SSE, unsigned 32 bit", "[sort_direct][sse][uint32]") {
//   dispatch_type<tsl::simd<uint32_t, tsl::sse>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, SSE, unsigned 64 bit", "[sort_direct][sse][uint64]") {
//   dispatch_type<tsl::simd<uint64_t, tsl::sse>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, SSE, float", "[sort_direct][sse][float]") {
//   dispatch_type<tsl::simd<uint32_t, tsl::sse>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, SSE, double", "[sort_direct][sse][double]") {
//   dispatch_type<tsl::simd<uint64_t, tsl::sse>>(global_max_seeds, global_max_repetitions);
// }
// #endif
// #ifdef TSL_CONTAINS_AVX2
// TEST_CASE("Direct Sort, avx2, signed 8 bit", "[sort_direct][avx2][int8]") {
//   dispatch_type<tsl::simd<int8_t, tsl::avx2>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, avx2, signed 16 bit", "[sort_direct][avx2][int16]") {
//   dispatch_type<tsl::simd<int16_t, tsl::avx2>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, avx2, signed 32 bit", "[sort_direct][avx2][int32]") {
//   dispatch_type<tsl::simd<int32_t, tsl::avx2>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, avx2, signed 64 bit", "[sort_direct][avx2][int64]") {
//   dispatch_type<tsl::simd<int64_t, tsl::avx2>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, avx2, unsigned 8 bit", "[sort_direct][avx2][uint8]") {
//   dispatch_type<tsl::simd<uint8_t, tsl::avx2>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, avx2, unsigned 16 bit", "[sort_direct][avx2][uint16]") {
//   dispatch_type<tsl::simd<uint16_t, tsl::avx2>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, avx2, unsigned 32 bit", "[sort_direct][avx2][uint32]") {
//   dispatch_type<tsl::simd<uint32_t, tsl::avx2>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, avx2, unsigned 64 bit", "[sort_direct][avx2][uint64]") {
//   dispatch_type<tsl::simd<uint64_t, tsl::avx2>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, avx2, float", "[sort_direct][avx2][float]") {
//   dispatch_type<tsl::simd<uint32_t, tsl::avx2>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, avx2, double", "[sort_direct][avx2][double]") {
//   dispatch_type<tsl::simd<uint64_t, tsl::avx2>>(global_max_seeds, global_max_repetitions);
// }
// #endif
// #ifdef TSL_CONTAINS_AVX512
// TEST_CASE("Direct Sort, AVX512, signed 8 bit", "[sort_direct][AVX512][int8]") {
//   dispatch_type<tsl::simd<int8_t, tsl::avx512>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, AVX512, signed 16 bit", "[sort_direct][AVX512][int16]") {
//   dispatch_type<tsl::simd<int16_t, tsl::avx512>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, AVX512, signed 32 bit", "[sort_direct][AVX512][int32]") {
//   dispatch_type<tsl::simd<int32_t, tsl::avx512>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, AVX512, signed 64 bit", "[sort_direct][AVX512][int64]") {
//   dispatch_type<tsl::simd<int64_t, tsl::avx512>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, AVX512, unsigned 8 bit", "[sort_direct][AVX512][uint8]") {
//   dispatch_type<tsl::simd<uint8_t, tsl::avx512>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, AVX512, unsigned 16 bit", "[sort_direct][AVX512][uint16]") {
//   dispatch_type<tsl::simd<uint16_t, tsl::avx512>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, AVX512, unsigned 32 bit", "[sort_direct][AVX512][uint32]") {
//   dispatch_type<tsl::simd<uint32_t, tsl::avx512>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, AVX512, unsigned 64 bit", "[sort_direct][AVX512][uint64]") {
//   dispatch_type<tsl::simd<uint64_t, tsl::avx512>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, AVX512, float", "[sort_direct][AVX512][float]") {
//   dispatch_type<tsl::simd<uint32_t, tsl::avx512>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, AVX512, double", "[sort_direct][AVX512][double]") {
//   dispatch_type<tsl::simd<uint64_t, tsl::avx512>>(global_max_seeds, global_max_repetitions);
// }
// #endif
// #ifdef TSL_CONTAINS_NEON
// TEST_CASE("Direct Sort, neon, signed 8 bit", "[sort_direct][neon][int8]") {
//   dispatch_type<tsl::simd<int8_t, tsl::neon>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, neon, signed 16 bit", "[sort_direct][neon][int16]") {
//   dispatch_type<tsl::simd<int16_t, tsl::neon>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, neon, signed 32 bit", "[sort_direct][neon][int32]") {
//   dispatch_type<tsl::simd<int32_t, tsl::neon>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, neon, signed 64 bit", "[sort_direct][neon][int64]") {
//   dispatch_type<tsl::simd<int64_t, tsl::neon>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, neon, unsigned 8 bit", "[sort_direct][neon][uint8]") {
//   dispatch_type<tsl::simd<uint8_t, tsl::neon>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, neon, unsigned 16 bit", "[sort_direct][neon][uint16]") {
//   dispatch_type<tsl::simd<uint16_t, tsl::neon>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, neon, unsigned 32 bit", "[sort_direct][neon][uint32]") {
//   dispatch_type<tsl::simd<uint32_t, tsl::neon>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, neon, unsigned 64 bit", "[sort_direct][neon][uint64]") {
//   dispatch_type<tsl::simd<uint64_t, tsl::neon>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, neon, float", "[sort_direct][neon][float]") {
//   dispatch_type<tsl::simd<uint32_t, tsl::neon>>(global_max_seeds, global_max_repetitions);
// }
// TEST_CASE("Direct Sort, neon, double", "[sort_direct][neon][double]") {
//   dispatch_type<tsl::simd<uint64_t, tsl::neon>>(global_max_seeds, global_max_repetitions);
// }
// #endif

TEST_CASE("Indirect Inplace", "[sort_indirect][avx512][inplace]") {
  using SimdStyle = tsl::simd<uint64_t, tsl::avx512>;
  using IndexStyle = tsl::simd<uint64_t, tsl::avx512>;
  using InplaceIndirectSorter = tuddbs::SortIndirect<SimdStyle, IndexStyle, tuddbs::TSL_SORT_ORDER::ASC,
                                                     tuddbs::OperatorHintSet<tuddbs::hints::sort_indirect::inplace>>;
  using GatherIndirectSorter = tuddbs::SortIndirect<SimdStyle, IndexStyle, tuddbs::TSL_SORT_ORDER::ASC,
                                                    tuddbs::OperatorHintSet<tuddbs::hints::sort_indirect::gather>>;

  auto data1 = static_cast<size_t*>(malloc(1024));
  auto data2 = static_cast<size_t*>(malloc(1024));

  GatherIndirectSorter gatherer(data1, data2);
  InplaceIndirectSorter inplacer(data1, data2);

  gatherer(0, 0);
  inplacer(0, 0);

  REQUIRE(true);

  free(data2);
  free(data1);
}