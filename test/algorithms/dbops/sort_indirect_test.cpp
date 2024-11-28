
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
#include "algorithms/dbops/sort/sort_utils.hpp"
#include "algorithms/utils/hinting.hpp"

const static size_t global_max_seeds = 4;
const static size_t global_max_repetitions = 1;

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

template <class SimdStyle, class IndexStyle, class HintSet, typename DataT = typename SimdStyle::base_type,
          typename IndexT = IndexStyle::base_type>
ssize_t run_tsl(DataT* base, DataT* data, DataT* reference, IndexT* base_index, IndexT* index, IndexT* reference_idx,
                const size_t array_element_count, const size_t data_array_size_B, const size_t idx_array_size_B,
                const size_t seed, const size_t runs) {
  const std::string tab("\t");
  size_t ret = 0;
  for (size_t rep = 0; rep < runs; ++rep) {
    /**
     * Reset the state to the original unsorted data
     * This step is measured, because we DO create a copy
     * of the base data and sort it. This simulates the actual
     * copy mechanism for the data.
     */
    memcpy(data, base, data_array_size_B);
    memcpy(index, base_index, idx_array_size_B);

    using sort_proxy = tuddbs::SingleColumnSort<SimdStyle, tuddbs::TSL_SORT_ORDER::ASC, HintSet, IndexStyle>;
    typename sort_proxy::sorter_t indirect_sort(data, index);
    indirect_sort(0, array_element_count);

    // Compare our permutation to the std::sort reference and print debug output, if it does not match.
    bool all_equal = true;
    for (size_t pos = 0; pos < array_element_count; ++pos) {
      all_equal &= (base[index[pos]] == base[reference_idx[pos]]);
    }
    // Accumulate the sanity over multiple runs
    ret |= static_cast<size_t>(!all_equal);
    REQUIRE(all_equal);
  }
  // Return the sanity information. 0 means its sane
  return static_cast<ssize_t>(ret);
}

template <typename DataT, typename IndexT>
void run_std(DataT* base, DataT* data, IndexT* base_index, IndexT* idx, const size_t array_element_count,
             const size_t data_array_size_B, const size_t idx_array_size_B, const size_t seed, const size_t runs) {
  // Custom comparator for std::sort to use the indirection for sorting an index array
  const auto customCompIndirect_LT = [&data](const IndexT& lhs, const IndexT& rhs) -> bool {
    return data[lhs] < data[rhs];
  };

  for (size_t rep = 0; rep < runs; ++rep) {
    /**
     * Reset the state to the original unsorted data
     * This step is measured, because we DO create a copy
     * of the base data and sort it. This simulates the actual
     * copy mechanism for the data. This creates a sorted
     * variant of the base data as well as a sorted index array.
     */
    memcpy(data, base, data_array_size_B);
    memcpy(idx, base_index, idx_array_size_B);
    // Sort the index array through indirection
    std::sort(idx, idx + array_element_count, customCompIndirect_LT);
    // Sort the data array direct
    std::sort(data, data + array_element_count);
  }
}

template <class SimdStyle, class IndexStyle, class HintSet, typename DataT = typename SimdStyle::base_type,
          typename IndexT = typename IndexStyle::base_type>
void dispatch_type(const size_t max_seeds, const size_t max_repetitions) {
  const size_t max_data_size = 512 * 1024 * 1024;

  // Make sure the index type can hold the targeted max data size. If not, just use its maximum
  const size_t data_size_boundary = std::min(std::numeric_limits<IndexT>::max() * sizeof(IndexT), max_data_size);
  std::cout << "Running ProcStyle " << tsl::type_name<typename SimdStyle::target_extension>() << " | IdxStyle "
            << tsl::type_name<typename IndexStyle::target_extension>() << " with (" << tsl::type_name<DataT>() << "-"
            << tsl::type_name<IndexT>() << "). Upperbound fixed to " << data_size_boundary << " for "
            << tsl::type_name<IndexT>() << std::endl;

  // Increase the tested data size from 256 Byte to 256 MiB or the maximum value representable by index type
  // std::vector<size_t> sizes{256, 16 * 1024, 2 * 1024 * 1024};
  std::vector<size_t> sizes{256, 2 * 1024, 4 * 1024, 8 * 1024, 16 * 1024};
  for (auto it = sizes.begin(); it != sizes.end(); ++it) {
    if (*it > data_size_boundary) {
      sizes.erase(it, sizes.end());
      break;
    }
  }
  for (auto size_B : sizes) {
    std::cout << "Data Size: " << size_B << "\r" << std::flush;
    const size_t data_array_size_B = size_B;
    const size_t array_element_count = data_array_size_B / sizeof(DataT);
    const size_t idx_array_size_B = array_element_count * sizeof(IndexT);

    // TSL Executor for common allocation structure
    using cpu_executor = tsl::executor<tsl::runtime::cpu>;
    cpu_executor exec;

    // Allocate the array for the to-be-sorted data (base), std::sort baseline (arr_1) and the vectorized execution
    // (arr_2)
    auto base = exec.allocate<DataT>(array_element_count, 64);
    auto arr_1 = exec.allocate<DataT>(array_element_count, 64);
    auto arr_2 = exec.allocate<DataT>(array_element_count, 64);

    // Allocate the INDEX array for the to-be-sorted data (base), std::sort baseline (arr) and the vectorized execution
    // (arr_2)
    auto base_idx = exec.allocate<IndexT>(array_element_count, 64);
    auto arr_idx = exec.allocate<IndexT>(array_element_count, 64);
    auto arr_idx_2 = exec.allocate<IndexT>(array_element_count, 64);

    ssize_t sanity = 0;
    for (size_t run = 0; run < max_seeds; ++run) {
      // Draw the current timestamp as seed
      const auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

      // Populte the to-be-sortred data once
      fill(base, array_element_count, seed);

      // Create the position list for the original data
      for (size_t i = 0; i < array_element_count; ++i) {
        base_idx[i] = i;
      }

      // Perform the sorting using std::sort
      run_std(base, arr_1, base_idx, arr_idx, array_element_count, data_array_size_B, idx_array_size_B, seed,
              max_repetitions);

      /**
       * Template-based dispatching for the TSL processing styles
       * We actually do not need the processing style for the index variant,
       * however this makes some typedefs easier later.
       */
      sanity |=
        run_tsl<SimdStyle, IndexStyle, HintSet>(base, arr_2, arr_1, base_idx, arr_idx_2, arr_idx, array_element_count,
                                                data_array_size_B, idx_array_size_B, seed, max_repetitions);
    }
    // // Check if the sorted output equals the permutation of std::sort, print some fancy debug stuff if not
    if (sanity != 0) {
      for (size_t i = 0; i < array_element_count; ++i) {
        if (base[arr_idx[i]] != base[arr_idx_2[i]]) {
          std::cout << "IdxMismatch at Index " << i << ": " << +arr_1[i] << "-" << +arr_2[i] << std::endl;
        }
      }
      std::cout << std::endl;
    }
    REQUIRE(sanity == 0);

    // Proper cleanup
    free(arr_idx_2);
    free(arr_idx);
    free(base_idx);

    free(arr_2);
    free(arr_1);
    free(base);
  }
}

using HS_INPL = tuddbs::OperatorHintSet<tuddbs::hints::sort::indirect_inplace>;
using HS_GATH = tuddbs::OperatorHintSet<tuddbs::hints::sort::indirect_gather>;

// Tests for Indirect Inplace
#ifdef TSL_CONTAINS_SSE
TEST_CASE("Indirect Sort, Inplace. UI8-UI8, SSE", "[ui8-ui8][sse][inplace]") {
  dispatch_type<tsl::simd<uint8_t, tsl::sse>, tsl::simd<uint8_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                     global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI8-UI16, SSE", "[ui8-ui16][sse][inplace]") {
  dispatch_type<tsl::simd<uint8_t, tsl::sse>, tsl::simd<uint16_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                      global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI8-UI32, SSE", "[ui8-ui32][sse][inplace]") {
  dispatch_type<tsl::simd<uint8_t, tsl::sse>, tsl::simd<uint32_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                      global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI8-UI64, SSE", "[ui8-ui64][sse][inplace]") {
  dispatch_type<tsl::simd<uint8_t, tsl::sse>, tsl::simd<uint64_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                      global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. UI16-UI16, SSE", "[ui16-ui16][sse][inplace]") {
  dispatch_type<tsl::simd<uint16_t, tsl::sse>, tsl::simd<uint16_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                       global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI16-UI32, SSE", "[ui16-ui32][sse][inplace]") {
  dispatch_type<tsl::simd<uint16_t, tsl::sse>, tsl::simd<uint32_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                       global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI16-UI64, SSE", "[ui16-ui64][sse][inplace]") {
  dispatch_type<tsl::simd<uint16_t, tsl::sse>, tsl::simd<uint64_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                       global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. UI32-UI32, SSE", "[ui32-ui32][sse][inplace]") {
  dispatch_type<tsl::simd<uint32_t, tsl::sse>, tsl::simd<uint32_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                       global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI32-UI64, SSE", "[ui32-ui64][sse][inplace]") {
  dispatch_type<tsl::simd<uint32_t, tsl::sse>, tsl::simd<uint64_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                       global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. UI64-UI64, SSE", "[ui64-ui64][sse][inplace]") {
  dispatch_type<tsl::simd<uint64_t, tsl::sse>, tsl::simd<uint64_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                       global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. UI8-i8, SSE", "[i8-i8][sse][inplace]") {
  dispatch_type<tsl::simd<int8_t, tsl::sse>, tsl::simd<int8_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                   global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i8-i16, SSE", "[i8-i16][sse][inplace]") {
  dispatch_type<tsl::simd<int8_t, tsl::sse>, tsl::simd<int16_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                    global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i8-i32, SSE", "[i8-i32][sse][inplace]") {
  dispatch_type<tsl::simd<int8_t, tsl::sse>, tsl::simd<int32_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                    global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i8-i64, SSE", "[i8-i64][sse][inplace]") {
  dispatch_type<tsl::simd<int8_t, tsl::sse>, tsl::simd<int64_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                    global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. i16-i16, SSE", "[i16-i16][sse][inplace]") {
  dispatch_type<tsl::simd<int16_t, tsl::sse>, tsl::simd<int16_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                     global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i16-i32, SSE", "[i16-i32][sse][inplace]") {
  dispatch_type<tsl::simd<int16_t, tsl::sse>, tsl::simd<int32_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                     global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i16-i64, SSE", "[i16-i64][sse][inplace]") {
  dispatch_type<tsl::simd<int16_t, tsl::sse>, tsl::simd<int64_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                     global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. i32-i32, SSE", "[i32-i32][sse][inplace]") {
  dispatch_type<tsl::simd<int32_t, tsl::sse>, tsl::simd<int32_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                     global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i32-i64, SSE", "[i32-i64][sse][inplace]") {
  dispatch_type<tsl::simd<int32_t, tsl::sse>, tsl::simd<int64_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                     global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. i64-i64, SSE", "[i64-i64][sse][inplace]") {
  dispatch_type<tsl::simd<int64_t, tsl::sse>, tsl::simd<int64_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                     global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. f32-ui32, SSE", "[f32-ui32][sse][inplace]") {
  dispatch_type<tsl::simd<float, tsl::sse>, tsl::simd<uint32_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                    global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. f32-ui32, SSE", "[f32-ui64][sse][inplace]") {
  dispatch_type<tsl::simd<float, tsl::sse>, tsl::simd<uint64_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                    global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. f64-ui64, SSE", "[f64-ui64][sse][inplace]") {
  dispatch_type<tsl::simd<float, tsl::sse>, tsl::simd<uint64_t, tsl::sse>, HS_INPL>(global_max_seeds,
                                                                                    global_max_repetitions);
}
#endif

#ifdef TSL_CONTAINS_AVX2
TEST_CASE("Indirect Sort, Inplace. UI8-UI8, AVX2", "[ui8-ui8][avx2][inplace]") {
  dispatch_type<tsl::simd<uint8_t, tsl::avx2>, tsl::simd<uint8_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                       global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI8-UI16, AVX2", "[ui8-ui16][avx2][inplace]") {
  dispatch_type<tsl::simd<uint8_t, tsl::avx2>, tsl::simd<uint16_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                        global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI8-UI32, AVX2", "[ui8-ui32][avx2][inplace]") {
  dispatch_type<tsl::simd<uint8_t, tsl::avx2>, tsl::simd<uint32_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                        global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI8-UI64, AVX2", "[ui8-ui64][avx2][inplace]") {
  dispatch_type<tsl::simd<uint8_t, tsl::avx2>, tsl::simd<uint64_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                        global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. UI16-UI16, AVX2", "[ui16-ui16][avx2][inplace]") {
  dispatch_type<tsl::simd<uint16_t, tsl::avx2>, tsl::simd<uint16_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                         global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI16-UI32, AVX2", "[ui16-ui32][avx2][inplace]") {
  dispatch_type<tsl::simd<uint16_t, tsl::avx2>, tsl::simd<uint32_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                         global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI16-UI64, AVX2", "[ui16-ui64][avx2][inplace]") {
  dispatch_type<tsl::simd<uint16_t, tsl::avx2>, tsl::simd<uint64_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                         global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. UI32-UI32, AVX2", "[ui32-ui32][avx2][inplace]") {
  dispatch_type<tsl::simd<uint32_t, tsl::avx2>, tsl::simd<uint32_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                         global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI32-UI64, AVX2", "[ui32-ui64][avx2][inplace]") {
  dispatch_type<tsl::simd<uint32_t, tsl::avx2>, tsl::simd<uint64_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                         global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. UI64-UI64, AVX2", "[ui64-ui64][avx2][inplace]") {
  dispatch_type<tsl::simd<uint64_t, tsl::avx2>, tsl::simd<uint64_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                         global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. i8-i8, AVX2", "[i8-i8][avx2][inplace]") {
  dispatch_type<tsl::simd<int8_t, tsl::avx2>, tsl::simd<int8_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                     global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i8-i16, AVX2", "[i8-i16][avx2][inplace]") {
  dispatch_type<tsl::simd<int8_t, tsl::avx2>, tsl::simd<int16_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                      global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i8-i32, AVX2", "[i8-i32][avx2][inplace]") {
  dispatch_type<tsl::simd<int8_t, tsl::avx2>, tsl::simd<int32_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                      global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i8-i64, AVX2", "[i8-i64][avx2][inplace]") {
  dispatch_type<tsl::simd<int8_t, tsl::avx2>, tsl::simd<int64_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                      global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. i16-i16, AVX2", "[i16-i16][avx2][inplace]") {
  dispatch_type<tsl::simd<int16_t, tsl::avx2>, tsl::simd<int16_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                       global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i16-i32, AVX2", "[i16-i32][avx2][inplace]") {
  dispatch_type<tsl::simd<int16_t, tsl::avx2>, tsl::simd<int32_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                       global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i16-i64, AVX2", "[i16-i64][avx2][inplace]") {
  dispatch_type<tsl::simd<int16_t, tsl::avx2>, tsl::simd<int64_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                       global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. i32-i32, AVX2", "[i32-i32][avx2][inplace]") {
  dispatch_type<tsl::simd<int32_t, tsl::avx2>, tsl::simd<int32_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                       global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i32-i64, AVX2", "[i32-i64][avx2][inplace]") {
  dispatch_type<tsl::simd<int32_t, tsl::avx2>, tsl::simd<int64_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                       global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. i64-i64, AVX2", "[i64-i64][avx2][inplace]") {
  dispatch_type<tsl::simd<int64_t, tsl::avx2>, tsl::simd<int64_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                       global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. f32-ui32, AVX2", "[f32-ui32][avx2][inplace]") {
  dispatch_type<tsl::simd<float, tsl::avx2>, tsl::simd<uint32_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                      global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. f32-ui32, AVX2", "[f32-ui64][avx2][inplace]") {
  dispatch_type<tsl::simd<float, tsl::avx2>, tsl::simd<uint64_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                      global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. f64-ui64, AVX2", "[f64-ui64][avx2][inplace]") {
  dispatch_type<tsl::simd<float, tsl::avx2>, tsl::simd<uint64_t, tsl::avx2>, HS_INPL>(global_max_seeds,
                                                                                      global_max_repetitions);
}
#endif

#ifdef TSL_CONTAINS_AVX512
TEST_CASE("Indirect Sort, Inplace. UI8-UI8, AVX512", "[ui8-ui8][avx512][inplace]") {
  dispatch_type<tsl::simd<uint8_t, tsl::avx512>, tsl::simd<uint8_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                           global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI8-UI16, AVX512", "[ui8-ui16][avx512][inplace]") {
  dispatch_type<tsl::simd<uint8_t, tsl::avx512>, tsl::simd<uint16_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                            global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI8-UI32, AVX512", "[ui8-ui32][avx512][inplace]") {
  dispatch_type<tsl::simd<uint8_t, tsl::avx512>, tsl::simd<uint32_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                            global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI8-UI64, AVX512", "[ui8-ui64][avx512][inplace]") {
  dispatch_type<tsl::simd<uint8_t, tsl::avx512>, tsl::simd<uint64_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                            global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. UI16-UI16, AVX512", "[ui16-ui16][avx512][inplace]") {
  dispatch_type<tsl::simd<uint16_t, tsl::avx512>, tsl::simd<uint16_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                             global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI16-UI32, AVX512", "[ui16-ui32][avx512][inplace]") {
  dispatch_type<tsl::simd<uint16_t, tsl::avx512>, tsl::simd<uint32_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                             global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI16-UI64, AVX512", "[ui16-ui64][avx512][inplace]") {
  dispatch_type<tsl::simd<uint16_t, tsl::avx512>, tsl::simd<uint64_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                             global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. UI32-UI32, AVX512", "[ui32-ui32][avx512][inplace]") {
  dispatch_type<tsl::simd<uint32_t, tsl::avx512>, tsl::simd<uint32_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                             global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI32-UI64, AVX512", "[ui32-ui64][avx512][inplace]") {
  dispatch_type<tsl::simd<uint32_t, tsl::avx512>, tsl::simd<uint64_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                             global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. UI64-UI64, AVX512", "[ui64-ui64][avx512][inplace]") {
  dispatch_type<tsl::simd<uint64_t, tsl::avx512>, tsl::simd<uint64_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                             global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. i8-i8, AVX512", "[i8-i8][avx512][inplace]") {
  dispatch_type<tsl::simd<int8_t, tsl::avx512>, tsl::simd<int8_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                         global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i8-i16, AVX512", "[i8-i16][avx512][inplace]") {
  dispatch_type<tsl::simd<int8_t, tsl::avx512>, tsl::simd<int16_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                          global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i8-i32, AVX512", "[i8-i32][avx512][inplace]") {
  dispatch_type<tsl::simd<int8_t, tsl::avx512>, tsl::simd<int32_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                          global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i8-i64, AVX512", "[i8-i64][avx512][inplace]") {
  dispatch_type<tsl::simd<int8_t, tsl::avx512>, tsl::simd<int64_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                          global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. i16-i16, AVX512", "[i16-i16][avx512][inplace]") {
  dispatch_type<tsl::simd<int16_t, tsl::avx512>, tsl::simd<int16_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                           global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i16-i32, AVX512", "[i16-i32][avx512][inplace]") {
  dispatch_type<tsl::simd<int16_t, tsl::avx512>, tsl::simd<int32_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                           global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i16-i64, AVX512", "[i16-i64][avx512][inplace]") {
  dispatch_type<tsl::simd<int16_t, tsl::avx512>, tsl::simd<int64_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                           global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. i32-i32, AVX512", "[i32-i32][avx512][inplace]") {
  dispatch_type<tsl::simd<int32_t, tsl::avx512>, tsl::simd<int32_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                           global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i32-i64, AVX512", "[i32-i64][avx512][inplace]") {
  dispatch_type<tsl::simd<int32_t, tsl::avx512>, tsl::simd<int64_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                           global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. i64-i64, AVX512", "[i64-i64][avx512][inplace]") {
  dispatch_type<tsl::simd<int64_t, tsl::avx512>, tsl::simd<int64_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                           global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. f32-ui32, AVX512", "[f32-ui32][avx512][inplace]") {
  dispatch_type<tsl::simd<float, tsl::avx512>, tsl::simd<uint32_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                          global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. f32-ui32, AVX512", "[f32-ui64][avx512][inplace]") {
  dispatch_type<tsl::simd<float, tsl::avx512>, tsl::simd<uint64_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                          global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. f64-ui64, AVX512", "[f64-ui64][avx512][inplace]") {
  dispatch_type<tsl::simd<float, tsl::avx512>, tsl::simd<uint64_t, tsl::avx512>, HS_INPL>(global_max_seeds,
                                                                                          global_max_repetitions);
}
#endif

#ifdef TSL_CONTAINS_NEON
TEST_CASE("Indirect Sort, Inplace. UI8-UI8, NEON", "[ui8-ui8][neon][inplace]") {
  dispatch_type<tsl::simd<uint8_t, tsl::neon>, tsl::simd<uint8_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                       global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI8-UI16, NEON", "[ui8-ui16][neon][inplace]") {
  dispatch_type<tsl::simd<uint8_t, tsl::neon>, tsl::simd<uint16_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                        global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI8-UI32, NEON", "[ui8-ui32][neon][inplace]") {
  dispatch_type<tsl::simd<uint8_t, tsl::neon>, tsl::simd<uint32_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                        global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI8-UI64, NEON", "[ui8-ui64][neon][inplace]") {
  dispatch_type<tsl::simd<uint8_t, tsl::neon>, tsl::simd<uint64_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                        global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. UI16-UI16, NEON", "[ui16-ui16][neon][inplace]") {
  dispatch_type<tsl::simd<uint16_t, tsl::neon>, tsl::simd<uint16_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                         global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI16-UI32, NEON", "[ui16-ui32][neon][inplace]") {
  dispatch_type<tsl::simd<uint16_t, tsl::neon>, tsl::simd<uint32_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                         global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI16-UI64, NEON", "[ui16-ui64][neon][inplace]") {
  dispatch_type<tsl::simd<uint16_t, tsl::neon>, tsl::simd<uint64_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                         global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. UI32-UI32, NEON", "[ui32-ui32][neon][inplace]") {
  dispatch_type<tsl::simd<uint32_t, tsl::neon>, tsl::simd<uint32_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                         global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. UI32-UI64, NEON", "[ui32-ui64][neon][inplace]") {
  dispatch_type<tsl::simd<uint32_t, tsl::neon>, tsl::simd<uint64_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                         global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. UI64-UI64, NEON", "[ui64-ui64][neon][inplace]") {
  dispatch_type<tsl::simd<uint64_t, tsl::neon>, tsl::simd<uint64_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                         global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. i8-i8, NEON", "[i8-i8][neon][inplace]") {
  dispatch_type<tsl::simd<int8_t, tsl::neon>, tsl::simd<int8_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                     global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i8-i16, NEON", "[i8-i16][neon][inplace]") {
  dispatch_type<tsl::simd<int8_t, tsl::neon>, tsl::simd<int16_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                      global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i8-i32, NEON", "[i8-i32][neon][inplace]") {
  dispatch_type<tsl::simd<int8_t, tsl::neon>, tsl::simd<int32_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                      global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i8-i64, NEON", "[i8-i64][neon][inplace]") {
  dispatch_type<tsl::simd<int8_t, tsl::neon>, tsl::simd<int64_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                      global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. i16-i16, NEON", "[i16-i16][neon][inplace]") {
  dispatch_type<tsl::simd<int16_t, tsl::neon>, tsl::simd<int16_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                       global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i16-i32, NEON", "[i16-i32][neon][inplace]") {
  dispatch_type<tsl::simd<int16_t, tsl::neon>, tsl::simd<int32_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                       global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i16-i64, NEON", "[i16-i64][neon][inplace]") {
  dispatch_type<tsl::simd<int16_t, tsl::neon>, tsl::simd<int64_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                       global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. i32-i32, NEON", "[i32-i32][neon][inplace]") {
  dispatch_type<tsl::simd<int32_t, tsl::neon>, tsl::simd<int32_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                       global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. i32-i64, NEON", "[i32-i64][neon][inplace]") {
  dispatch_type<tsl::simd<int32_t, tsl::neon>, tsl::simd<int64_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                       global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. i64-i64, NEON", "[i64-i64][neon][inplace]") {
  dispatch_type<tsl::simd<int64_t, tsl::neon>, tsl::simd<int64_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                       global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. f32-ui32, NEON", "[f32-ui32][neon][inplace]") {
  dispatch_type<tsl::simd<float, tsl::neon>, tsl::simd<uint32_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                      global_max_repetitions);
}
TEST_CASE("Indirect Sort, Inplace. f32-ui32, NEON", "[f32-ui64][neon][inplace]") {
  dispatch_type<tsl::simd<float, tsl::neon>, tsl::simd<uint64_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                      global_max_repetitions);
}

TEST_CASE("Indirect Sort, Inplace. f64-ui64, NEON", "[f64-ui64][neon][inplace]") {
  dispatch_type<tsl::simd<float, tsl::neon>, tsl::simd<uint64_t, tsl::neon>, HS_INPL>(global_max_seeds,
                                                                                      global_max_repetitions);
}
#endif

// Tests for Indirect Gather
#ifdef TSL_CONTAINS_SSE
TEST_CASE("Indirect Sort, Gather. UI8-UI8, SSE", "[ui8-ui8][sse][gather]") {
  dispatch_type<tsl::simd<uint8_t, tsl::sse>, tsl::simd<uint8_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                     global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI8-UI16, SSE", "[ui8-ui16][sse][gather]") {
  dispatch_type<tsl::simd<uint8_t, tsl::sse>, tsl::simd<uint16_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                      global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI8-UI32, SSE", "[ui8-ui32][sse][gather]") {
  dispatch_type<tsl::simd<uint8_t, tsl::sse>, tsl::simd<uint32_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                      global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI8-UI64, SSE", "[ui8-ui64][sse][gather]") {
  dispatch_type<tsl::simd<uint8_t, tsl::sse>, tsl::simd<uint64_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                      global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. UI16-UI16, SSE", "[ui16-ui16][sse][gather]") {
  dispatch_type<tsl::simd<uint16_t, tsl::sse>, tsl::simd<uint16_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                       global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI16-UI32, SSE", "[ui16-ui32][sse][gather]") {
  dispatch_type<tsl::simd<uint16_t, tsl::sse>, tsl::simd<uint32_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                       global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI16-UI64, SSE", "[ui16-ui64][sse][gather]") {
  dispatch_type<tsl::simd<uint16_t, tsl::sse>, tsl::simd<uint64_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                       global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. UI32-UI32, SSE", "[ui32-ui32][sse][gather]") {
  dispatch_type<tsl::simd<uint32_t, tsl::sse>, tsl::simd<uint32_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                       global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI32-UI64, SSE", "[ui32-ui64][sse][gather]") {
  dispatch_type<tsl::simd<uint32_t, tsl::sse>, tsl::simd<uint64_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                       global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. UI64-UI64, SSE", "[ui64-ui64][sse][gather]") {
  dispatch_type<tsl::simd<uint64_t, tsl::sse>, tsl::simd<uint64_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                       global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. UI8-i8, SSE", "[i8-i8][sse][gather]") {
  dispatch_type<tsl::simd<int8_t, tsl::sse>, tsl::simd<int8_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                   global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i8-i16, SSE", "[i8-i16][sse][gather]") {
  dispatch_type<tsl::simd<int8_t, tsl::sse>, tsl::simd<int16_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                    global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i8-i32, SSE", "[i8-i32][sse][gather]") {
  dispatch_type<tsl::simd<int8_t, tsl::sse>, tsl::simd<int32_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                    global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i8-i64, SSE", "[i8-i64][sse][gather]") {
  dispatch_type<tsl::simd<int8_t, tsl::sse>, tsl::simd<int64_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                    global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. i16-i16, SSE", "[i16-i16][sse][gather]") {
  dispatch_type<tsl::simd<int16_t, tsl::sse>, tsl::simd<int16_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                     global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i16-i32, SSE", "[i16-i32][sse][gather]") {
  dispatch_type<tsl::simd<int16_t, tsl::sse>, tsl::simd<int32_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                     global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i16-i64, SSE", "[i16-i64][sse][gather]") {
  dispatch_type<tsl::simd<int16_t, tsl::sse>, tsl::simd<int64_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                     global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. i32-i32, SSE", "[i32-i32][sse][gather]") {
  dispatch_type<tsl::simd<int32_t, tsl::sse>, tsl::simd<int32_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                     global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i32-i64, SSE", "[i32-i64][sse][gather]") {
  dispatch_type<tsl::simd<int32_t, tsl::sse>, tsl::simd<int64_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                     global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. i64-i64, SSE", "[i64-i64][sse][gather]") {
  dispatch_type<tsl::simd<int64_t, tsl::sse>, tsl::simd<int64_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                     global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. f32-ui32, SSE", "[f32-ui32][sse][gather]") {
  dispatch_type<tsl::simd<float, tsl::sse>, tsl::simd<uint32_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                    global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. f32-ui32, SSE", "[f32-ui64][sse][gather]") {
  dispatch_type<tsl::simd<float, tsl::sse>, tsl::simd<uint64_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                    global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. f64-ui64, SSE", "[f64-ui64][sse][gather]") {
  dispatch_type<tsl::simd<float, tsl::sse>, tsl::simd<uint64_t, tsl::sse>, HS_GATH>(global_max_seeds,
                                                                                    global_max_repetitions);
}
#endif

#ifdef TSL_CONTAINS_AVX2
TEST_CASE("Indirect Sort, Gather. UI8-UI8, AVX2", "[ui8-ui8][avx2][gather]") {
  dispatch_type<tsl::simd<uint8_t, tsl::avx2>, tsl::simd<uint8_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                       global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI8-UI16, AVX2", "[ui8-ui16][avx2][gather]") {
  dispatch_type<tsl::simd<uint8_t, tsl::avx2>, tsl::simd<uint16_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                        global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI8-UI32, AVX2", "[ui8-ui32][avx2][gather]") {
  dispatch_type<tsl::simd<uint8_t, tsl::avx2>, tsl::simd<uint32_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                        global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI8-UI64, AVX2", "[ui8-ui64][avx2][gather]") {
  dispatch_type<tsl::simd<uint8_t, tsl::avx2>, tsl::simd<uint64_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                        global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. UI16-UI16, AVX2", "[ui16-ui16][avx2][gather]") {
  dispatch_type<tsl::simd<uint16_t, tsl::avx2>, tsl::simd<uint16_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                         global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI16-UI32, AVX2", "[ui16-ui32][avx2][gather]") {
  dispatch_type<tsl::simd<uint16_t, tsl::avx2>, tsl::simd<uint32_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                         global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI16-UI64, AVX2", "[ui16-ui64][avx2][gather]") {
  dispatch_type<tsl::simd<uint16_t, tsl::avx2>, tsl::simd<uint64_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                         global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. UI32-UI32, AVX2", "[ui32-ui32][avx2][gather]") {
  dispatch_type<tsl::simd<uint32_t, tsl::avx2>, tsl::simd<uint32_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                         global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI32-UI64, AVX2", "[ui32-ui64][avx2][gather]") {
  dispatch_type<tsl::simd<uint32_t, tsl::avx2>, tsl::simd<uint64_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                         global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. UI64-UI64, AVX2", "[ui64-ui64][avx2][gather]") {
  dispatch_type<tsl::simd<uint64_t, tsl::avx2>, tsl::simd<uint64_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                         global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. i8-i8, AVX2", "[i8-i8][avx2][gather]") {
  dispatch_type<tsl::simd<int8_t, tsl::avx2>, tsl::simd<int8_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                     global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i8-i16, AVX2", "[i8-i16][avx2][gather]") {
  dispatch_type<tsl::simd<int8_t, tsl::avx2>, tsl::simd<int16_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                      global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i8-i32, AVX2", "[i8-i32][avx2][gather]") {
  dispatch_type<tsl::simd<int8_t, tsl::avx2>, tsl::simd<int32_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                      global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i8-i64, AVX2", "[i8-i64][avx2][gather]") {
  dispatch_type<tsl::simd<int8_t, tsl::avx2>, tsl::simd<int64_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                      global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. i16-i16, AVX2", "[i16-i16][avx2][gather]") {
  dispatch_type<tsl::simd<int16_t, tsl::avx2>, tsl::simd<int16_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                       global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i16-i32, AVX2", "[i16-i32][avx2][gather]") {
  dispatch_type<tsl::simd<int16_t, tsl::avx2>, tsl::simd<int32_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                       global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i16-i64, AVX2", "[i16-i64][avx2][gather]") {
  dispatch_type<tsl::simd<int16_t, tsl::avx2>, tsl::simd<int64_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                       global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. i32-i32, AVX2", "[i32-i32][avx2][gather]") {
  dispatch_type<tsl::simd<int32_t, tsl::avx2>, tsl::simd<int32_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                       global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i32-i64, AVX2", "[i32-i64][avx2][gather]") {
  dispatch_type<tsl::simd<int32_t, tsl::avx2>, tsl::simd<int64_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                       global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. i64-i64, AVX2", "[i64-i64][avx2][gather]") {
  dispatch_type<tsl::simd<int64_t, tsl::avx2>, tsl::simd<int64_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                       global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. f32-ui32, AVX2", "[f32-ui32][avx2][gather]") {
  dispatch_type<tsl::simd<float, tsl::avx2>, tsl::simd<uint32_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                      global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. f32-ui32, AVX2", "[f32-ui64][avx2][gather]") {
  dispatch_type<tsl::simd<float, tsl::avx2>, tsl::simd<uint64_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                      global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. f64-ui64, AVX2", "[f64-ui64][avx2][gather]") {
  dispatch_type<tsl::simd<float, tsl::avx2>, tsl::simd<uint64_t, tsl::avx2>, HS_GATH>(global_max_seeds,
                                                                                      global_max_repetitions);
}
#endif

#ifdef TSL_CONTAINS_AVX512
TEST_CASE("Indirect Sort, Gather. UI8-UI8, AVX512", "[ui8-ui8][avx512][gather]") {
  dispatch_type<tsl::simd<uint8_t, tsl::avx512>, tsl::simd<uint8_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                           global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI8-UI16, AVX512", "[ui8-ui16][avx512][gather]") {
  dispatch_type<tsl::simd<uint8_t, tsl::avx512>, tsl::simd<uint16_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                            global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI8-UI32, AVX512", "[ui8-ui32][avx512][gather]") {
  dispatch_type<tsl::simd<uint8_t, tsl::avx512>, tsl::simd<uint32_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                            global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI8-UI64, AVX512", "[ui8-ui64][avx512][gather]") {
  dispatch_type<tsl::simd<uint8_t, tsl::avx512>, tsl::simd<uint64_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                            global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. UI16-UI16, AVX512", "[ui16-ui16][avx512][gather]") {
  dispatch_type<tsl::simd<uint16_t, tsl::avx512>, tsl::simd<uint16_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                             global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI16-UI32, AVX512", "[ui16-ui32][avx512][gather]") {
  dispatch_type<tsl::simd<uint16_t, tsl::avx512>, tsl::simd<uint32_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                             global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI16-UI64, AVX512", "[ui16-ui64][avx512][gather]") {
  dispatch_type<tsl::simd<uint16_t, tsl::avx512>, tsl::simd<uint64_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                             global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. UI32-UI32, AVX512", "[ui32-ui32][avx512][gather]") {
  dispatch_type<tsl::simd<uint32_t, tsl::avx512>, tsl::simd<uint32_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                             global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI32-UI64, AVX512", "[ui32-ui64][avx512][gather]") {
  dispatch_type<tsl::simd<uint32_t, tsl::avx512>, tsl::simd<uint64_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                             global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. UI64-UI64, AVX512", "[ui64-ui64][avx512][gather]") {
  dispatch_type<tsl::simd<uint64_t, tsl::avx512>, tsl::simd<uint64_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                             global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. i8-i8, AVX512", "[i8-i8][avx512][gather]") {
  dispatch_type<tsl::simd<int8_t, tsl::avx512>, tsl::simd<int8_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                         global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i8-i16, AVX512", "[i8-i16][avx512][gather]") {
  dispatch_type<tsl::simd<int8_t, tsl::avx512>, tsl::simd<int16_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                          global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i8-i32, AVX512", "[i8-i32][avx512][gather]") {
  dispatch_type<tsl::simd<int8_t, tsl::avx512>, tsl::simd<int32_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                          global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i8-i64, AVX512", "[i8-i64][avx512][gather]") {
  dispatch_type<tsl::simd<int8_t, tsl::avx512>, tsl::simd<int64_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                          global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. i16-i16, AVX512", "[i16-i16][avx512][gather]") {
  dispatch_type<tsl::simd<int16_t, tsl::avx512>, tsl::simd<int16_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                           global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i16-i32, AVX512", "[i16-i32][avx512][gather]") {
  dispatch_type<tsl::simd<int16_t, tsl::avx512>, tsl::simd<int32_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                           global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i16-i64, AVX512", "[i16-i64][avx512][gather]") {
  dispatch_type<tsl::simd<int16_t, tsl::avx512>, tsl::simd<int64_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                           global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. i32-i32, AVX512", "[i32-i32][avx512][gather]") {
  dispatch_type<tsl::simd<int32_t, tsl::avx512>, tsl::simd<int32_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                           global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i32-i64, AVX512", "[i32-i64][avx512][gather]") {
  dispatch_type<tsl::simd<int32_t, tsl::avx512>, tsl::simd<int64_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                           global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. i64-i64, AVX512", "[i64-i64][avx512][gather]") {
  dispatch_type<tsl::simd<int64_t, tsl::avx512>, tsl::simd<int64_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                           global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. f32-ui32, AVX512", "[f32-ui32][avx512][gather]") {
  dispatch_type<tsl::simd<float, tsl::avx512>, tsl::simd<uint32_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                          global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. f32-ui32, AVX512", "[f32-ui64][avx512][gather]") {
  dispatch_type<tsl::simd<float, tsl::avx512>, tsl::simd<uint64_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                          global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. f64-ui64, AVX512", "[f64-ui64][avx512][gather]") {
  dispatch_type<tsl::simd<float, tsl::avx512>, tsl::simd<uint64_t, tsl::avx512>, HS_GATH>(global_max_seeds,
                                                                                          global_max_repetitions);
}
#endif

#ifdef TSL_CONTAINS_NEON
TEST_CASE("Indirect Sort, Gather. UI8-UI8, NEON", "[ui8-ui8][neon][gather]") {
  dispatch_type<tsl::simd<uint8_t, tsl::neon>, tsl::simd<uint8_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                       global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI8-UI16, NEON", "[ui8-ui16][neon][gather]") {
  dispatch_type<tsl::simd<uint8_t, tsl::neon>, tsl::simd<uint16_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                        global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI8-UI32, NEON", "[ui8-ui32][neon][gather]") {
  dispatch_type<tsl::simd<uint8_t, tsl::neon>, tsl::simd<uint32_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                        global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI8-UI64, NEON", "[ui8-ui64][neon][gather]") {
  dispatch_type<tsl::simd<uint8_t, tsl::neon>, tsl::simd<uint64_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                        global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. UI16-UI16, NEON", "[ui16-ui16][neon][gather]") {
  dispatch_type<tsl::simd<uint16_t, tsl::neon>, tsl::simd<uint16_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                         global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI16-UI32, NEON", "[ui16-ui32][neon][gather]") {
  dispatch_type<tsl::simd<uint16_t, tsl::neon>, tsl::simd<uint32_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                         global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI16-UI64, NEON", "[ui16-ui64][neon][gather]") {
  dispatch_type<tsl::simd<uint16_t, tsl::neon>, tsl::simd<uint64_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                         global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. UI32-UI32, NEON", "[ui32-ui32][neon][gather]") {
  dispatch_type<tsl::simd<uint32_t, tsl::neon>, tsl::simd<uint32_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                         global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. UI32-UI64, NEON", "[ui32-ui64][neon][gather]") {
  dispatch_type<tsl::simd<uint32_t, tsl::neon>, tsl::simd<uint64_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                         global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. UI64-UI64, NEON", "[ui64-ui64][neon][gather]") {
  dispatch_type<tsl::simd<uint64_t, tsl::neon>, tsl::simd<uint64_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                         global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. i8-i8, NEON", "[i8-i8][neon][gather]") {
  dispatch_type<tsl::simd<int8_t, tsl::neon>, tsl::simd<int8_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                     global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i8-i16, NEON", "[i8-i16][neon][gather]") {
  dispatch_type<tsl::simd<int8_t, tsl::neon>, tsl::simd<int16_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                      global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i8-i32, NEON", "[i8-i32][neon][gather]") {
  dispatch_type<tsl::simd<int8_t, tsl::neon>, tsl::simd<int32_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                      global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i8-i64, NEON", "[i8-i64][neon][gather]") {
  dispatch_type<tsl::simd<int8_t, tsl::neon>, tsl::simd<int64_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                      global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. i16-i16, NEON", "[i16-i16][neon][gather]") {
  dispatch_type<tsl::simd<int16_t, tsl::neon>, tsl::simd<int16_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                       global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i16-i32, NEON", "[i16-i32][neon][gather]") {
  dispatch_type<tsl::simd<int16_t, tsl::neon>, tsl::simd<int32_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                       global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i16-i64, NEON", "[i16-i64][neon][gather]") {
  dispatch_type<tsl::simd<int16_t, tsl::neon>, tsl::simd<int64_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                       global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. i32-i32, NEON", "[i32-i32][neon][gather]") {
  dispatch_type<tsl::simd<int32_t, tsl::neon>, tsl::simd<int32_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                       global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. i32-i64, NEON", "[i32-i64][neon][gather]") {
  dispatch_type<tsl::simd<int32_t, tsl::neon>, tsl::simd<int64_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                       global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. i64-i64, NEON", "[i64-i64][neon][gather]") {
  dispatch_type<tsl::simd<int64_t, tsl::neon>, tsl::simd<int64_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                       global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. f32-ui32, NEON", "[f32-ui32][neon][gather]") {
  dispatch_type<tsl::simd<float, tsl::neon>, tsl::simd<uint32_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                      global_max_repetitions);
}
TEST_CASE("Indirect Sort, Gather. f32-ui32, NEON", "[f32-ui64][neon][gather]") {
  dispatch_type<tsl::simd<float, tsl::neon>, tsl::simd<uint64_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                      global_max_repetitions);
}

TEST_CASE("Indirect Sort, Gather. f64-ui64, NEON", "[f64-ui64][neon][gather]") {
  dispatch_type<tsl::simd<float, tsl::neon>, tsl::simd<uint64_t, tsl::neon>, HS_GATH>(global_max_seeds,
                                                                                      global_max_repetitions);
}
#endif

// TEST_CASE("Indirect Sort, Gather", "[u32-ui32][avx512][gather]") {
//   dispatch_type<tsl::simd<uint32_t, tsl::avx512>, tsl::simd<uint32_t, tsl::avx512>, HS_GATH>(global_max_seeds,
//                                                                                              global_max_repetitions);
// }