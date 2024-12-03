// ------------------------------------------------------------------- //
/*
   This file is part of the SimdOperators Project.
   Author(s): Alexander Krause.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, version 3.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
// ------------------------------------------------------------------- //

/**
 * @file sort_core_inplace.hpp
 * @brief
 */

#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_CORE_GATHER_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_CORE_GATHER_HPP

#include "algorithms/dbops/sort/sort_core.hpp"
#include "algorithms/dbops/sort/sort_utils.hpp"

namespace tuddbs {
  namespace gather_sort {
    template <typename T, typename U>
    void detect_cluster(std::deque<Cluster>& clusters, const T* const data, U* indexes, size_t left, size_t right) {
      T run_value;
      size_t run_length = 1;
      if (clusters.size() == 0) {
        run_length = 1;
        left = 0;
        run_value = data[indexes[0]];
      } else {
        Cluster& previous_cluster = clusters.back();
        left = previous_cluster.start + previous_cluster.len;
        run_value = data[indexes[left]];
        const T previous_run_value = data[indexes[previous_cluster.start]];
      }
      ++left;
      if (left >= right) {
        return;
      }
      for (; left < right; ++left) {
        const T current_value = data[indexes[left]];
        if (current_value != run_value) {
          if (run_length > 1) {
            clusters.emplace_back(left - run_length, run_length);
          }
          run_length = 1;
          run_value = current_value;
        } else {
          ++run_length;
        }
      }
      if (run_length > 1) {
        clusters.emplace_back(left - run_length, run_length);
      }
    }

    /**
     * @brief Template based helper to set the pivot register based on the actual processing style
     *
     * @tparam SimdStyle Provides the size of the data values, potentially used to populate the pivot register
     * @tparam IndexStyle Provides the integral mask definition, potentially used to populate the pivot register
     */
    template <class SimdStyle, class IndexStyle>
    __attribute__((always_inline)) auto set_pivot(typename SimdStyle::base_type pivot) {
      // If the base type is (u)int8_t, we use the Index style to set the pivot register
      if constexpr (sizeof(typename SimdStyle::base_type) == 1) {
        return tsl::set1<IndexStyle>(pivot);
      } else {
        return tsl::set1<SimdStyle>(pivot);
      }
    }

    // Called when there are less than 4 vector registers left
    template <TSL_SORT_ORDER order, typename T, typename U>
    void insertion_sort_fallback(T const* __restrict__ data, U* __restrict__ indexes, const ssize_t left_boundary,
                                 const ssize_t right_boundary) {
      ssize_t i, j;
      T val;
      U idx;
      for (i = left_boundary + 1; i < right_boundary; i++) {
        idx = indexes[i];
        val = data[idx];
        j = i - 1;

        if constexpr (order == TSL_SORT_ORDER::ASC) {
          while (j >= left_boundary && data[indexes[j]] > val) {
            indexes[j + 1] = indexes[j];
            j = j - 1;
          }
        } else {
          while (j >= left_boundary && data[indexes[j]] < val) {
            indexes[j + 1] = indexes[j];
            j = j - 1;
          }
        }
        indexes[j + 1] = idx;
      }
    };

    /**
     * @brief Templated helper to distinguish the return type for gathered elements, i.e. if the data elements are
     * integers or floating point values
     */
    template <class SimdStyle, class IndexStyle, typename data_t = typename SimdStyle::base_type,
              typename idx_t = typename IndexStyle::base_type>
    __attribute__((always_inline)) inline auto gather_data(
      const data_t* data, const typename IndexStyle::register_type idx_reg,
      const typename IndexStyle::register_type gather_mask_register) {
      if constexpr (std::is_floating_point_v<data_t>) {
        return tsl::reinterpret<IndexStyle, SimdStyle>(tsl::binary_and<IndexStyle>(
          tsl::gather<IndexStyle>(data, idx_reg, TSL_CVAL(int, sizeof(data_t))), gather_mask_register));
      } else {
        return tsl::binary_and<IndexStyle>(tsl::gather<IndexStyle>(data, idx_reg, TSL_CVAL(int, sizeof(data_t))),
                                           gather_mask_register);
      }
    }

    /**
     * @brief Templated helper for produce a bitmask using either SimdStyle or IndexStyle, based on the data type and
     * index type combination.
     *
     * @tparam SimdStyle TSL processing style for data elements, if the base_type is 1 Byte wide or signed (otherwise
     * the gather_mask truncation will do bad things to the sign bit).
     * @tparam IndexStyle TSL processing style for data elements, if the base_type is a 2 Byte or wider usigned integer
     * @tparam type The comparison operation, i.e. LessThan or Equal
     * @tparam SimdStyle::base_type The type of the data elements
     * @tparam IndexStyle::base_type The type of the index elements
     */
    template <class SimdStyle, class IndexStyle, SORT_TYPE type, TSL_SORT_ORDER order,
              typename T = typename SimdStyle::base_type, typename U = typename IndexStyle::base_type>
    __attribute__((always_inline)) inline auto get_bitmask(const typename SimdStyle::register_type val_reg,
                                                           const typename SimdStyle::register_type pivot_reg) ->
      typename IndexStyle::imask_type {
      using mask_t = typename SimdStyle::imask_type;
      using idx_mask_t = typename IndexStyle::imask_type;

      if constexpr ((sizeof(typename SimdStyle::base_type) == 1) &&
                    (!std::is_signed_v<typename SimdStyle::base_type>)) {
        return compare<IndexStyle, type, order>(val_reg, pivot_reg);
      } else {
        // The integral mask differs based on the ratio of data type to index type.
        // We need to shift the bits of the data type to the position, where the index
        // processing style would expect it to be.
        const mask_t uncured_mask_lt = compare<SimdStyle, type, order>(val_reg, pivot_reg);
        constexpr size_t cure_shift_distance = sizeof(U) / sizeof(T);
        constexpr size_t cure_shift_count = IndexStyle::vector_element_count();
        idx_mask_t mask_lt = 0;
        for (size_t i = 0; i < cure_shift_count; ++i) {
          mask_lt |= (((uncured_mask_lt >> (i * cure_shift_distance)) & 0b1) << i);
        }
        return mask_lt;
      }
    }

    /**
     * @brief Processes a vector register with all-valid elements
     *
     * @tparam SimdStyle TSL processing style for data elements
     * @tparam IndexStyle TSL processing style for index elements
     * @tparam type The comparison function, i.e. LessThan or Equal
     * @tparam SimdStyle::base_type Derived type for the data elements
     * @tparam IndexStyle::base_type Derived type for the index elements
     */
    template <class SimdStyle, class IndexStyle, SORT_TYPE type, TSL_SORT_ORDER order,
              typename T = typename SimdStyle::base_type, typename U = typename IndexStyle::base_type>
    __attribute__((always_inline)) void do_tsl_sort(T* data, U* indexes,
                                                    const typename SimdStyle::register_type pivot_reg,
                                                    const typename IndexStyle::register_type idx_reg, size_t& l_w,
                                                    size_t& r_w) {
      using mask_t = typename SimdStyle::imask_type;
      using idx_mask_t = typename IndexStyle::imask_type;

      const U gather_mask_value = (sizeof(T) == 8) ? (-1ull) : ((1ull << (sizeof(T) * CHAR_BIT)) - 1);
      const typename IndexStyle::register_type gather_mask_register = tsl::set1<IndexStyle>(gather_mask_value);
      const auto val_reg = gather_data<SimdStyle, IndexStyle>(data, idx_reg, gather_mask_register);

      const auto mask_lt = get_bitmask<SimdStyle, IndexStyle, type, order>(val_reg, pivot_reg);

      const size_t nb_low = tsl::mask_population_count<IndexStyle>(mask_lt);
      const size_t nb_high = IndexStyle::vector_element_count() - nb_low;

      tsl::compress_store<IndexStyle>(mask_lt, &indexes[l_w], idx_reg);
      l_w += nb_low;
      r_w -= nb_high;
      tsl::compress_store<IndexStyle>(static_cast<mask_t>(~mask_lt), &indexes[r_w], idx_reg);
    }

    /**
     * @brief Processes a remainder-register, i.e. not all lanes are valid.
     *
     * @tparam SimdStyle TSL processing style for data elements
     * @tparam IndexStyle TSL processing style for index elements
     * @tparam type The comparison function, i.e. LessThan or Equal
     * @tparam SimdStyle::base_type Derived type for the data elements
     * @tparam IndexStyle::base_type Derived type for the index elements
     * @param data The actual data array
     * @param indexes The index array corresponding to data
     * @param pivot_reg A register containing the current pivot element
     * @param idx_reg A register containing the to-be-sorted indexes
     * @param l_w left-side write pointer
     * @param r_w right-side write pointer
     * @param valid A mask determining the valid elements in the idx_reg
     */
    template <class SimdStyle, class IndexStyle, SORT_TYPE type, TSL_SORT_ORDER order,
              typename T = typename SimdStyle::base_type, typename U = typename IndexStyle::base_type>
    inline void do_tsl_sort_masked(T const* __restrict__ data, U* __restrict__ indexes, const auto pivot_reg,
                                   const typename IndexStyle::register_type idx_reg, size_t& l_w, size_t& r_w,
                                   const typename IndexStyle::imask_type valid) {
      using mask_t = typename SimdStyle::imask_type;
      using idx_mask_t = typename IndexStyle::imask_type;

      /**
       * If we perform a gather with an index type that is wider than the data type,
       * we load more bits than we need. This mask nullifies all bits that are part
       * of another data value. E.g. Gathering uint8_t values with a uint64_t index
       * results in loading 8x 8bit values per index, thus we only want to keep bits[0:7]
       * for the LT or EQ comparison.
       */
      const U gather_mask_value = (sizeof(T) == 8) ? (-1ull) : ((1ull << (sizeof(T) * CHAR_BIT)) - 1);
      const typename IndexStyle::register_type gather_mask_register = tsl::set1<IndexStyle>(gather_mask_value);
      const auto val_reg = gather_data<SimdStyle, IndexStyle>(data, idx_reg, gather_mask_register);

      // Create a bitmask for
      const auto mask_lt = get_bitmask<SimdStyle, IndexStyle, type, order>(val_reg, pivot_reg) & valid;
      const idx_mask_t mask_gt = static_cast<mask_t>(~mask_lt) & valid;

      const size_t nb_low = tsl::mask_population_count<IndexStyle>(mask_lt);
      const size_t nb_high = tsl::mask_population_count<IndexStyle>(mask_gt);

      tsl::compress_store<IndexStyle>(mask_lt, &indexes[l_w], idx_reg);
      l_w += nb_low;
      r_w -= nb_high;
      tsl::compress_store<IndexStyle>(mask_gt, &indexes[r_w], idx_reg);
    }
  }  // namespace gather_sort
}  // namespace tuddbs
#endif