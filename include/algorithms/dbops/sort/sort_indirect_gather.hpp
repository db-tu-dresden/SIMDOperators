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
 * @file sort_indirect_gather.hpp
 * @brief
 */

#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_INDIRECT_GATHER_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_INDIRECT_GATHER_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "sort_utils.hpp"
#include "tslintrin.hpp"

#pragma once

namespace tuddbs {
  namespace sort_indirect_gather {

    /**
     * @brief Templated helper to make calling the comparison with TYPE a one-liner and increase readability
     *
     * @tparam SimdStyle TSL processing style to compare the value register with the pivot register
     * @tparam type The comparison type, i.e. LessThan or Equal
     */
    template <class SimdStyle, SORT_TYPE type, TSL_SORT_ORDER order>
    __attribute__((always_inline)) inline auto compare(const typename SimdStyle::register_type val_reg,
                                                       const typename SimdStyle::register_type pivot_reg) ->
      typename SimdStyle::imask_type {
      if constexpr (type == SORT_TYPE::SORT_EQ) {
        return tsl::to_integral<SimdStyle>(tsl::equal<SimdStyle>(val_reg, pivot_reg));
      } else {
        if constexpr (order == TSL_SORT_ORDER::ASC) {
          return tsl::to_integral<SimdStyle>(tsl::less_than<SimdStyle>(val_reg, pivot_reg));
        } else {
          return tsl::to_integral<SimdStyle>(tsl::greater_than<SimdStyle>(val_reg, pivot_reg));
        }
      }
    }

    /**
     * @brief Templated helper to ease the call to compare with a remainder vector register
     */
    template <class SimdStyle, SORT_TYPE type, TSL_SORT_ORDER order>
    __attribute__((always_inline)) inline auto compare(const typename SimdStyle::register_type val_reg,
                                                       const typename SimdStyle::register_type pivot_reg,
                                                       const typename SimdStyle::imask_type valid) ->
      typename SimdStyle::imask_type {
      return compare<SimdStyle, type, order>(val_reg, pivot_reg) & valid;
    }

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
    inline void do_tsl_sort_masked(T* data, U* indexes, const auto pivot_reg,
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
    void insertion_sort_fallback(T* data, U* indexes, const ssize_t left_boundary, const ssize_t right_boundary) {
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

    template <class SimdStyle, class IndexStyle, TSL_SORT_ORDER SortOrderT, typename T = typename SimdStyle::base_type,
              typename U = typename IndexStyle::base_type>
    void partition(T* data, U* indexes, size_t left, size_t right, T pivot) {
      static_assert(sizeof(T) <= sizeof(U), "The index type (U) must be at least as wide as the data type (T).");

      using idx_reg_t = typename IndexStyle::register_type;

      const size_t left_start = left;
      const size_t right_start = right;

      // Determine the data type for the pivot vector register depending on the DataType<>IndexType combination
      const auto pivot_vec = set_pivot<SimdStyle, IndexStyle>(pivot);
      size_t left_w = left;
      size_t right_w = right;

      /* Load data and Index from left side */
      // First register
      idx_reg_t idx_l = tsl::loadu<IndexStyle>(&indexes[left]);
      left += IndexStyle::vector_element_count();

      // Preload second register
      idx_reg_t idx_l_adv = tsl::loadu<IndexStyle>(&indexes[left]);
      left += IndexStyle::vector_element_count();

      /* Load data and Index from right side */
      // First register
      right -= IndexStyle::vector_element_count();
      idx_reg_t idx_r = tsl::loadu<IndexStyle>(&indexes[right]);

      // Preload second register
      right -= IndexStyle::vector_element_count();
      idx_reg_t idx_r_adv = tsl::loadu<IndexStyle>(&indexes[right]);

      /* Working copies */
      idx_reg_t idxs;

      if (left == right) {
        // If left equals right, we have buffered exactly 4 registers and can apply the sort directly
        do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, idx_l, left_w,
                                                                           right_w);
        do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, idx_l_adv, left_w,
                                                                           right_w);
        do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, idx_r, left_w,
                                                                           right_w);
        do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, idx_r_adv, left_w,
                                                                           right_w);
      } else if (left + IndexStyle::vector_element_count() > right) {
        // If there is less than a vector register remaining between left and right,
        // we load the remainder and mask out everything that we already loaded/processed in right

        const typename IndexStyle::imask_type valid_mask = ((1ull << (right - left)) - 1) & (-1ull >> 1);
        const idx_reg_t remainder_idx_vec = tsl::loadu<IndexStyle>(&indexes[left]);

        do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, idx_l, left_w,
                                                                           right_w);
        do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, idx_l_adv, left_w,
                                                                           right_w);
        do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, idx_r, left_w,
                                                                           right_w);
        do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, idx_r_adv, left_w,
                                                                           right_w);

        /* Remainder */
        do_tsl_sort_masked<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
          data, indexes, pivot_vec, remainder_idx_vec, left_w, right_w, valid_mask);
      } else {
        // left and right are multiple vector registers apart, let's do the algorithm
        while (left + IndexStyle::vector_element_count() <= right) {
          const size_t left_con = (left - left_w);
          const size_t right_con = (right_w - right);
          if (left_con <= right_con) {
            idxs = idx_l;
            idx_l = idx_l_adv;
            idx_l_adv = tsl::loadu<IndexStyle>(&indexes[left]);
            left += IndexStyle::vector_element_count();
          } else {
            idxs = idx_r;
            idx_r = idx_r_adv;
            right -= IndexStyle::vector_element_count();
            idx_r_adv = tsl::loadu<IndexStyle>(&indexes[right]);
          }

          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, idxs, left_w,
                                                                             right_w);
        }

        /* -- Cleanup Phase -- */
        if ((left < right) && (left + IndexStyle::vector_element_count() > right)) {
          const typename IndexStyle::imask_type valid_mask = ((1ull << (right - left)) - 1) & (-1ull >> 1);
          const idx_reg_t remainder_idx_vec = tsl::loadu<IndexStyle>(&indexes[left]);

          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, idx_l, left_w,
                                                                             right_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, idx_l_adv,
                                                                             left_w, right_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, idx_r, left_w,
                                                                             right_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, idx_r_adv,
                                                                             left_w, right_w);

          /* Remainder */
          do_tsl_sort_masked<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
            data, indexes, pivot_vec, remainder_idx_vec, left_w, right_w, valid_mask);
        } else {
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, idx_l, left_w,
                                                                             right_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, idx_l_adv,
                                                                             left_w, right_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, idx_r, left_w,
                                                                             right_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, idx_r_adv,
                                                                             left_w, right_w);
        }
      }

      /* Start partitioning the right side into [ values == PIVOT | values > Pivot ] */
      size_t pivot_l = left_w;
      size_t pivot_l_w = left_w;

      size_t pivot_r = right_start;
      size_t pivot_r_w = right_start;

      if ((pivot_r - pivot_l) < 4 * IndexStyle::vector_element_count()) {
        /* Data too small to fit in 4 registers */
        insertion_sort_fallback<SortOrderT>(data, indexes, pivot_l, pivot_r);

        for (size_t i = pivot_l; i < right_start; ++i) {
          if (data[i] > pivot) {
            pivot_r_w = i;
            break;
          }
        }
      } else {
        idx_l = tsl::loadu<IndexStyle>(&indexes[pivot_l]);
        pivot_l += IndexStyle::vector_element_count();

        idx_l_adv = tsl::loadu<IndexStyle>(&indexes[pivot_l]);
        pivot_l += IndexStyle::vector_element_count();

        pivot_r -= IndexStyle::vector_element_count();
        idx_r = tsl::loadu<IndexStyle>(&indexes[pivot_r]);

        pivot_r -= IndexStyle::vector_element_count();
        idx_r_adv = tsl::loadu<IndexStyle>(&indexes[pivot_r]);

        if (pivot_l == pivot_r) {
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, idx_l, pivot_l_w,
                                                                             pivot_r_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, idx_l_adv,
                                                                             pivot_l_w, pivot_r_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, idx_r, pivot_l_w,
                                                                             pivot_r_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, idx_r_adv,
                                                                             pivot_l_w, pivot_r_w);
        } else if (pivot_l + IndexStyle::vector_element_count() > pivot_r) {
          const typename IndexStyle::imask_type valid_mask = ((1ull << (pivot_r - pivot_l)) - 1) & (-1ull >> 1);
          const idx_reg_t remainder_idx_vec = tsl::loadu<IndexStyle>(&indexes[pivot_l]);

          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, idx_l, pivot_l_w,
                                                                             pivot_r_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, idx_l_adv,
                                                                             pivot_l_w, pivot_r_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, idx_r, pivot_l_w,
                                                                             pivot_r_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, idx_r_adv,
                                                                             pivot_l_w, pivot_r_w);

          /* Remainder */
          do_tsl_sort_masked<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
            data, indexes, pivot_vec, remainder_idx_vec, pivot_l_w, pivot_r_w, valid_mask);
        } else {
          while (pivot_l + IndexStyle::vector_element_count() <= pivot_r) {
            const size_t left_con = (pivot_l - pivot_l_w);
            const size_t right_con = (pivot_r_w - pivot_r);
            if (left_con <= right_con) {
              idxs = idx_l;

              idx_l = idx_l_adv;

              idx_l_adv = tsl::loadu<IndexStyle>(&indexes[pivot_l]);

              pivot_l += IndexStyle::vector_element_count();
            } else {
              idxs = idx_r;

              idx_r = idx_r_adv;

              pivot_r -= IndexStyle::vector_element_count();
              idx_r_adv = tsl::loadu<IndexStyle>(&indexes[pivot_r]);
            }

            do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, idxs,
                                                                               pivot_l_w, pivot_r_w);
          }

          /* -- Cleanup Phase -- */
          if ((pivot_l < pivot_r) && (pivot_l + IndexStyle::vector_element_count() > pivot_r)) {
            const typename IndexStyle::imask_type valid_mask = ((1ull << (pivot_r - pivot_l)) - 1) & (-1ull >> 1);
            const idx_reg_t remainder_idx_vec = tsl::loadu<IndexStyle>(&indexes[pivot_l]);

            do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, idx_l,
                                                                               pivot_l_w, pivot_r_w);
            do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, idx_l_adv,
                                                                               pivot_l_w, pivot_r_w);
            do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, idx_r,
                                                                               pivot_l_w, pivot_r_w);
            do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, idx_r_adv,
                                                                               pivot_l_w, pivot_r_w);

            /* Remainder */
            do_tsl_sort_masked<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
              data, indexes, pivot_vec, remainder_idx_vec, pivot_l_w, pivot_r_w, valid_mask);
          } else {
            do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, idx_l,
                                                                               pivot_l_w, pivot_r_w);
            do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, idx_l_adv,
                                                                               pivot_l_w, pivot_r_w);
            do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, idx_r,
                                                                               pivot_l_w, pivot_r_w);
            do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, idx_r_adv,
                                                                               pivot_l_w, pivot_r_w);
          }
        }
      }

      /* -- Left Side -- */
      if ((left_w - left_start) < (4 * IndexStyle::vector_element_count())) {
        insertion_sort_fallback<SortOrderT>(data, indexes, left_start, left_w);
      } else {
        const auto pivot_ls = get_pivot_indirect(data, indexes, left_start, left_w - 1);
        partition<SimdStyle, IndexStyle, SortOrderT>(data, indexes, left_start, left_w, pivot_ls);
      }

      /* -- Right Side -- */
      if ((right_start - pivot_r_w) < (4 * IndexStyle::vector_element_count())) {
        insertion_sort_fallback<SortOrderT>(data, indexes, right_w, right_start);
      } else {
        const auto pivot_rs = get_pivot_indirect(data, indexes, pivot_r_w, right_start - 1);
        partition<SimdStyle, IndexStyle, SortOrderT>(data, indexes, pivot_r_w, right_start, pivot_rs);
      }
    }

    /**
     * @brief Wrapper for calling vectorized sorting
     *
     * @tparam SimdStyle The processing style to be used for signed and floating point values
     * @tparam IndexStyle The processing style for all data types, except signed anf floating point
     * @tparam SimdStyle::base_type Derived data type for the data values
     * @tparam IndexStyle::base_type Derived data type for the index values
     * @param data The to-be-sorted array
     * @param indexes Equally sized to data, contains indexes corresponding to elements in data
     * @param left The boundary to start from the left side
     * @param right The boundary to start from the right side
     * @param pivot The pivot element for the range between left and right
     */
    template <class SimdStyle, class IndexStyle, TSL_SORT_ORDER SortOrderT, typename T = typename SimdStyle::base_type,
              typename U = typename IndexStyle::base_type>
    void tsl_sort(T* data, U* indexes, size_t left, size_t right, T pivot) {
      if ((right - left) < (4 * SimdStyle::vector_element_count())) {
        insertion_sort_fallback<SortOrderT>(data, indexes, left, right);
        return;
      }
      partition<SimdStyle, IndexStyle, SortOrderT>(data, indexes, left, right, pivot);
    }
  }  // namespace sort_indirect_gather
}  // namespace tuddbs

#endif