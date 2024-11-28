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
 * @file sort_direct.hpp
 * @brief
 */

#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_DIRECT_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_DIRECT_HPP

#include <climits>
#include <cstddef>
#include <iterable.hpp>
#include <tuple>
#include <type_traits>

#include "algorithms/dbops/dbops_hints.hpp"
#include "algorithms/dbops/sort/sort_utils.hpp"
#include "algorithms/utils/hinting.hpp"
#include "tsl.hpp"

namespace tuddbs {
  template <tsl::VectorProcessingStyle _SimdStyle, TSL_SORT_ORDER SortOrderT>
  class SingleColumnSortDirect {
   public:
    using SimdStyle = _SimdStyle;
    using DataT = SimdStyle::base_type;

   private:
    DataT* m_data;

   public:
    explicit SingleColumnSortDirect(SimdOpsIterable auto p_data) : m_data{p_data} {}
    explicit SingleColumnSortDirect(SimdOpsIterable auto p_data, [[maybe_unused]] SimdOpsIterable auto p_idx)
      : m_data{p_data} {}

    auto operator()(const size_t left, const size_t right) {
      if ((right - left) < (4 * 64)) {
        std::sort(m_data + left, m_data + right);
        return;
      }
      const DataT pivot = tuddbs::get_pivot(m_data, left, right);
      partition<SimdStyle>(m_data, left, right, pivot);
    }

   private:
    /**
     * @brief See do_tsl_sort_masked. This function behaves the same, except we do not need to mask any elements,
     * because all elements are valid.
     */
    template <class SimdStyle, typename T = typename SimdStyle::base_type>
    void do_tsl_sort(T* data, const typename SimdStyle::register_type pivot_reg,
                     const typename SimdStyle::register_type val_reg, ssize_t& l_w, ssize_t& r_w) {
      using CmpLeftSide =
        std::conditional_t<SortOrderT == TSL_SORT_ORDER::ASC, tsl::functors::less_than<SimdStyle, tsl::workaround>,
                           tsl::functors::greater_than<SimdStyle, tsl::workaround> >;
      using CmpRightSide =
        std::conditional_t<SortOrderT == TSL_SORT_ORDER::ASC, tsl::functors::greater_than<SimdStyle, tsl::workaround>,
                           tsl::functors::less_than<SimdStyle, tsl::workaround> >;

      const typename SimdStyle::imask_type mask_lt =
        tsl::to_integral<SimdStyle>(CmpLeftSide::apply(val_reg, pivot_reg));
      const typename SimdStyle::imask_type mask_gt =
        tsl::to_integral<SimdStyle>(CmpRightSide::apply(val_reg, pivot_reg));
      const size_t nb_low = tsl::mask_population_count<SimdStyle>(mask_lt);
      const size_t nb_high = tsl::mask_population_count<SimdStyle>(mask_gt);

      tsl::compress_store<SimdStyle>(mask_lt, &data[l_w], val_reg);
      l_w += nb_low;
      r_w -= nb_high;
      tsl::compress_store<SimdStyle>(mask_gt, &data[r_w], val_reg);
    };

    /**
     * @brief Compares a SIMD register from the dataset against a second SIMD register containing a pivot element.
     * Conditionally selects the comparator for ascending or descending sort. val_reg is compared against pivot_reg. All
     * elements, which are smaller than the pivot element are written to the left side write pointer, all elements
     * greater than pivot are written to the right side write pointer. Elements equal to the pivot element are
     * discarded.
     *
     * @tparam SimdStyle The TSL processing style to be used for simdified sorting
     * @tparam SimdStyle::base_type The type of the to be sorted data elements
     * @param data  The data array. Necessary, because we sort inplace.
     * @param pivot_reg A SIMD register containing the pivot element to compare against.
     * @param val_reg A SIMD register containing several values from data.
     * @param l_w The write pointer from the left side of data.
     * @param r_w The write pointer from the right side of data.
     * @param valid A mask to determine elements of a remainder processing. This is a corner case handler for the
     * remainder handling in the CleanUp phase of partition().
     */
    template <class SimdStyle, typename T = typename SimdStyle::base_type>
    void do_tsl_sort_masked(T* data, const typename SimdStyle::register_type pivot_reg,
                            const typename SimdStyle::register_type val_reg, ssize_t& l_w, ssize_t& r_w,
                            const typename SimdStyle::imask_type valid) {
      using CmpLeftSide =
        std::conditional_t<SortOrderT == TSL_SORT_ORDER::ASC, tsl::functors::less_than<SimdStyle, tsl::workaround>,
                           tsl::functors::greater_than<SimdStyle, tsl::workaround> >;
      using CmpRightSide =
        std::conditional_t<SortOrderT == TSL_SORT_ORDER::ASC, tsl::functors::greater_than<SimdStyle, tsl::workaround>,
                           tsl::functors::less_than<SimdStyle, tsl::workaround> >;

      const typename SimdStyle::imask_type mask_lt =
        tsl::to_integral<SimdStyle>(CmpLeftSide::apply(val_reg, pivot_reg)) & valid;
      const typename SimdStyle::imask_type mask_gt =
        tsl::to_integral<SimdStyle>(CmpRightSide::apply(val_reg, pivot_reg)) & valid;
      const size_t nb_low = tsl::mask_population_count<SimdStyle>(mask_lt);
      const size_t nb_high = tsl::mask_population_count<SimdStyle>(mask_gt);

      tsl::compress_store<SimdStyle>(mask_lt, &data[l_w], val_reg);
      l_w += nb_low;
      r_w -= nb_high;
      tsl::compress_store<SimdStyle>(mask_gt, &data[r_w], val_reg);
    };

    /**
     * @brief Recursive divide and conquer method to sort the given array. In each step, data will be sorted in 3
     * partitions [<pivot | pivot | >pivot].
     *
     * @tparam SimdStyle The TSL processing style to be used for simdified sorting
     * @tparam SimdStyle::base_type The type of the to be sorted data elements
     * @param data The array which will be sorted in-place.
     * @param left Where should the sorting start in the array
     * @param right The index after the rightmost element. Sorting an array with 10 elements, from start to end, would
     * require 10 as value.
     * @param pivot The element to compare against. Must be an actual element of the dataset. Will be determined
     * recursively.
     */
    template <class SimdStyle, typename T = typename SimdStyle::base_type>
    void partition(T* data, ssize_t left, ssize_t right, T pivot) {
      const size_t VEC_SIZE = SimdStyle::vector_element_count();
      const ssize_t left_start = left;
      const ssize_t right_start = right;

      // Broadcast the pivot element to a SIMD register
      const auto pivot_vec = tsl::set1<SimdStyle>(pivot);
      ssize_t left_w = left;
      ssize_t right_w = right;

      /* Load data and Index from left side */
      // First register
      typename SimdStyle::register_type vals_l = tsl::loadu<SimdStyle>(&data[left]);
      left += VEC_SIZE;

      // Preload second register. We must do that, because we need to preserve data in order to avoid accidently
      // overwriting elements, which have not been sorted yet.
      typename SimdStyle::register_type vals_l_adv = tsl::loadu<SimdStyle>(&data[left]);
      left += VEC_SIZE;

      /* Load data and Index from left side */
      // First register
      right -= VEC_SIZE;
      typename SimdStyle::register_type vals_r = tsl::loadu<SimdStyle>(&data[right]);
      // Preload second register. We must do that, because we need to preserve data in order to avoid accidently
      // overwriting elements, which have not been sorted yet.
      right -= VEC_SIZE;
      typename SimdStyle::register_type vals_r_adv = tsl::loadu<SimdStyle>(&data[right]);

      /* Working copies */
      typename SimdStyle::register_type vals;
      typename SimdStyle::register_type remainder_vec;
      bool last_iteration_left = false;

      // We have exactly 4 Registers to sort and already preloaded them. Apply the sorting to all and be done with it.
      if (left == right) {
        do_tsl_sort<SimdStyle>(data, pivot_vec, vals_l, left_w, right_w);
        do_tsl_sort<SimdStyle>(data, pivot_vec, vals_l_adv, left_w, right_w);
        do_tsl_sort<SimdStyle>(data, pivot_vec, vals_r, left_w, right_w);
        do_tsl_sort<SimdStyle>(data, pivot_vec, vals_r_adv, left_w, right_w);
      } else if (left + VEC_SIZE > right) {
        // We have 4 preloaded registers, but some leftover elements, which do not fill up an entire register. Preserve
        // the remainder, apply the sort to the preloaded elements. The preloaded remainder register contains elements,
        // that were already preloaded before and must thus be masked as invalid.
        const typename SimdStyle::imask_type valid_mask = (1ull << (right - left)) - 1;
        remainder_vec = tsl::loadu<SimdStyle>(&data[left]);

        do_tsl_sort<SimdStyle>(data, pivot_vec, vals_l, left_w, right_w);
        do_tsl_sort<SimdStyle>(data, pivot_vec, vals_l_adv, left_w, right_w);
        do_tsl_sort<SimdStyle>(data, pivot_vec, vals_r, left_w, right_w);
        do_tsl_sort<SimdStyle>(data, pivot_vec, vals_r_adv, left_w, right_w);

        /* Remainder */
        do_tsl_sort_masked<SimdStyle>(data, pivot_vec, remainder_vec, left_w, right_w, valid_mask);
      } else {
        // We preloaded 4 registers and there are more elements left than a single SIMD register can hold. After sorting
        // a register, we check if we have to read from the left or right side of the array, depending on how many
        // elements were written to either side. This part is iteratively repeated, until one of the previous corner
        // cases appears during processing.
        while (left + VEC_SIZE <= right) {
          const ssize_t left_con = (left - left_w);
          const ssize_t right_con = (right_w - right);
          if (left_con <= right_con) {
            vals = vals_l;
            vals_l = vals_l_adv;
            vals_l_adv = tsl::loadu<SimdStyle>(&data[left]);
            left += VEC_SIZE;
            last_iteration_left = true;
          } else {
            vals = vals_r;
            vals_r = vals_r_adv;
            right -= VEC_SIZE;
            vals_r_adv = tsl::loadu<SimdStyle>(&data[right]);
            last_iteration_left = false;
          }

          do_tsl_sort<SimdStyle>(data, pivot_vec, vals, left_w, right_w);
        }

        /* Cleanup Phase */
        if ((left < right) && (left + VEC_SIZE > right)) {
          const typename SimdStyle::imask_type valid_mask = (1ull << (right - left)) - 1;
          remainder_vec = tsl::loadu<SimdStyle>(&data[left]);
          do_tsl_sort<SimdStyle>(data, pivot_vec, vals_l, left_w, right_w);
          do_tsl_sort<SimdStyle>(data, pivot_vec, vals_l_adv, left_w, right_w);
          do_tsl_sort<SimdStyle>(data, pivot_vec, vals_r, left_w, right_w);
          do_tsl_sort<SimdStyle>(data, pivot_vec, vals_r_adv, left_w, right_w);

          /* Remainder */
          do_tsl_sort_masked<SimdStyle>(data, pivot_vec, remainder_vec, left_w, right_w, valid_mask);
        } else {
          do_tsl_sort<SimdStyle>(data, pivot_vec, vals_l, left_w, right_w);
          do_tsl_sort<SimdStyle>(data, pivot_vec, vals_l_adv, left_w, right_w);
          do_tsl_sort<SimdStyle>(data, pivot_vec, vals_r, left_w, right_w);
          do_tsl_sort<SimdStyle>(data, pivot_vec, vals_r_adv, left_w, right_w);
        }
      }

      // We know the pivot element and thus can just write it in the "gap" which is left empty, because do_tsl_sort
      // ignores elements equal to pivot.
      for (size_t i = left_w; i < right_w; ++i) {
        data[i] = pivot;
      }

      /* -- Left Side Recursive Descend -- */
      if ((left_w == left_start) || ((left_w - left_start) < (4 * VEC_SIZE))) {
        // If we have fewer elements than 4 registers can hold, apply the scalar fallback, based on the required sorting
        // order.
        if constexpr (SortOrderT == TSL_SORT_ORDER::ASC) {
          std::sort(data + left_start, data + left_w);
        } else {
          std::sort(data + left_start, data + left_w, std::greater<T>{});
        }
      } else {
        const auto pivot_l = get_pivot(data, left_start, left_w - 1);
        partition<SimdStyle>(data, left_start, left_w, pivot_l);
      }

      /* -- Right Side Recursive Descend -- */
      if ((right_start == right_w) || ((right_start - right_w) < (4 * VEC_SIZE))) {
        // If we have fewer elements than 4 registers can hold, apply the scalar fallback, based on the required sorting
        // order.
        if constexpr (SortOrderT == TSL_SORT_ORDER::ASC) {
          std::sort(data + right_w, data + right_start);
        } else {
          std::sort(data + right_w, data + right_start, std::greater<T>{});
        }
      } else {
        const auto pivot_r = get_pivot(data, left_w, right_start - 1);
        partition<SimdStyle>(data, right_w, right_start, pivot_r);
      }
    }
  };

}  // namespace tuddbs

#endif