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

#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_CORE_INPLACE_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_CORE_INPLACE_HPP

#include "algorithms/dbops/sort/sort_core.hpp"
#include "algorithms/dbops/sort/sort_utils.hpp"

namespace tuddbs {
  namespace sort_inplace {
    template <typename T, typename U>
    void detect_cluster(std::deque<Cluster>& clusters, T* data, U* indexes, size_t left, size_t right) {
      T run_value = data[left];
      size_t run_length = 1;
      ++left;
      if (left > right) {
        return;
      }
      for (; left < right; ++left) {
        const T current_value = data[left];
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

    template <TSL_SORT_ORDER order, typename T, typename U>
    void insertion_sort_fallback(T* data, U* indexes, const ssize_t left_boundary, const ssize_t right_boundary) {
      const size_t idxs_count = (right_boundary - left_boundary);
      ssize_t i, j;
      T val;
      U idx;
      for (i = left_boundary + 1; i < right_boundary; i++) {
        val = data[i];
        idx = indexes[i];
        j = i - 1;

        if constexpr (order == TSL_SORT_ORDER::ASC) {
          while (j >= 0 && data[j] > val) {
            data[j + 1] = data[j];
            indexes[j + 1] = indexes[j];
            j = j - 1;
          }
        } else {
          while (j >= 0 && data[j] < val) {
            data[j + 1] = data[j];
            indexes[j + 1] = indexes[j];
            j = j - 1;
          }
        }
        data[j + 1] = val;
        indexes[j + 1] = idx;
      }
    };

    template <class S, class I>
    constexpr size_t idx_arr_len = sizeof(typename I::base_type) / sizeof(typename S::base_type);

    template <class S, class I>
    constexpr size_t bits_per_idx_register =
      (I::vector_element_count() == 64) ? -1ull : (1ull << I::vector_element_count()) - 1;

    template <class SimdStyle, class IndexStyle, typename U = typename IndexStyle::base_type>
    inline void compress_store_index_array(typename SimdStyle::imask_type full_mask, U* indexes,
                                           const idx_arr_t<SimdStyle, IndexStyle> idx_tmparr) {
#pragma unroll(idx_arr_len<SimdStyle, IndexStyle>)
      for (size_t i = 0; i < idx_arr_len<SimdStyle, IndexStyle>; ++i) {
        /**
         * No binary masking necessary, as we guarantee that the IndexStyle::base_type is equally sized or wider
         * than SimdStyle::base_type. Thus, IndexStyle::imask_type will always be smaller or equally sized than
         * SimdStyle::base_type --> Upper/Unnecessary bits will be automatically truncated.
         */
        const typename IndexStyle::imask_type idx_mask = static_cast<typename IndexStyle::imask_type>(
          (full_mask >> (i * IndexStyle::vector_element_count())) & bits_per_idx_register<SimdStyle, IndexStyle>);
        tsl::compress_store<IndexStyle>(idx_mask, indexes,
                                        tsl::loadu<IndexStyle>(&idx_tmparr[i * IndexStyle::vector_element_count()]));
        /**
         * We always 'just' advance the indexes pointer by the amount of written elements, both for left and right
         * side. That is, for left we always start at the 'first' write position. For right, we position the indexes
         * pointer at the leftmost position, since we know nb_high from the full_mask.
         */
        indexes += tsl::mask_population_count<IndexStyle>(idx_mask);
      }
    }

    template <class SimdStyle, class IndexStyle, SORT_TYPE type, TSL_SORT_ORDER order,
              typename T = typename SimdStyle::base_type, typename U = typename IndexStyle::base_type>
    inline void do_tsl_sort(T* data, U* indexes, const typename SimdStyle::register_type pivot_reg,
                            const typename SimdStyle::register_type val_reg,
                            const idx_arr_t<SimdStyle, IndexStyle> idx_tmparr, ssize_t& l_w, ssize_t& r_w) {
      using mask_t = typename SimdStyle::imask_type;
      const mask_t mask_lt = compare<SimdStyle, type, order>(val_reg, pivot_reg);
      const size_t nb_low = tsl::mask_population_count<SimdStyle>(mask_lt);
      const size_t nb_high = SimdStyle::vector_element_count() - nb_low;

      tsl::compress_store<SimdStyle>(mask_lt, &data[l_w], val_reg);
      compress_store_index_array<SimdStyle, IndexStyle>(mask_lt, &indexes[l_w], idx_tmparr);
      l_w += nb_low;
      r_w -= nb_high;
      tsl::compress_store<SimdStyle>(static_cast<mask_t>(~mask_lt), &data[r_w], val_reg);
      compress_store_index_array<SimdStyle, IndexStyle>(static_cast<mask_t>(~mask_lt), &indexes[r_w], idx_tmparr);
    }

    template <class SimdStyle, class IndexStyle, SORT_TYPE type, TSL_SORT_ORDER order,
              typename T = typename SimdStyle::base_type, typename U = typename IndexStyle::base_type>
    inline void do_tsl_sort_masked(T* data, U* indexes, const typename SimdStyle::register_type pivot_reg,
                                   const typename SimdStyle::register_type val_reg,
                                   const idx_arr_t<SimdStyle, IndexStyle> idx_tmparr, ssize_t& l_w, ssize_t& r_w,
                                   const typename SimdStyle::imask_type valid) {
      using mask_t = typename SimdStyle::imask_type;

      const mask_t mask_lt = compare<SimdStyle, type, order>(val_reg, pivot_reg, valid);
      const mask_t mask_gt = static_cast<mask_t>(~mask_lt) & valid;

      const size_t nb_low = tsl::mask_population_count<SimdStyle>(mask_lt);
      const size_t nb_high = tsl::mask_population_count<SimdStyle>(mask_gt);

      tsl::compress_store<SimdStyle>(mask_lt, &data[l_w], val_reg);
      compress_store_index_array<SimdStyle, IndexStyle>(mask_lt, &indexes[l_w], idx_tmparr);
      l_w += nb_low;
      r_w -= nb_high;
      tsl::compress_store<SimdStyle>(mask_gt, &data[r_w], val_reg);
      compress_store_index_array<SimdStyle, IndexStyle>(mask_gt, &indexes[r_w], idx_tmparr);
    }
  }  // namespace sort_inplace
}  // namespace tuddbs
#endif