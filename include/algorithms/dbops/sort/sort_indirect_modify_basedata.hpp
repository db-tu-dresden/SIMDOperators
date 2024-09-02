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
 * @file sort_indirect_modify_basedata.hpp
 * @brief
 */

#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_INDIRECT_MODIFY_BASEDATA_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_INDIRECT_MODIFY_BASEDATA_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "sort_utils.hpp"
#include "tslintrin.hpp"

namespace tuddbs {
  namespace sort_indirect_copy_basedata {
    template <class S, class I>
    constexpr size_t idx_arr_len = sizeof(typename I::base_type) / sizeof(typename S::base_type);

    template <class S, class I>
    constexpr size_t bits_per_idx_register =
      (I::vector_element_count() == 64) ? -1ull : (1ull << I::vector_element_count()) - 1;

    template <class S, class I>
    using idx_arr_t = std::array<typename I::base_type, S::vector_element_count()>;

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

    template <class SimdStyle, SORT_TYPE type, tuddbs::TSL_SORT_ORDER order>
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

    template <class SimdStyle, SORT_TYPE type, TSL_SORT_ORDER order>
    __attribute__((always_inline)) inline auto compare(const typename SimdStyle::register_type val_reg,
                                                       const typename SimdStyle::register_type pivot_reg,
                                                       const typename SimdStyle::imask_type valid) ->
      typename SimdStyle::imask_type {
      return compare<SimdStyle, type, order>(val_reg, pivot_reg) & valid;
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
    inline void do_avx_sort_masked(T* data, U* indexes, const typename SimdStyle::register_type pivot_reg,
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

    template <class SimdStyle, class IndexStyle, TSL_SORT_ORDER SortOrderT, typename T = typename SimdStyle::base_type,
              typename U = typename IndexStyle::base_type>
    void partition(T* data, U* indexes, ssize_t left, ssize_t right, T pivot) {
      static_assert(sizeof(T) <= sizeof(U), "The index type (U) must be at least as wide as the data type (T).");

      using data_reg_t = typename SimdStyle::register_type;

      /* For now we assume that IndexStyle will use a wider or equally sized type than SimdStyle for the underlying data
       */
      const auto load_idx_arr = [](U const* memory) -> idx_arr_t<SimdStyle, IndexStyle> {
        idx_arr_t<SimdStyle, IndexStyle> idxs;
#pragma unroll(idx_arr_len)
        for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
          idxs[i] = memory[i];
        }
        return idxs;
      };

      const ssize_t left_start = left;
      const ssize_t right_start = right;

      const auto pivot_vec = tsl::set1<SimdStyle>(pivot);
      ssize_t left_w = left;
      ssize_t right_w = right;

      /* Load data and Index from left side */
      // First register
      data_reg_t vals_l = tsl::loadu<SimdStyle>(&data[left]);
      idx_arr_t<SimdStyle, IndexStyle> idx_l = load_idx_arr(&indexes[left]);
      left += SimdStyle::vector_element_count();

      // Preload second register
      data_reg_t vals_l_adv = tsl::loadu<SimdStyle>(&data[left]);
      idx_arr_t<SimdStyle, IndexStyle> idx_l_adv = load_idx_arr(&indexes[left]);
      left += SimdStyle::vector_element_count();

      /* Load data and Index from right side */
      // First register
      right -= SimdStyle::vector_element_count();
      data_reg_t vals_r = tsl::loadu<SimdStyle>(&data[right]);
      idx_arr_t<SimdStyle, IndexStyle> idx_r = load_idx_arr(&indexes[right]);

      // Preload second register
      right -= SimdStyle::vector_element_count();
      data_reg_t vals_r_adv = tsl::loadu<SimdStyle>(&data[right]);
      idx_arr_t<SimdStyle, IndexStyle> idx_r_adv = load_idx_arr(&indexes[right]);

      /* Working copies */
      data_reg_t vals;
      idx_arr_t<SimdStyle, IndexStyle> idxs;

      if (left == right) {
        do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, vals_l, idx_l,
                                                                           left_w, right_w);
        do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, vals_l_adv,
                                                                           idx_l_adv, left_w, right_w);
        do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, vals_r, idx_r,
                                                                           left_w, right_w);
        do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, vals_r_adv,
                                                                           idx_r_adv, left_w, right_w);
      } else if (left + SimdStyle::vector_element_count() > right) {
        const typename SimdStyle::imask_type valid_mask = ((1ull << (right - left)) - 1) & (-1ull >> 1);
        const data_reg_t remainder_val_vec = tsl::loadu<SimdStyle>(&data[left]);
        const idx_arr_t<SimdStyle, IndexStyle> remainder_idx_vec = load_idx_arr(&indexes[left]);

        do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, vals_l, idx_l,
                                                                           left_w, right_w);
        do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, vals_l_adv,
                                                                           idx_l_adv, left_w, right_w);
        do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, vals_r, idx_r,
                                                                           left_w, right_w);
        do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, vals_r_adv,
                                                                           idx_r_adv, left_w, right_w);

        /* Remainder */
        do_avx_sort_masked<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
          data, indexes, pivot_vec, remainder_val_vec, remainder_idx_vec, left_w, right_w, valid_mask);
      } else {
        while (left + SimdStyle::vector_element_count() <= right) {
          const ssize_t left_con = (left - left_w);
          const ssize_t right_con = (right_w - right);
          if (left_con <= right_con) {
            vals = vals_l;
            idxs = idx_l;

            vals_l = vals_l_adv;
            idx_l = idx_l_adv;

            vals_l_adv = tsl::loadu<SimdStyle>(&data[left]);
            idx_l_adv = load_idx_arr(&indexes[left]);

            left += SimdStyle::vector_element_count();
          } else {
            vals = vals_r;
            idxs = idx_r;

            vals_r = vals_r_adv;
            idx_r = idx_r_adv;

            right -= SimdStyle::vector_element_count();
            vals_r_adv = tsl::loadu<SimdStyle>(&data[right]);
            idx_r_adv = load_idx_arr(&indexes[right]);
          }

          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, vals, idxs,
                                                                             left_w, right_w);
        }

        /* -- Cleanup Phase -- */
        if ((left < right) && (left + SimdStyle::vector_element_count() > right)) {
          const typename SimdStyle::imask_type valid_mask = ((1ull << (right - left)) - 1) & (-1ull >> 1);
          const data_reg_t remainder_val_vec = tsl::loadu<SimdStyle>(&data[left]);
          const idx_arr_t<SimdStyle, IndexStyle> remainder_idx_vec = load_idx_arr(&indexes[left]);

          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, vals_l, idx_l,
                                                                             left_w, right_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, vals_l_adv,
                                                                             idx_l_adv, left_w, right_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, vals_r, idx_r,
                                                                             left_w, right_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, vals_r_adv,
                                                                             idx_r_adv, left_w, right_w);

          /* Remainder */
          do_avx_sort_masked<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
            data, indexes, pivot_vec, remainder_val_vec, remainder_idx_vec, left_w, right_w, valid_mask);
        } else {
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, vals_l, idx_l,
                                                                             left_w, right_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, vals_l_adv,
                                                                             idx_l_adv, left_w, right_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, vals_r, idx_r,
                                                                             left_w, right_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, vals_r_adv,
                                                                             idx_r_adv, left_w, right_w);
        }
      }

      /* Start partitioning the right side into [ values == PIVOT | values > Pivot ] */
      ssize_t pivot_l = left_w;
      ssize_t pivot_l_w = left_w;

      ssize_t pivot_r = right_start;
      ssize_t pivot_r_w = right_start;
      if ((pivot_r - pivot_l) < 4 * SimdStyle::vector_element_count()) {
        /* Data too small to fit in 4 registers */
        insertion_sort_fallback<SortOrderT>(data, indexes, pivot_l, pivot_r);

        for (size_t i = pivot_l; i < right_start; ++i) {
          if (data[i] > pivot) {
            pivot_r_w = i;
            break;
          }
        }
      } else {
        vals_l = tsl::loadu<SimdStyle>(&data[pivot_l]);
        idx_l = load_idx_arr(&indexes[pivot_l]);
        pivot_l += SimdStyle::vector_element_count();

        vals_l_adv = tsl::loadu<SimdStyle>(&data[pivot_l]);
        idx_l_adv = load_idx_arr(&indexes[pivot_l]);
        pivot_l += SimdStyle::vector_element_count();

        pivot_r -= SimdStyle::vector_element_count();
        vals_r = tsl::loadu<SimdStyle>(&data[pivot_r]);
        idx_r = load_idx_arr(&indexes[pivot_r]);

        pivot_r -= SimdStyle::vector_element_count();
        vals_r_adv = tsl::loadu<SimdStyle>(&data[pivot_r]);
        idx_r_adv = load_idx_arr(&indexes[pivot_r]);

        if (pivot_l == pivot_r) {
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, vals_l, idx_l,
                                                                             pivot_l_w, pivot_r_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, vals_l_adv,
                                                                             idx_l_adv, pivot_l_w, pivot_r_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, vals_r, idx_r,
                                                                             pivot_l_w, pivot_r_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, vals_r_adv,
                                                                             idx_r_adv, pivot_l_w, pivot_r_w);
        } else if (pivot_l + SimdStyle::vector_element_count() > pivot_r) {
          const typename SimdStyle::imask_type valid_mask = ((1ull << (pivot_r - pivot_l)) - 1) & (-1ull >> 1);
          const data_reg_t remainder_val_vec = tsl::loadu<SimdStyle>(&data[pivot_l]);
          const idx_arr_t<SimdStyle, IndexStyle> remainder_idx_vec = load_idx_arr(&indexes[pivot_l]);

          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, vals_l, idx_l,
                                                                             pivot_l_w, pivot_r_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, vals_l_adv,
                                                                             idx_l_adv, pivot_l_w, pivot_r_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, vals_r, idx_r,
                                                                             pivot_l_w, pivot_r_w);
          do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, vals_r_adv,
                                                                             idx_r_adv, pivot_l_w, pivot_r_w);

          /* Remainder */
          do_avx_sort_masked<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
            data, indexes, pivot_vec, remainder_val_vec, remainder_idx_vec, pivot_l_w, pivot_r_w, valid_mask);
        } else {
          while (pivot_l + SimdStyle::vector_element_count() <= pivot_r) {
            const ssize_t left_con = (pivot_l - pivot_l_w);
            const ssize_t right_con = (pivot_r_w - pivot_r);
            if (left_con <= right_con) {
              vals = vals_l;
              idxs = idx_l;

              vals_l = vals_l_adv;
              idx_l = idx_l_adv;

              vals_l_adv = tsl::loadu<SimdStyle>(&data[pivot_l]);
              idx_l_adv = load_idx_arr(&indexes[pivot_l]);

              pivot_l += SimdStyle::vector_element_count();
            } else {
              vals = vals_r;
              idxs = idx_r;

              vals_r = vals_r_adv;
              idx_r = idx_r_adv;

              pivot_r -= SimdStyle::vector_element_count();
              vals_r_adv = tsl::loadu<SimdStyle>(&data[pivot_r]);
              idx_r_adv = load_idx_arr(&indexes[pivot_r]);
            }

            do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, vals, idxs,
                                                                               pivot_l_w, pivot_r_w);
          }

          /* -- Cleanup Phase -- */
          if ((pivot_l < pivot_r) && (pivot_l + SimdStyle::vector_element_count() > pivot_r)) {
            const typename SimdStyle::imask_type valid_mask = ((1ull << (pivot_r - pivot_l)) - 1) & (-1ull >> 1);
            const data_reg_t remainder_val_vec = tsl::loadu<SimdStyle>(&data[pivot_l]);
            const idx_arr_t<SimdStyle, IndexStyle> remainder_idx_vec = load_idx_arr(&indexes[pivot_l]);

            do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, vals_l, idx_l,
                                                                               pivot_l_w, pivot_r_w);
            do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, vals_l_adv,
                                                                               idx_l_adv, pivot_l_w, pivot_r_w);
            do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, vals_r, idx_r,
                                                                               pivot_l_w, pivot_r_w);
            do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, vals_r_adv,
                                                                               idx_r_adv, pivot_l_w, pivot_r_w);

            /* Remainder */
            do_avx_sort_masked<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
              data, indexes, pivot_vec, remainder_val_vec, remainder_idx_vec, pivot_l_w, pivot_r_w, valid_mask);
          } else {
            do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, vals_l, idx_l,
                                                                               pivot_l_w, pivot_r_w);
            do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, vals_l_adv,
                                                                               idx_l_adv, pivot_l_w, pivot_r_w);
            do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, vals_r, idx_r,
                                                                               pivot_l_w, pivot_r_w);
            do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec, vals_r_adv,
                                                                               idx_r_adv, pivot_l_w, pivot_r_w);
          }
        }
      }

      /* -- Left Side -- */
      const auto pivot_ls = get_pivot(data, left_start, left_w);
      if ((left_w - left_start) < (4 * SimdStyle::vector_element_count())) {
        insertion_sort_fallback<SortOrderT>(data, indexes, left_start, left_w);
      } else {
        partition<SimdStyle, IndexStyle, SortOrderT>(data, indexes, left_start, left_w, pivot_ls);
      }

      /* -- Right Side -- */
      const auto pivot_rs = get_pivot(data, pivot_r_w, right_start);
      if ((right_start - pivot_r_w) < (4 * SimdStyle::vector_element_count())) {
        insertion_sort_fallback<SortOrderT>(data, indexes, right_w, right_start);
      } else {
        partition<SimdStyle, IndexStyle, SortOrderT>(data, indexes, pivot_r_w, right_start, pivot_rs);
      }
    }

    template <class SimdStyle, class IndexStyle, TSL_SORT_ORDER SortOrderT, typename T = typename SimdStyle::base_type,
              typename U = typename IndexStyle::base_type>
    void tsl_sort(T* data, U* indexes, ssize_t left, ssize_t right, T pivot) {
      if ((right - left) < (4 * SimdStyle::vector_element_count())) {
        insertion_sort_fallback<SortOrderT>(data, indexes, left, right);
        return;
      }
      partition<SimdStyle, IndexStyle, SortOrderT>(data, indexes, left, right, pivot);
    }
  }  // namespace sort_indirect_copy_basedata
}  // namespace tuddbs

#endif