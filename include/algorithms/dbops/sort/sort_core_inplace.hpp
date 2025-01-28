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
    /**
     * @brief Detect contiguous runs of the same value in a given range. Basically, a run-length encoding, stored as a
     * Cluster struct, but maybe out of order.
     *
     * @details The inplace variant sorts the index array indirectly by sorting the data array first and applying this
     * sort permutation to the index as well. Thus, we do not need to leverage indirect access through indexes into data
     * at this point. The parameter is here for future compatibility.
     *
     * @param clusters A deque to pull and push runs from and to
     * @param data The data column
     * @param indexes The to-be-sorted column containing positions that point to values in data.
     * @param left The leftmost index to sort in indexes
     * @param right The rightmost index to sort in indexes
     */
    template <typename T, typename U>
    void detect_cluster(std::deque<Cluster>& clusters, T* data, [[maybe_unused]] U* indexes, size_t left,
                        size_t right) {
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

    /**
     * @brief The fallback sort implementation, whenever there is not enough data to leverage SIMDified
     * sorting.
     *
     * @tparam order Ascending or Descending sort
     * @tparam T Datatype of the column data
     * @tparam U Datatype of the position list contained in indexes
     * @param data The data column
     * @param indexes The to-be-sorted column containing positions that point to values in data.
     * @param left_boundary The leftmost index to sort in indexes
     * @param right_boundary The rightmost index to sort in indexes
     */
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

    /**
     * @brief A templated helper to get the size of the index array helper as compile time constant.
     *
     * @tparam S The SIMD processing style for the data column
     * @tparam I The SIMD processing style for the index column
     */
    template <class S, class I>
    constexpr size_t idx_arr_len = sizeof(typename I::base_type) / sizeof(typename S::base_type);

    /**
     * @brief A templated helper to get the bit count in an index simd register lane
     *
     * @tparam S The SIMD processing style for the data column
     * @tparam I The SIMD processing style for the index column
     */
    template <class S, class I>
    constexpr size_t bits_per_idx_register =
      (I::vector_element_count() == 64) ? -1ull : (1ull << I::vector_element_count()) - 1;

    /**
     * @brief Stores the valid entries of an index register but takes differences between the data types of data and
     * indexes into account.
     *
     * @tparam SimdStyle  S The SIMD processing style for the data column
     * @tparam IndexStyle I The SIMD processing style for the index column
     * @tparam IndexStyle::base_type The data type of the positions in indexes
     * @param full_mask The complete mask from the pivot comparison.
     * @param indexes The position list to store
     * @param idx_tmparr The loaded index positions, not a SIMD register but an array helper.
     */
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
         * We always 'just' advance the index pointer by the amount of written elements, both for left and right
         * side. That is, for left we always start at the 'first' write position. For right, we position the indexes
         * pointer at the leftmost position, since we know nb_high from the full_mask.
         */
        indexes += tsl::mask_population_count<IndexStyle>(idx_mask);
      }
    }

    /**
     * @brief Compares a data SIMD register against a pivot element and stores the results accordingly.
     *
     * @details Elements from the value register are compared against the pivot register for the less_than relation.
     * Every element truly smaller than pivot will be stored using a compress intrinsic to the left side of the result,
     * according to the left write pointer. For writing all greater_equal elements to the right side, we first must
     * decrement the write pointer because we cannot write "backwards".
     *
     * @tparam SimdStyle  S The SIMD processing style for the data column
     * @tparam IndexStyle I The SIMD processing style for the index column
     * @tparam type
     * @tparam order
     * @tparam SimdStyle::base_type
     * @tparam IndexStyle::base_type
     * @param data The data column, used to indirectly sort indexes.
     * @param indexes The to-be-sorted column containing positions, that point to values in data.
     * @param pivot_reg A SIMD register, that contains the current pivot element.
     * @param val_reg A SIMD register, corresponding to SimdStyle, that contains a set of elements from data.
     * @param idx_tmparr An std::array helper, that contains positions from indexes, that correspond to val_reg.
     * @param l_w The left side write pointer.
     * @param r_w The right side write pointer.
     */
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

    /**
     * @brief A templated helper function for masking out invalid elements. Behaves the same as do_tsl_sort() otherwise.
     *
     * @param valid A bitmask indicating the valid elements in val_reg.
     */
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

    /**
     * @brief Partitions the data and performs a recursive sort on the left and right sides.
     *
     * @details
     * The SortStateT is used to determine the final step of the partitioning function.
     * If SortStateT is of type DefaultSortState, the function will recursively call itself on the left and right side.
     * Otherwise, the function will detect clusters on the left and right side. Those clusters will be stored in the
     * ClusteredSortState. The clusters are necessary for the multi-column sort. If SortStateT is of type
     * TailClusteredSortState, the function will return a ClusteredRange, which contains the start and end index of the
     * cluster.
     *
     * @tparam SimdStyle
     * @tparam IndexStyle
     * @tparam SortOrderT
     * @tparam SimdStyle::base_type
     * @tparam IndexStyle::base_type
     * @tparam SortStateT
     * @param state
     * @param data
     * @param indexes
     * @param left
     * @param right
     * @param pivot
     * @param level
     * @return std::conditional_t<std::is_same_v<SortStateT, TailClusteredSortState>, ClusteredRange, void>
     */
    template <class SimdStyle, class IndexStyle, TSL_SORT_ORDER SortOrderT, typename T = typename SimdStyle::base_type,
              typename U = typename IndexStyle::base_type, class SortStateT = DefaultSortState>
    auto partition(SortStateT& state, T* data, U* indexes, ssize_t left, ssize_t right, T pivot, size_t level = 0)
      -> std::conditional_t<std::is_same_v<SortStateT, TailClusteredSortState>, ClusteredRange, void> {
      static_assert(sizeof(T) <= sizeof(U), "The index type (U) must be at least as wide as the data type (T).");

      using data_reg_t = typename SimdStyle::register_type;

      /* For now we assume that IndexStyle will use a wider or equally sized type than SimdStyle for the underlying data
       */
      const auto load_idx_arr = [](U const* memory) -> idx_arr_t<SimdStyle, IndexStyle> {
        idx_arr_t<SimdStyle, IndexStyle> idxs;
#pragma unroll(sort_inplace::idx_arr_len<SimdStyle, IndexStyle>)
        for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
          idxs[i] = memory[i];
        }
        return idxs;
      };

      const ssize_t left_start = left;
      const ssize_t right_start = right;

      // Broadcast the pivot element to a SIMD register
      const auto pivot_vec = tsl::set1<SimdStyle>(pivot);
      ssize_t left_w = left;
      ssize_t right_w = right;

      /* Load data and Index from left side */
      // First register
      data_reg_t vals_l = tsl::loadu<SimdStyle>(&data[left]);
      idx_arr_t<SimdStyle, IndexStyle> idx_l = load_idx_arr(&indexes[left]);
      left += SimdStyle::vector_element_count();

      // Preload second register. We must do that, because we need to preserve data in order to avoid accidently
      // overwriting elements, which have not been sorted yet.
      data_reg_t vals_l_adv = tsl::loadu<SimdStyle>(&data[left]);
      idx_arr_t<SimdStyle, IndexStyle> idx_l_adv = load_idx_arr(&indexes[left]);
      left += SimdStyle::vector_element_count();

      /* Load data and Index from right side */
      // First register
      right -= SimdStyle::vector_element_count();
      data_reg_t vals_r = tsl::loadu<SimdStyle>(&data[right]);
      idx_arr_t<SimdStyle, IndexStyle> idx_r = load_idx_arr(&indexes[right]);

      // Preload second register. We must do that, because we need to preserve data in order to avoid accidently
      // overwriting elements, which have not been sorted yet.
      right -= SimdStyle::vector_element_count();
      data_reg_t vals_r_adv = tsl::loadu<SimdStyle>(&data[right]);
      idx_arr_t<SimdStyle, IndexStyle> idx_r_adv = load_idx_arr(&indexes[right]);

      /* Working copies */
      data_reg_t vals;
      idx_arr_t<SimdStyle, IndexStyle> idxs;

      // We have exactly 4 Registers to sort and already preloaded them. Apply the sorting to all and be done with it.
      if (left == right) {
        sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
          data, indexes, pivot_vec, vals_l, idx_l, left_w, right_w);
        sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
          data, indexes, pivot_vec, vals_l_adv, idx_l_adv, left_w, right_w);
        sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
          data, indexes, pivot_vec, vals_r, idx_r, left_w, right_w);
        sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
          data, indexes, pivot_vec, vals_r_adv, idx_r_adv, left_w, right_w);
      } else if (left + SimdStyle::vector_element_count() > right) {
        // We have 4 preloaded registers, but some leftover elements, which do not fill up an entire register. Preserve
        // the remainder, apply the sort to the preloaded elements. The preloaded remainder register contains elements,
        // that were already preloaded before and must thus be masked as invalid.
        const typename SimdStyle::imask_type valid_mask = ((1ull << (right - left)) - 1) & (-1ull >> 1);
        const data_reg_t remainder_val_vec = tsl::loadu<SimdStyle>(&data[left]);
        const idx_arr_t<SimdStyle, IndexStyle> remainder_idx_vec = load_idx_arr(&indexes[left]);

        sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
          data, indexes, pivot_vec, vals_l, idx_l, left_w, right_w);
        sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
          data, indexes, pivot_vec, vals_l_adv, idx_l_adv, left_w, right_w);
        sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
          data, indexes, pivot_vec, vals_r, idx_r, left_w, right_w);
        sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
          data, indexes, pivot_vec, vals_r_adv, idx_r_adv, left_w, right_w);

        /* Remainder */
        sort_inplace::do_tsl_sort_masked<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
          data, indexes, pivot_vec, remainder_val_vec, remainder_idx_vec, left_w, right_w, valid_mask);
      } else {
        // We preloaded 4 registers and there are more elements left than a single SIMD register can hold. After sorting
        // a register, we check if we have to read from the left or right side of the array, depending on how many
        // elements were written to either side. This part is iteratively repeated, until one of the previous corner
        // cases appears during processing.
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

          sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec,
                                                                                           vals, idxs, left_w, right_w);
        }

        /* -- Cleanup Phase -- */
        // The two corner cases now can only be either exactly 4 registers are left to process or 4 registers and some
        // remainder.
        if ((left < right) && (left + SimdStyle::vector_element_count() > right)) {
          const typename SimdStyle::imask_type valid_mask = ((1ull << (right - left)) - 1) & (-1ull >> 1);
          const data_reg_t remainder_val_vec = tsl::loadu<SimdStyle>(&data[left]);
          const idx_arr_t<SimdStyle, IndexStyle> remainder_idx_vec = load_idx_arr(&indexes[left]);

          sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
            data, indexes, pivot_vec, vals_l, idx_l, left_w, right_w);
          sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
            data, indexes, pivot_vec, vals_l_adv, idx_l_adv, left_w, right_w);
          sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
            data, indexes, pivot_vec, vals_r, idx_r, left_w, right_w);
          sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
            data, indexes, pivot_vec, vals_r_adv, idx_r_adv, left_w, right_w);

          /* Remainder */
          sort_inplace::do_tsl_sort_masked<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
            data, indexes, pivot_vec, remainder_val_vec, remainder_idx_vec, left_w, right_w, valid_mask);
        } else {
          sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
            data, indexes, pivot_vec, vals_l, idx_l, left_w, right_w);
          sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
            data, indexes, pivot_vec, vals_l_adv, idx_l_adv, left_w, right_w);
          sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
            data, indexes, pivot_vec, vals_r, idx_r, left_w, right_w);
          sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
            data, indexes, pivot_vec, vals_r_adv, idx_r_adv, left_w, right_w);
        }
      }

      // Contrary to direct sort, we must preserve the indexes of all pivot elements and thus re-sort the right side of
      // the partition. The algorithm is analogue to the first part, but now leverages a greater_than comparator for
      // compare().

      /* Start partitioning the right side into [ values == PIVOT | values > Pivot ] */
      ssize_t pivot_l = left_w;
      ssize_t pivot_l_w = left_w;

      ssize_t pivot_r = right_start;
      ssize_t pivot_r_w = right_start;
      if ((pivot_r - pivot_l) < 4 * SimdStyle::vector_element_count()) {
        /* Data too small to fit in 4 registers */
        sort_inplace::insertion_sort_fallback<SortOrderT>(data, indexes, pivot_l, pivot_r);

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
          sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
            data, indexes, pivot_vec, vals_l, idx_l, pivot_l_w, pivot_r_w);
          sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
            data, indexes, pivot_vec, vals_l_adv, idx_l_adv, pivot_l_w, pivot_r_w);
          sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
            data, indexes, pivot_vec, vals_r, idx_r, pivot_l_w, pivot_r_w);
          sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
            data, indexes, pivot_vec, vals_r_adv, idx_r_adv, pivot_l_w, pivot_r_w);
        } else if (pivot_l + SimdStyle::vector_element_count() > pivot_r) {
          const typename SimdStyle::imask_type valid_mask = ((1ull << (pivot_r - pivot_l)) - 1) & (-1ull >> 1);
          const data_reg_t remainder_val_vec = tsl::loadu<SimdStyle>(&data[pivot_l]);
          const idx_arr_t<SimdStyle, IndexStyle> remainder_idx_vec = load_idx_arr(&indexes[pivot_l]);

          sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
            data, indexes, pivot_vec, vals_l, idx_l, pivot_l_w, pivot_r_w);
          sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
            data, indexes, pivot_vec, vals_l_adv, idx_l_adv, pivot_l_w, pivot_r_w);
          sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
            data, indexes, pivot_vec, vals_r, idx_r, pivot_l_w, pivot_r_w);
          sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
            data, indexes, pivot_vec, vals_r_adv, idx_r_adv, pivot_l_w, pivot_r_w);

          /* Remainder */
          sort_inplace::do_tsl_sort_masked<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
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

            sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
              data, indexes, pivot_vec, vals, idxs, pivot_l_w, pivot_r_w);
          }

          /* -- Cleanup Phase -- */
          if ((pivot_l < pivot_r) && (pivot_l + SimdStyle::vector_element_count() > pivot_r)) {
            const typename SimdStyle::imask_type valid_mask = ((1ull << (pivot_r - pivot_l)) - 1) & (-1ull >> 1);
            const data_reg_t remainder_val_vec = tsl::loadu<SimdStyle>(&data[pivot_l]);
            const idx_arr_t<SimdStyle, IndexStyle> remainder_idx_vec = load_idx_arr(&indexes[pivot_l]);

            sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
              data, indexes, pivot_vec, vals_l, idx_l, pivot_l_w, pivot_r_w);
            sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
              data, indexes, pivot_vec, vals_l_adv, idx_l_adv, pivot_l_w, pivot_r_w);
            sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
              data, indexes, pivot_vec, vals_r, idx_r, pivot_l_w, pivot_r_w);
            sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
              data, indexes, pivot_vec, vals_r_adv, idx_r_adv, pivot_l_w, pivot_r_w);

            /* Remainder */
            sort_inplace::do_tsl_sort_masked<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
              data, indexes, pivot_vec, remainder_val_vec, remainder_idx_vec, pivot_l_w, pivot_r_w, valid_mask);
          } else {
            sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
              data, indexes, pivot_vec, vals_l, idx_l, pivot_l_w, pivot_r_w);
            sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
              data, indexes, pivot_vec, vals_l_adv, idx_l_adv, pivot_l_w, pivot_r_w);
            sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
              data, indexes, pivot_vec, vals_r, idx_r, pivot_l_w, pivot_r_w);
            sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
              data, indexes, pivot_vec, vals_r_adv, idx_r_adv, pivot_l_w, pivot_r_w);
          }
        }
      }

      // To avoid copy-pasting the complete partition code for all in-place cluster detection variants, we use a compile
      // time switch to detect plain sorting, tail or leaf clustering.
      if constexpr (std::is_same_v<SortStateT, DefaultSortState>) {
        /* -- Left Side -- */
        if ((left_w - left_start) < (4 * SimdStyle::vector_element_count())) {
          sort_inplace::insertion_sort_fallback<SortOrderT>(data, indexes, left_start, left_w);
        } else {
          const auto pivot_ls = get_pivot(data, left_start, left_w);
          partition<SimdStyle, IndexStyle, SortOrderT>(state, data, indexes, left_start, left_w, pivot_ls);
        }

        /* -- Right Side -- */
        if ((right_start - pivot_r_w) < (4 * SimdStyle::vector_element_count())) {
          sort_inplace::insertion_sort_fallback<SortOrderT>(data, indexes, right_w, right_start);
        } else {
          const auto pivot_rs = get_pivot(data, pivot_r_w, right_start);
          partition<SimdStyle, IndexStyle, SortOrderT>(state, data, indexes, pivot_r_w, right_start, pivot_rs);
        }
      } else if constexpr (std::is_same_v<SortStateT, LeafClusteredSortState>) {
        /* -- Left Side -- */
        if ((left_w - left_start) < (4 * SimdStyle::vector_element_count())) {
          sort_inplace::insertion_sort_fallback<SortOrderT>(data, indexes, left_start, left_w);
          sort_inplace::detect_cluster(state.clusters, data, indexes, left_start, left_w);
        } else {
          const auto pivot_ls = get_pivot(data, left_start, left_w);
          partition<SimdStyle, IndexStyle, SortOrderT>(state, data, indexes, left_start, left_w, pivot_ls);
        }

        /* -- Right Side -- */
        if ((right_start - pivot_r_w) < (4 * SimdStyle::vector_element_count())) {
          sort_inplace::insertion_sort_fallback<SortOrderT>(data, indexes, right_w, right_start);
          sort_inplace::detect_cluster(state.clusters, data, indexes, left_w, right_start);
        } else {
          const auto pivot_rs = get_pivot(data, pivot_r_w, right_start);
          sort_inplace::detect_cluster(state.clusters, data, indexes, left_w, pivot_r_w);
          partition<SimdStyle, IndexStyle, SortOrderT>(state, data, indexes, pivot_r_w, right_start, pivot_rs);
        }
      } else if constexpr (std::is_same_v<SortStateT, TailClusteredSortState>) {
        ClusteredRange left_range, right_range;
        bool left_leaf = false;
        bool right_leaf = false;

        /* -- Left Side -- */
        if ((left_w - left_start) < (4 * SimdStyle::vector_element_count())) {
          sort_inplace::insertion_sort_fallback<SortOrderT>(data, indexes, left_start, left_w);
          left_range = ClusteredRange{static_cast<size_t>(left_start), static_cast<size_t>(left_w)};
          left_leaf = true;
        } else {
          const auto pivot_ls = get_pivot(data, left_start, left_w);
          left_range =
            partition<SimdStyle, IndexStyle, SortOrderT>(state, data, indexes, left_start, left_w, pivot_ls, level + 1);
        }

        /* -- Right Side -- */
        if ((right_start - pivot_r_w) < (4 * SimdStyle::vector_element_count())) {
          sort_inplace::insertion_sort_fallback<SortOrderT>(data, indexes, right_w, right_start);
          right_range = ClusteredRange{static_cast<size_t>(right_w), static_cast<size_t>(right_start)};
          right_leaf = true;
        } else {
          const auto pivot_rs = get_pivot(data, pivot_r_w, right_start);
          right_range = partition<SimdStyle, IndexStyle, SortOrderT>(state, data, indexes, pivot_r_w, right_start,
                                                                     pivot_rs, level + 1);
        }

        if (left_leaf) {
          if (right_leaf) {
            sort_inplace::detect_cluster(state.clusters, data, indexes, left_start, right_start);
          } else {
            sort_inplace::detect_cluster(state.clusters, data, indexes, left_start, right_range.start);
          }
        } else {
          if (right_leaf) {
            sort_inplace::detect_cluster(state.clusters, data, indexes, left_range.end, right_start);
          } else {
            sort_inplace::detect_cluster(state.clusters, data, indexes, left_range.end, right_range.start);
          }
        }
        return ClusteredRange{static_cast<size_t>(left_start), static_cast<size_t>(right_start)};
      } else {
        static_assert(!std::is_same_v<SortStateT, SortStateT>,
                      "SortStateT must be of type (DefaultSortSTate, LeafClusteredSortState, TailClusteredSortState)");
      }
    }
  }  // namespace sort_inplace
}  // namespace tuddbs
#endif
