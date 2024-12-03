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
 * @file sort_indirect_inplace_cluster_leaf.hpp
 * @brief
 */

#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_INDIRECT_INPLACE_CLUSTER_LEAF_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_INDIRECT_INPLACE_CLUSTER_LEAF_HPP

#include <climits>
#include <cstddef>
#include <deque>
#include <iterable.hpp>
#include <tuple>
#include <type_traits>

#include "algorithms/dbops/dbops_hints.hpp"
#include "algorithms/dbops/sort/sort_core_inplace.hpp"
#include "algorithms/dbops/sort/sort_utils.hpp"
#include "algorithms/utils/hinting.hpp"
#include "algorithms/utils/sorthints.hpp"
#include "tsl.hpp"

namespace tuddbs {
  template <tsl::VectorProcessingStyle _SimdStyle, tsl::VectorProcessingStyle _IndexStyle, TSL_SORT_ORDER SortOrderT,
            class HintSet = OperatorHintSet<hints::sort::indirect_inplace>>
  class LeafClusteringSingleColumnSortIndirectInplace {
    static_assert(has_hints_mutual_excluding<HintSet, std::tuple<hints::sort::indirect_inplace>,
                                             std::tuple<hints::sort::indirect_gather>>,
                  "Indirect sort can only be either inplace or gather, but both were given");
    static_assert(has_hints_mutual_excluding<HintSet, std::tuple<hints::sort::leaf_clustering>,
                                             std::tuple<hints::sort::tail_clustering>>,
                  "Trying to instantiate leaf clustering, but tail clustering is also given or tail hint is missing.");

   public:
    using SimdStyle = _SimdStyle;
    using IndexStyle = _IndexStyle;
    using DataT = SimdStyle::base_type;
    using IdxT = IndexStyle::base_type;

   private:
    DataT* m_data;
    IdxT* m_idx;
    std::deque<tuddbs::Cluster> clusters;

   public:
    explicit LeafClusteringSingleColumnSortIndirectInplace(SimdOpsIterable auto p_data, SimdOpsIterable auto p_idx)
      : m_data{p_data}, m_idx{p_idx}, clusters{} {}

    auto operator()(const size_t left, const size_t right) {
      if ((right - left) < (4 * SimdStyle::vector_element_count())) {
        sort_inplace::insertion_sort_fallback<SortOrderT>(m_data, m_idx, left, right);
        sort_inplace::detect_cluster(clusters, m_data, m_idx, left, right);
        return;
      }

      const DataT pivot = tuddbs::get_pivot_indirect(m_data, m_idx, left, right - 1);
      partition<SimdStyle, IndexStyle>(clusters, m_data, m_idx, left, right, pivot);
    }

    std::deque<tuddbs::Cluster> getClusters() const { return clusters; }

   private:
    template <class SimdStyle, class IndexStyle, typename T = typename SimdStyle::base_type,
              typename U = typename IndexStyle::base_type>
    void partition(std::deque<Cluster>& cluster, T* data, U* indexes, ssize_t left, ssize_t right, T pivot) {
      static_assert(sizeof(T) <= sizeof(U), "The index type (U) must be at least as wide as the data type (T).");
      using data_reg_t = typename SimdStyle::register_type;

      /* For now we assume that IndexStyle will use a wider or equally sized type than SimdStyle for the underlying data
       */
      const auto load_idx_arr = [](U const* memory) -> idx_arr_t<SimdStyle, IndexStyle> {
        idx_arr_t<SimdStyle, IndexStyle> idxs;
#pragma unroll(SimdStyle::vector_element_count())
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
        sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
          data, indexes, pivot_vec, vals_l, idx_l, left_w, right_w);
        sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
          data, indexes, pivot_vec, vals_l_adv, idx_l_adv, left_w, right_w);
        sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
          data, indexes, pivot_vec, vals_r, idx_r, left_w, right_w);
        sort_inplace::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
          data, indexes, pivot_vec, vals_r_adv, idx_r_adv, left_w, right_w);
      } else if (left + SimdStyle::vector_element_count() > right) {
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

      /* -- Left Side -- */
      if ((left_w - left_start) < (4 * SimdStyle::vector_element_count())) {
        sort_inplace::insertion_sort_fallback<SortOrderT>(data, indexes, left_start, left_w);
        sort_inplace::detect_cluster(cluster, data, indexes, left_start, left_w);
      } else {
        const auto pivot_ls = get_pivot(data, left_start, left_w);
        partition<SimdStyle, IndexStyle>(cluster, data, indexes, left_start, left_w, pivot_ls);
      }

      /* -- Right Side -- */
      if ((right_start - pivot_r_w) < (4 * SimdStyle::vector_element_count())) {
        sort_inplace::insertion_sort_fallback<SortOrderT>(data, indexes, right_w, right_start);
        sort_inplace::detect_cluster(cluster, data, indexes, left_w, right_start);
      } else {
        const auto pivot_rs = get_pivot(data, pivot_r_w, right_start);
        sort_inplace::detect_cluster(cluster, data, indexes, left_w, pivot_r_w);
        partition<SimdStyle, IndexStyle>(cluster, data, indexes, pivot_r_w, right_start, pivot_rs);
      }
    }
  };
};  // namespace tuddbs

#endif