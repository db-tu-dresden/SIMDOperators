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
 * @file sort_indirect_gather_cluster_tail.hpp
 * @brief
 */

#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_INDIRECT_GATHER_CLUSTER_TAIL_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_INDIRECT_GATHER_CLUSTER_TAIL_HPP

#include <climits>
#include <cstddef>
#include <deque>
#include <iterable.hpp>
#include <tuple>
#include <type_traits>

#include "algorithms/dbops/dbops_hints.hpp"
#include "algorithms/dbops/sort/sort_core_gather.hpp"
#include "algorithms/dbops/sort/sort_utils.hpp"
#include "algorithms/utils/hinting.hpp"
#include "algorithms/utils/sorthints.hpp"
#include "tsl.hpp"

namespace tuddbs {
  template <tsl::VectorProcessingStyle _SimdStyle, tsl::VectorProcessingStyle _IndexStyle, TSL_SORT_ORDER SortOrderT,
            class HintSet = OperatorHintSet<hints::sort::indirect_inplace>>
  class TailClusteringSingleColumnSortIndirectGather {
    static_assert(has_hints_mutual_excluding<HintSet, std::tuple<hints::sort::indirect_gather>,
                                             std::tuple<hints::sort::indirect_inplace>>,
                  "Indirect sort can only be either gather or inplace, but both were given");
    static_assert(has_hints_mutual_excluding<HintSet, std::tuple<hints::sort::tail_clustering>,
                                             std::tuple<hints::sort::leaf_clustering>>,
                  "Trying to instantiate tail clustering, but leaf clustering is also given or tail hint is missing.");

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
    explicit TailClusteringSingleColumnSortIndirectGather(SimdOpsIterable auto p_data, SimdOpsIterable auto p_idx)
      : m_data{p_data}, m_idx{p_idx}, clusters{} {}

    auto operator()(const size_t left, const size_t right) {
      if ((right - left) < (4 * SimdStyle::vector_element_count())) {
        gather_sort::insertion_sort_fallback<SortOrderT>(m_data, m_idx, left, right);
        gather_sort::detect_cluster(clusters, m_data, m_idx, left, right);
        return;
      }

      const DataT pivot = tuddbs::get_pivot_indirect(m_data, m_idx, left, right - 1);
      static_cast<void>(partition<SimdStyle, IndexStyle>(clusters, m_data, m_idx, left, right, pivot));
    }

    std::deque<tuddbs::Cluster> & getClusters() { return clusters; }

   private:
    template <class SimdStyle, class IndexStyle, typename T = typename SimdStyle::base_type,
              typename U = typename IndexStyle::base_type>
    ClusteredRange partition(std::deque<Cluster>& cluster, T const* __restrict__ data, U* __restrict__ indexes,
                             size_t left, size_t right, T pivot) {
      static_assert(sizeof(T) <= sizeof(U), "The index type (U) must be at least as wide as the data type (T).");

      using idx_reg_t = typename IndexStyle::register_type;

      const size_t left_start = left;
      const size_t right_start = right;

      // Determine the data type for the pivot vector register depending on the DataType<>IndexType combination
      const auto pivot_vec = gather_sort::set_pivot<SimdStyle, IndexStyle>(pivot);
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
        gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, idx_l,
                                                                                        left_w, right_w);
        gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec,
                                                                                        idx_l_adv, left_w, right_w);
        gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, idx_r,
                                                                                        left_w, right_w);
        gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec,
                                                                                        idx_r_adv, left_w, right_w);
      } else if (left + IndexStyle::vector_element_count() > right) {
        // If there is less than a vector register remaining between left and right,
        // we load the remainder and mask out everything that we already loaded/processed in right

        const typename IndexStyle::imask_type valid_mask = ((1ull << (right - left)) - 1) & (-1ull >> 1);
        const idx_reg_t remainder_idx_vec = tsl::loadu<IndexStyle>(&indexes[left]);

        gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, idx_l,
                                                                                        left_w, right_w);
        gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec,
                                                                                        idx_l_adv, left_w, right_w);
        gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec, idx_r,
                                                                                        left_w, right_w);
        gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec,
                                                                                        idx_r_adv, left_w, right_w);

        /* Remainder */
        gather_sort::do_tsl_sort_masked<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
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

          gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec,
                                                                                          idxs, left_w, right_w);
        }

        /* -- Cleanup Phase -- */
        if ((left < right) && (left + IndexStyle::vector_element_count() > right)) {
          const typename IndexStyle::imask_type valid_mask = ((1ull << (right - left)) - 1) & (-1ull >> 1);
          const idx_reg_t remainder_idx_vec = tsl::loadu<IndexStyle>(&indexes[left]);

          gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec,
                                                                                          idx_l, left_w, right_w);
          gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec,
                                                                                          idx_l_adv, left_w, right_w);
          gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec,
                                                                                          idx_r, left_w, right_w);
          gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec,
                                                                                          idx_r_adv, left_w, right_w);

          /* Remainder */
          gather_sort::do_tsl_sort_masked<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(
            data, indexes, pivot_vec, remainder_idx_vec, left_w, right_w, valid_mask);
        } else {
          gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec,
                                                                                          idx_l, left_w, right_w);
          gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec,
                                                                                          idx_l_adv, left_w, right_w);
          gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec,
                                                                                          idx_r, left_w, right_w);
          gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_LT, SortOrderT>(data, indexes, pivot_vec,
                                                                                          idx_r_adv, left_w, right_w);
        }
      }

      /* Start partitioning the right side into [ values == PIVOT | values > Pivot ] */
      size_t pivot_l = left_w;
      size_t pivot_l_w = left_w;

      size_t pivot_r = right_start;
      size_t pivot_r_w = right_start;

      if ((pivot_r - pivot_l) < 4 * IndexStyle::vector_element_count()) {
        /* Data too small to fit in 4 registers */
        gather_sort::insertion_sort_fallback<SortOrderT>(data, indexes, pivot_l, pivot_r);

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
          gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec,
                                                                                          idx_l, pivot_l_w, pivot_r_w);
          gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
            data, indexes, pivot_vec, idx_l_adv, pivot_l_w, pivot_r_w);
          gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec,
                                                                                          idx_r, pivot_l_w, pivot_r_w);
          gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
            data, indexes, pivot_vec, idx_r_adv, pivot_l_w, pivot_r_w);
        } else if (pivot_l + IndexStyle::vector_element_count() > pivot_r) {
          const typename IndexStyle::imask_type valid_mask = ((1ull << (pivot_r - pivot_l)) - 1) & (-1ull >> 1);
          const idx_reg_t remainder_idx_vec = tsl::loadu<IndexStyle>(&indexes[pivot_l]);

          gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec,
                                                                                          idx_l, pivot_l_w, pivot_r_w);
          gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
            data, indexes, pivot_vec, idx_l_adv, pivot_l_w, pivot_r_w);
          gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec,
                                                                                          idx_r, pivot_l_w, pivot_r_w);
          gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
            data, indexes, pivot_vec, idx_r_adv, pivot_l_w, pivot_r_w);

          /* Remainder */
          gather_sort::do_tsl_sort_masked<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
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

            gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(data, indexes, pivot_vec,
                                                                                            idxs, pivot_l_w, pivot_r_w);
          }

          /* -- Cleanup Phase -- */
          if ((pivot_l < pivot_r) && (pivot_l + IndexStyle::vector_element_count() > pivot_r)) {
            const typename IndexStyle::imask_type valid_mask = ((1ull << (pivot_r - pivot_l)) - 1) & (-1ull >> 1);
            const idx_reg_t remainder_idx_vec = tsl::loadu<IndexStyle>(&indexes[pivot_l]);

            gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
              data, indexes, pivot_vec, idx_l, pivot_l_w, pivot_r_w);
            gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
              data, indexes, pivot_vec, idx_l_adv, pivot_l_w, pivot_r_w);
            gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
              data, indexes, pivot_vec, idx_r, pivot_l_w, pivot_r_w);
            gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
              data, indexes, pivot_vec, idx_r_adv, pivot_l_w, pivot_r_w);

            /* Remainder */
            gather_sort::do_tsl_sort_masked<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
              data, indexes, pivot_vec, remainder_idx_vec, pivot_l_w, pivot_r_w, valid_mask);
          } else {
            gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
              data, indexes, pivot_vec, idx_l, pivot_l_w, pivot_r_w);
            gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
              data, indexes, pivot_vec, idx_l_adv, pivot_l_w, pivot_r_w);
            gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
              data, indexes, pivot_vec, idx_r, pivot_l_w, pivot_r_w);
            gather_sort::do_tsl_sort<SimdStyle, IndexStyle, SORT_TYPE::SORT_EQ, SortOrderT>(
              data, indexes, pivot_vec, idx_r_adv, pivot_l_w, pivot_r_w);
          }
        }
      }

      ClusteredRange left_range, right_range;
      bool left_leaf = false;
      bool right_leaf = false;

      /* -- Left Side -- */
      if ((left_w - left_start) < (4 * IndexStyle::vector_element_count())) {
        gather_sort::insertion_sort_fallback<SortOrderT>(data, indexes, left_start, left_w);
        left_range = ClusteredRange{static_cast<size_t>(left_start), static_cast<size_t>(left_w)};
        left_leaf = true;
      } else {
        const auto pivot_ls = get_pivot_indirect(data, indexes, left_start, left_w - 1);
        left_range = partition<SimdStyle, IndexStyle>(cluster, data, indexes, left_start, left_w, pivot_ls);
      }

      /* -- Right Side -- */
      if ((right_start - pivot_r_w) < (4 * IndexStyle::vector_element_count())) {
        gather_sort::insertion_sort_fallback<SortOrderT>(data, indexes, right_w, right_start);
        right_range = ClusteredRange{static_cast<size_t>(right_w), static_cast<size_t>(right_start)};
        right_leaf = true;
      } else {
        const auto pivot_rs = get_pivot_indirect(data, indexes, pivot_r_w, right_start - 1);
        right_range = partition<SimdStyle, IndexStyle>(cluster, data, indexes, pivot_r_w, right_start, pivot_rs);
      }

      if (left_leaf) {
        if (right_leaf) {
          gather_sort::detect_cluster(cluster, data, indexes, left_start, right_start);
        } else {
          gather_sort::detect_cluster(cluster, data, indexes, left_start, right_range.start);
        }
      } else {
        if (right_leaf) {
          gather_sort::detect_cluster(cluster, data, indexes, left_range.end, right_start);
        } else {
          gather_sort::detect_cluster(cluster, data, indexes, left_range.end, right_range.start);
        }
      }

      return ClusteredRange{static_cast<size_t>(left_start), static_cast<size_t>(right_start)};
    }
  };
}  // namespace tuddbs

#endif