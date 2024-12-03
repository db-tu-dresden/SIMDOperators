
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
 * @file sort_indirect_inplace_cluster_tail.hpp
 * @brief
 */

#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_INDIRECT_INPLACE_CLUSTER_TAIL_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_INDIRECT_INPLACE_CLUSTER_TAIL_HPP

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
  class TailClusteringSingleColumnSortIndirectInplace {
    static_assert(has_hints_mutual_excluding<HintSet, std::tuple<hints::sort::indirect_inplace>,
                                             std::tuple<hints::sort::indirect_gather>>,
                  "Indirect sort can only be either inplace or gather, but both were given");
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
    TailClusteredSortState cluster_state;
    // std::deque<tuddbs::Cluster> clusters;

   public:
    explicit TailClusteringSingleColumnSortIndirectInplace(SimdOpsIterable auto p_data, SimdOpsIterable auto p_idx)
      : m_data{p_data}, m_idx{p_idx}, cluster_state{} {}

    auto operator()(const size_t left, const size_t right) {
      if ((right - left) < (4 * SimdStyle::vector_element_count())) {
        sort_inplace::insertion_sort_fallback<SortOrderT>(m_data, m_idx, left, right);
        sort_inplace::detect_cluster(this->getClusters(), m_data, m_idx, left, right);
        return;
      }

      const DataT pivot = tuddbs::get_pivot_indirect(m_data, m_idx, left, right - 1);
      // We dont need the top level ClusteredRange, it is a recursion helper and  thus we intentionally discard it here.
      static_cast<void>(sort_inplace::partition<SimdStyle, IndexStyle, SortOrderT>(cluster_state, m_data, m_idx, left, right, pivot));
    }

    std::deque<tuddbs::Cluster> & getClusters() { return cluster_state.clusters; }

  };
};  // namespace tuddbs

#endif