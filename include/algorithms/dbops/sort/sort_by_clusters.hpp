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
 * @file sort_multi_indirect.hpp
 * @brief
 */

#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_MULTI_INDIRECT_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_MULTI_INDIRECT_HPP

#include <climits>
#include <cstddef>
#include <deque>
#include <iterable.hpp>
#include <tuple>
#include <type_traits>

#include "algorithms/dbops/dbops_hints.hpp"
#include "algorithms/dbops/sort/sort.hpp"
#include "algorithms/dbops/sort/sort_utils.hpp"
#include "algorithms/utils/hinting.hpp"
#include "algorithms/utils/sorthints.hpp"
#include "tsl.hpp"

namespace tuddbs {
  template <tsl::VectorProcessingStyle _SimdStyle, tsl::VectorProcessingStyle _IndexStyle,
            class HintSet = OperatorHintSet<hints::sort::indirect_gather>>
  class ClusterSortIndirect {
    static_assert(has_hints_mutual_excluding<HintSet, std::tuple<hints::sort::indirect_gather>,
                                             std::tuple<hints::sort::indirect_inplace>>,
                  "Cluster Sort only supports Gather refinement as of now, but inplace was (also) given");

   public:
    using SimdStyle = _SimdStyle;
    using IndexStyle = _IndexStyle;
    using DataT = SimdStyle::base_type;
    using IdxT = IndexStyle::base_type;

   private:
    using asc_refiner = SingleColumnSort<SimdStyle, tuddbs::TSL_SORT_ORDER::ASC, HintSet, IndexStyle>;
    using desc_refiner = SingleColumnSort<SimdStyle, tuddbs::TSL_SORT_ORDER::DESC, HintSet, IndexStyle>;

    IdxT* m_idx;
    std::deque<tuddbs::Cluster>* m_clusters;

   public:
    explicit ClusterSortIndirect(SimdOpsIterable auto p_idx, std::deque<tuddbs::Cluster>* p_clusters)
      : m_idx{p_idx}, m_clusters{p_clusters} {}

    auto operator()(SimdOpsIterable auto p_data, tuddbs::TSL_SORT_ORDER order) {
      if (order == tuddbs::TSL_SORT_ORDER::ASC) {
        refine<typename asc_refiner::sorter_t>(p_data);
      } else {
        refine<typename desc_refiner::sorter_t>(p_data);
      }
    }

   private:
    template <class RefinerT>
    void refine(SimdOpsIterable auto p_data) {
      const size_t cluster_count = m_clusters->size();
      for (size_t i = 0; i < cluster_count; ++i) {
        tuddbs::Cluster& c = m_clusters->front();
        m_clusters->pop_front();
        if (c.len == 1) {
          continue;
        }
        const size_t start_pos = c.start;
        const size_t end_pos = start_pos + c.len;
        RefinerT refiner(p_data, m_idx);
        refiner(start_pos, end_pos);

        DataT curr_value = p_data[m_idx[start_pos]];
        std::deque<tuddbs::Cluster> new_clusters;
        size_t curr_start = start_pos;
        for (size_t i = start_pos + 1; i != end_pos; ++i) {
          DataT run_value = p_data[m_idx[i]];
          if (run_value != curr_value) {
            new_clusters.push_back(tuddbs::Cluster(curr_start, i - curr_start));
            curr_start = i;
            curr_value = run_value;
          }
        }
        new_clusters.emplace_back(tuddbs::Cluster(curr_start, end_pos - curr_start));

        while (!new_clusters.empty()) {
          m_clusters->push_back(new_clusters.front());
          new_clusters.pop_front();
        }
      }
    }
  };
}  // namespace tuddbs
#endif