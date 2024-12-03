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

#include <climits>
#include <cstddef>
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
  class SingleColumnSortIndirectGather {
    static_assert(has_hints_mutual_excluding<HintSet, std::tuple<hints::sort::indirect_gather>,
                                             std::tuple<hints::sort::indirect_inplace>>,
                  "Indirect sort can only be either gather or inplace, but both were given");

   public:
    using SimdStyle = _SimdStyle;
    using IndexStyle = _IndexStyle;
    using DataT = SimdStyle::base_type;
    using IdxT = IndexStyle::base_type;

   private:
    DataT* m_data;
    IdxT* m_idx;
    DefaultSortState state;

   public:
    explicit SingleColumnSortIndirectGather(SimdOpsIterable auto p_data, SimdOpsIterable auto p_idx)
      : m_data{p_data}, m_idx{p_idx}, state{} {}

    auto operator()(const size_t left, const size_t right) {
      if ((right - left) < (4 * SimdStyle::vector_element_count())) {
        gather_sort::insertion_sort_fallback<SortOrderT>(m_data, m_idx, left, right);
        return;
      }
      const DataT pivot = tuddbs::get_pivot_indirect(m_data, m_idx, left, right - 1);
      gather_sort::partition<SimdStyle, IndexStyle, SortOrderT>(state, m_data, m_idx, left, right, pivot);
    }
  };
};  // namespace tuddbs

#endif