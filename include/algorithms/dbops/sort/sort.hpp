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
 * @file sort_indirect.hpp
 * @brief
 */

#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_PROXY_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_PROXY_HPP

#include "algorithms/dbops/dbops_hints.hpp"
#include "algorithms/dbops/sort/sort_direct.hpp"
#include "algorithms/dbops/sort/sort_indirect_gather.hpp"
#include "algorithms/dbops/sort/sort_indirect_inplace.hpp"
#include "algorithms/dbops/sort/sort_indirect_inplace_cluster_leaf.hpp"
#include "algorithms/dbops/sort/sort_indirect_inplace_cluster_tail.hpp"
#include "algorithms/utils/hinting.hpp"
#include "tsl.hpp"

namespace tuddbs {
  template <tsl::VectorProcessingStyle _SimdStyle, TSL_SORT_ORDER SortOrderT = TSL_SORT_ORDER::ASC,
            class HintSet = OperatorHintSet<hints::sort::direct>, tsl::VectorProcessingStyle _IndexStyle = _SimdStyle>
  struct SingleColumnSort {
    using sorter_t = std::conditional_t<
      has_hints<HintSet, hints::sort::direct>, SingleColumnSortDirect<_SimdStyle, SortOrderT>,
      std::conditional_t<
        has_hints<HintSet, hints::sort::indirect_inplace>,
        SingleColumnSortIndirectInplace<_SimdStyle, _IndexStyle, SortOrderT, HintSet>,
        std::conditional_t<has_hints<HintSet, hints::sort::indirect_gather>,
                           SingleColumnSortIndirectGather<_SimdStyle, _IndexStyle, SortOrderT, HintSet>, void> > >;
  };

  template <tsl::VectorProcessingStyle _SimdStyle, TSL_SORT_ORDER SortOrderT = TSL_SORT_ORDER::ASC,
            class HintSet = OperatorHintSet<hints::sort::indirect_inplace>,
            tsl::VectorProcessingStyle _IndexStyle = _SimdStyle>
  struct ClusteringSingleColumnSort {
    using sorter_t = std::conditional_t<
      has_hints<HintSet, hints::sort::indirect_inplace> && has_hints<HintSet, hints::sort::tail_clustering>,
      TailClusteringSingleColumnSortIndirectInplace<_SimdStyle, _IndexStyle, SortOrderT, HintSet>,
      std::conditional_t<
        has_hints<HintSet, hints::sort::indirect_inplace> && has_hints<HintSet, hints::sort::leaf_clustering>,
        LeafClusteringSingleColumnSortIndirectInplace<_SimdStyle, _IndexStyle, SortOrderT, HintSet>, void> >;
  };
}  // namespace tuddbs

#endif