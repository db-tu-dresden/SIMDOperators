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
#include "algorithms/dbops/sort/sort_indirect_gather_cluster_leaf.hpp"
#include "algorithms/dbops/sort/sort_indirect_gather_cluster_tail.hpp"
#include "algorithms/dbops/sort/sort_indirect_inplace.hpp"
#include "algorithms/dbops/sort/sort_indirect_inplace_cluster_leaf.hpp"
#include "algorithms/dbops/sort/sort_indirect_inplace_cluster_tail.hpp"
#include "algorithms/utils/hinting.hpp"
#include "tsl.hpp"

/**
 * @brief This is a convenience proxy to select a column sorter, based on the given hints from the namespace
 * tuddbs::sort.
 *
 * @tparam _SimdStyle The TSL processing style, which is used to access the data column
 * @tparam SortOrderT Sort the data ascending or descending.
 * @tparam HintSet The Set of hints to help instantiate the appropriate Sorter. Supported versions: direct,
 * indirect_inplace, indirect_gather.
 * @tparam _IndexStyle The TSL processing style, which is used to access the index column with indirect sorting.
 */
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
                           SingleColumnSortIndirectGather<_SimdStyle, _IndexStyle, SortOrderT, HintSet>, void>>>;
  };

  /**
   * @brief This is a convenience proxy to select an indirect, i.e. index column, sorter, based on the given hints from
   * the namespace tuddbs::sort. See HintSet for more information.
   *
   * @tparam _SimdStyle The TSL processing style, which is used to access the data column
   * @tparam SortOrderT Sort the data ascending or descending.
   * @tparam HintSet The Set of hints to help instantiate the appropriate Sorter. Possible Combinations:
   * indirect_inplace with [tail_clustering | leaf_clustering] or indirect_gather with [tail_clustering |
   * leaf_clustering]
   * @tparam _IndexStyle The TSL processing style, which is used to access the index column
   */
  template <tsl::VectorProcessingStyle _SimdStyle, TSL_SORT_ORDER SortOrderT = TSL_SORT_ORDER::ASC,
            class HintSet = OperatorHintSet<hints::sort::indirect_inplace>,
            tsl::VectorProcessingStyle _IndexStyle = _SimdStyle>
  struct ClusteringSingleColumnSort {
    using sorter_t = std::conditional_t<
      has_hints<HintSet, hints::sort::indirect_inplace> && has_hints<HintSet, hints::sort::tail_clustering>,
      TailClusteringSingleColumnSortIndirectInplace<_SimdStyle, _IndexStyle, SortOrderT, HintSet>,
      std::conditional_t<
        has_hints<HintSet, hints::sort::indirect_inplace> && has_hints<HintSet, hints::sort::leaf_clustering>,
        LeafClusteringSingleColumnSortIndirectInplace<_SimdStyle, _IndexStyle, SortOrderT, HintSet>,
        std::conditional_t<
          has_hints<HintSet, hints::sort::indirect_gather> && has_hints<HintSet, hints::sort::tail_clustering>,
          TailClusteringSingleColumnSortIndirectGather<_SimdStyle, _IndexStyle, SortOrderT, HintSet>,
          std::conditional_t<
            has_hints<HintSet, hints::sort::indirect_gather> && has_hints<HintSet, hints::sort::leaf_clustering>,
            LeafClusteringSingleColumnSortIndirectGather<_SimdStyle, _IndexStyle, SortOrderT, HintSet>, void>>>>;
  };
}  // namespace tuddbs

#endif