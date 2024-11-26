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

#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_INDIRECT_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_INDIRECT_HPP

#include <climits>
#include <cstddef>
#include <iterable.hpp>
#include <tuple>
#include <type_traits>

#include "algorithms/dbops/dbops_hints.hpp"
#include "algorithms/dbops/sort/sort_utils.hpp"
#include "algorithms/utils/hinting.hpp"
#include "algorithms/utils/sorthints.hpp"
#include "tsl.hpp"

namespace tuddbs {
  template <tsl::VectorProcessingStyle _SimdStyle, tsl::VectorProcessingStyle _IndexStyle, TSL_SORT_ORDER SortOrderT,
            class HintSet = OperatorHintSet<hints::sort_indirect::inplace>>
  class SortIndirect {
    static_assert(has_hints_mutual_excluding<HintSet, std::tuple<hints::sort_indirect::inplace>,
                                             std::tuple<hints::sort_indirect::gather>> ||
                    has_hints_mutual_excluding<HintSet, std::tuple<hints::sort_indirect::gather>,
                                               std::tuple<hints::sort_indirect::inplace>>,
                  "Erreur");

   public:
    using SimdStyle = _SimdStyle;
    using IndexStyle = _IndexStyle;
    using DataT = SimdStyle::base_type;
    using IdxT = IndexStyle::base_type;

   private:
    DataT* m_data;
    IdxT* m_idx;

   public:
    explicit SortIndirect(SimdOpsIterable auto p_data, SimdOpsIterable auto p_idx) : m_data{p_data}, m_idx{p_idx} {}

    template <class HS = HintSet>
    auto operator()(const size_t left, const size_t right,
                    enable_if_has_hint_t<HS, hints::sort_indirect::inplace> = {}) {
      std::cout << "Indirect Sort with inplace hint enabled." << std::endl;
    }

    template <class HS = HintSet>
    auto operator()(const size_t left, const size_t right,
                    enable_if_has_hint_t<HS, hints::sort_indirect::gather> = {}) {
      std::cout << "Indirect Sort with gather hint enabled." << std::endl;
    }

   private:
  };
};  // namespace tuddbs

#endif