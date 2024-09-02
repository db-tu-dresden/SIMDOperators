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
 * @file sort.hpp
 * @brief
 */
#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_HPP

#include "algorithms/dbops/dbops_hints.hpp"
#include "tslintrin.hpp"

namespace tuddbs {
  template <tsl::VectorProcessingStyle _SimdStyle, template <class, class> class CompareFun,
            class HintSet = OperatorHintSet<hints::intermediate::position_list>, typename Idof = tsl::workaround>
  class Generic_Sort {
   public:
    using SimdStyle = _SimdStyle;
    using PositionType = size_t using DataSinkType = PositionType *;
    using base_type = typename SimdStyle::base_type;
    using result_base_type = typename SimdStyle::imask_type;
  }

}  // namespace tuddbs

#endif