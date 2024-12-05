// ------------------------------------------------------------------- //
/*
   This file is part of the SimdOperators Project.
   Author(s): Johannes Pietrzyk.

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
 * @file group.hpp
 * @brief
 */
#ifndef SIMDOPS_INCLUE_ALGORITHMS_DBOPS_GROUPBY_GROUPBY_HPP
#define SIMDOPS_INCLUE_ALGORITHMS_DBOPS_GROUPBY_GROUPBY_HPP

#include "algorithms/dbops/dbops_hints.hpp"
#include "algorithms/dbops/group_aggregate/group_sum.hpp"
#include "algorithms/dbops/groupby/groupby_hints.hpp"
#include "algorithms/dbops/groupby/groupby_simd_linear_displacement.hpp"
#include "algorithms/utils/hashing.hpp"
#include "tslintrin.hpp"

namespace tuddbs {

  template <tsl::VectorProcessingStyle _SimdStyle, tsl::TSLArithmetic _PositionType,
            class HintSet = OperatorHintSet<hints::hashing::size_exp_2, hints::hashing::linear_displacement>,
            typename Idof = tsl::workaround>
  struct Group {
    using base_class =
      std::conditional_t<has_hints<HintSet, hints::hashing::linear_displacement> &&
                           !has_hint<HintSet, hints::hashing::refill>,
                         Grouper_SIMD_Linear_Displacement<_SimdStyle, _PositionType, HintSet, Idof>, void>;
    using builder_t = typename base_class::builder_t;
    using grouper_t = typename base_class::grouper_t;
  };

  template <tsl::VectorProcessingStyle _SimdStyle, tsl::TSLArithmetic _ValueType = typename _SimdStyle::base_type,
            class HintSet = OperatorHintSet<hints::hashing::size_exp_2, hints::hashing::linear_displacement>,
            typename Idof = tsl::workaround>
  struct GroupAggregate_Sum {
    using base_class =
      std::conditional_t<has_hints<HintSet, hints::hashing::linear_displacement> &&
                           !has_hint<HintSet, hints::hashing::refill>,
                         Grouper_Aggregate_SUM_SIMD_Linear_Displacement<_SimdStyle, _ValueType, HintSet, Idof>, void>;
    using builder_t = typename base_class::builder_t;
    using grouper_t = typename base_class::grouper_t;
  };

}  // namespace tuddbs

#endif