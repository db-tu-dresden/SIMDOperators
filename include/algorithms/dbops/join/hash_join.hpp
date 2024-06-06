// ------------------------------------------------------------------- //
/*
   This file is part of the SimdOperators Project.
   Author(s): Lennart Schmidt.

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
 * @file hash_join.hpp
 * @brief
 */
#ifndef SIMDOPS_INCLUE_ALGORITHMS_DBOPS_JOIN_HASH_JOIN_HPP
#define SIMDOPS_INCLUE_ALGORITHMS_DBOPS_JOIN_HASH_JOIN_HPP

#include "algorithms/dbops/dbops_hints.hpp"
#include "algorithms/dbops/join/hash_join_hints.hpp"
#include "algorithms/dbops/join/hash_join_simd_linear_probing.hpp"
#include "algorithms/utils/hashing.hpp"
#include "tslintrin.hpp"

namespace tuddbs {

  template <tsl::VectorProcessingStyle _SimdStyle, tsl::TSLArithmetic _PositionType,
            class HintSet = OperatorHintSet<hints::hashing::size_exp_2, hints::hashing::linear_displacement>,
            typename Idof = tsl::workaround>
  struct Hash_Join {
    using base_class =
      std::conditional_t<has_hints<HintSet, hints::hashing::linear_displacement> &&
                           !has_hint<HintSet, hints::hashing::refill>,
                         Hash_Join_SIMD_Linear_Probing<_SimdStyle, _PositionType, HintSet, Idof>, void>;
    using builder_t = typename base_class::builder_t;
    using prober_t = typename base_class::prober_t;
  };

}  // namespace tuddbs

#endif