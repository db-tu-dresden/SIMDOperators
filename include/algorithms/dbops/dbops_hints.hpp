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
 * @file dbops_hints.hpp
 * @brief
 */
#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_HINTS_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_HINTS_HPP

#include "algorithms/utils/hinting.hpp"
namespace tuddbs {

  namespace hints {
    namespace operators {
      struct preserve_original_positions {};
      struct collect_metrics {};
    }  // namespace operators
    namespace intermediate {
      /**
       * @brief Tag to identify that an operator produces a position list.
       */
      struct position_list {};
      /**
       * @brief Tag to identify that an operator produces a bitmask.
       */
      struct bit_mask {};

      struct dense_bit_mask {};

    }  // namespace intermediate

    namespace arithmetic {
      struct add {};
      struct sub {};
      struct div {};
      struct mul {};
      struct sum {};
      struct average {};
    }  // namespace arithmetic
  }  // namespace hints

  template <typename HS>
  struct intermediate_hint_helper_t {
    static_assert(count_hints<HS, hints::intermediate::dense_bit_mask, hints::intermediate::bit_mask,
                              hints::intermediate::position_list> <= 1,
                  "Intermediate type can be only one of the supported types");
    constexpr static bool use_dense_bitmask = has_hint<HS, hints::intermediate::dense_bit_mask>;
    constexpr static bool use_bitmask = has_hint<HS, hints::intermediate::bit_mask>;
    constexpr static bool use_position_list = has_hint<HS, hints::intermediate::position_list>;
  };

  template <typename HS>
  struct arithmetic_hint_helper_t {
    // static_assert(count_hints<HS, hints::intermediate::dense_bit_mask, hints::intermediate::bit_mask,
    // hints::intermediate::position_list> <= 1,
    // "Intermediate type can be only one of the supported types");
    constexpr static bool perform_add = has_hint<HS, hints::arithmetic::add>;
    constexpr static bool perform_sub = has_hint<HS, hints::arithmetic::sub>;
    constexpr static bool perform_div = has_hint<HS, hints::arithmetic::div>;
    constexpr static bool perform_mul = has_hint<HS, hints::arithmetic::mul>;
  };

  /* Intermediates */
  template <typename HS, typename HintHelper = intermediate_hint_helper_t<HS>>
  using activate_for_dense_bit_mask =
    typename std::enable_if_t<HintHelper::use_dense_bitmask, hints::intermediate::dense_bit_mask>;
  template <typename HS, typename HintHelper = intermediate_hint_helper_t<HS>>
  using activate_for_bit_mask = typename std::enable_if_t<HintHelper::use_bitmask, hints::intermediate::bit_mask>;
  template <typename HS, typename HintHelper = intermediate_hint_helper_t<HS>>
  using activate_for_position_list =
    typename std::enable_if_t<HintHelper::use_position_list, hints::intermediate::position_list>;

  /* Arithmetics */
  template <typename HS, typename HintHelper = arithmetic_hint_helper_t<HS>>
  using activate_for_add = typename std::enable_if_t<HintHelper::perform_add, hints::arithmetic::add>;

  template <typename HS, typename HintHelper = arithmetic_hint_helper_t<HS>>
  using activate_for_sub = typename std::enable_if_t<HintHelper::perform_sub, hints::arithmetic::sub>;

  template <typename HS, typename HintHelper = arithmetic_hint_helper_t<HS>>
  using activate_for_div = typename std::enable_if_t<HintHelper::perform_div, hints::arithmetic::div>;

  template <typename HS, typename HintHelper = arithmetic_hint_helper_t<HS>>
  using activate_for_mul = typename std::enable_if_t<HintHelper::perform_mul, hints::arithmetic::mul>;

}  // namespace tuddbs
#endif