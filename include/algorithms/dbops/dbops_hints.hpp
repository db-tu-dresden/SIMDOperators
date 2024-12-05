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
  }    // namespace hints

  template <typename HS>
  struct intermediate_hint_helper_t {
    static_assert(count_hints<HS, hints::intermediate::dense_bit_mask, hints::intermediate::bit_mask,
                              hints::intermediate::position_list> <= 1,
                  "Intermediate type can be only one of the supported types");
    constexpr static bool use_dense_bitmask = has_hint<HS, hints::intermediate::dense_bit_mask>;
    constexpr static bool use_bitmask = has_hint<HS, hints::intermediate::bit_mask>;
    constexpr static bool use_position_list = has_hint<HS, hints::intermediate::position_list>;
  };

  template <typename HS, typename HintHelper = intermediate_hint_helper_t<HS>>
  using activate_for_dense_bit_mask =
    typename std::enable_if_t<HintHelper::use_dense_bitmask, hints::intermediate::dense_bit_mask>;
  template <typename HS, typename HintHelper = intermediate_hint_helper_t<HS>>
  using activate_for_bit_mask = typename std::enable_if_t<HintHelper::use_bitmask, hints::intermediate::bit_mask>;
  template <typename HS, typename HintHelper = intermediate_hint_helper_t<HS>>
  using activate_for_position_list =
    typename std::enable_if_t<HintHelper::use_position_list, hints::intermediate::position_list>;

}  // namespace tuddbs
#endif