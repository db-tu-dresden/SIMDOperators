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
 * @file simdops.hpp
 * @brief
 */

#ifndef SIMDOPS_INCLUDE_DBOPS_SIMDOPS_HPP
#define SIMDOPS_INCLUDE_DBOPS_SIMDOPS_HPP

#include <type_traits>

namespace tuddbs {
  /**
   * @brief Tag to identify that an operator produces a position list.
   */
  struct position_list {};
  /**
   * @brief Tag to identify that an operator produces a bitmask.
   */
  struct bit_mask {};

  template <typename... Args>
  struct OperatorHintSet {
    template <typename Arg>
    using has_type_t = std::disjunction<std::is_same<Arg, Args>...>;
  };
  template <typename HS, typename Arg>
  inline constexpr bool has_hint = HS::template has_type_t<Arg>::value;

}  // namespace tuddbs
#endif