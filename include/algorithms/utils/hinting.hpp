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
 * @file hinting.hpp
 * @brief
 */
#ifndef SIMDOPS_INCLUDE_ALGORITHMS_UTILS_HINTING_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_UTILS_HINTING_HPP

#include <tuple>
#include <type_traits>

namespace tuddbs {
  template <typename... Args>
  struct OperatorHintSet {
    template <typename Hint>
    using has_type_t = std::disjunction<std::is_same<Hint, Args>...>;
    template <typename... Hints>
    using has_types_t = std::conjunction<has_type_t<Hints>...>;
  };
  template <typename HS, typename Arg>
  inline constexpr bool has_hint = HS::template has_type_t<Arg>::value;

  template <typename HS, typename... Args>
  inline constexpr bool has_hints = HS::template has_types_t<Args...>::value;

  template <typename HS, typename... Args>
  inline constexpr bool has_any_hint = (has_hint<HS, Args> || ...);

  template <typename HS, typename Arg, typename T = Arg>
  using enable_if_has_hint_t = typename std::enable_if_t<has_hint<HS, Arg>, T>;

  template <typename HS, typename Arg, typename T = Arg>
  using disable_if_has_hint_t = typename std::enable_if_t<!has_hint<HS, Arg>, T>;

  template <typename HS, typename... Args>
  using enable_if_has_hints_t = typename std::enable_if_t<has_hints<HS, Args...>, std::tuple<Args...>>;

}  // namespace tuddbs
#endif