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
 * @file sort_core.hpp
 * @brief
 */

#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_CORE_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_CORE_HPP

#include "algorithms/dbops/sort/sort_utils.hpp"

namespace tuddbs {
  template <typename T, typename U>
  void detect_cluster(std::deque<Cluster>& clusters, T* data, U* indexes, size_t left, size_t right) {
    T run_value = data[left];
    size_t run_length = 1;
    ++left;
    if (left > right) {
      return;
    }
    for (; left < right; ++left) {
      const T current_value = data[left];
      if (current_value != run_value) {
        if (run_length > 1) {
          clusters.emplace_back(left - run_length, run_length);
        }
        run_length = 1;
        run_value = current_value;
      } else {
        ++run_length;
      }
    }
    if (run_length > 1) {
      clusters.emplace_back(left - run_length, run_length);
    }
  }

  /**
   * @brief Templated helper to make calling the comparison with TYPE a one-liner and increase readability
   *
   * @tparam SimdStyle TSL processing style to compare the value register with the pivot register
   * @tparam type The comparison type, i.e. LessThan or Equal
   */
  template <class SimdStyle, SORT_TYPE type, TSL_SORT_ORDER order>
  __attribute__((always_inline)) inline auto compare(const typename SimdStyle::register_type val_reg,
                                                     const typename SimdStyle::register_type pivot_reg) ->
    typename SimdStyle::imask_type {
    if constexpr (type == SORT_TYPE::SORT_EQ) {
      return tsl::to_integral<SimdStyle>(tsl::equal<SimdStyle>(val_reg, pivot_reg));
    } else {
      if constexpr (order == TSL_SORT_ORDER::ASC) {
        return tsl::to_integral<SimdStyle>(tsl::less_than<SimdStyle>(val_reg, pivot_reg));
      } else {
        return tsl::to_integral<SimdStyle>(tsl::greater_than<SimdStyle>(val_reg, pivot_reg));
      }
    }
  }

  template <class SimdStyle, SORT_TYPE type, TSL_SORT_ORDER order>
  __attribute__((always_inline)) inline auto compare(const typename SimdStyle::register_type val_reg,
                                                     const typename SimdStyle::register_type pivot_reg,
                                                     const typename SimdStyle::imask_type valid) ->
    typename SimdStyle::imask_type {
    return compare<SimdStyle, type, order>(val_reg, pivot_reg) & valid;
  }
}  // namespace tuddbs

#endif