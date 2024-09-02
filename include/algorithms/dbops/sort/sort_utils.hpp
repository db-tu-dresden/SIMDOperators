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
 * @file sort_utils.hpp
 * @brief
 */

#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_UTILS_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SORT_SORT_UTILS_HPP

#include <cstdint>

namespace tuddbs {
  enum class TSL_SORT_ORDER { ASC, DESC };
  enum class SORT_TYPE { SORT_EQ, SORT_LT };

  // This is a magic number and might be changed
  const size_t Med9_threshold = 40;

  template <typename T>
  T median(T a, T b, T c) {
    if ((a > b) ^ (a > c)) {
      return a;
    } else if ((b < a) ^ (b < c)) {
      return b;
    } else {
      return c;
    }
  }

  template <typename T>
  T median3(T* data, const size_t a, const size_t b, const size_t c) {
    return median(data[a], data[b], data[c]);
  }

  template <typename T>
  T median9(T* data, const size_t left, const size_t right) {
    const size_t d = (right - left) / 8;
    return median(median3(data, left, left + d, left + 2 * d), median3(data, left + 3 * d, left + 4 * d, left + 5 * d),
                  median3(data, right - 2 * d, right - d, right));
  }

  template <typename T>
  T get_pivot(T* data, const size_t left, const size_t right) {
    const size_t dist = right - left;
    if (dist > Med9_threshold) {
      return median9(data, left, right);
    } else {
      return median3(data, left, left + (dist / 2), right);
    }
  }

  template <typename data_t, typename index_t>
  data_t median3_indirect(data_t* data, index_t* indexes, const size_t a, const size_t b, const size_t c) {
    return median(data[indexes[a]], data[indexes[b]], data[indexes[c]]);
  }

  template <typename data_t, typename index_t>
  data_t median9_indirect(data_t* data, index_t* indexes, const size_t left, const size_t right) {
    const size_t d = (right - left) / 8;
    return median(median3_indirect(data, indexes, left, left + d, left + 2 * d),
                  median3_indirect(data, indexes, left + 3 * d, left + 4 * d, left + 5 * d),
                  median3_indirect(data, indexes, right - 2 * d, right - d, right));
  }

  template <typename data_t, typename index_t>
  data_t get_pivot_indirect(data_t* data, index_t* indexes, const size_t left, const size_t right) {
    const size_t dist = right - left;
    if (dist > Med9_threshold) {
      return median9_indirect(data, indexes, left, right);
    } else {
      return median3_indirect(data, indexes, left, left + (dist / 2), right);
    }
  }
}  // namespace tuddbs

#endif