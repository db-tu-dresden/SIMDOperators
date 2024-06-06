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
 * @file hash_join_hints.hpp
 * @brief
 */
#ifndef SIMDOPS_INCLUE_ALGORITHMS_DBOPS_JOIN_HASH_JOIN_HINTS_HPP
#define SIMDOPS_INCLUE_ALGORITHMS_DBOPS_JOIN_HASH_JOIN_HINTS_HPP

namespace tuddbs {

  namespace hints {
    namespace hash_join {
      struct global_first_occurence_required {};
      struct keys_may_contain_empty_indicator {};
    }  // namespace hash_join
  }  // namespace hints

}  // namespace tuddbs

#endif