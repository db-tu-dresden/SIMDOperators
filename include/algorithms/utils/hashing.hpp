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
 * @file hashing.hpp
 * @brief
 */

#ifndef SIMDOPS_INCLUDE_ALGORITHMS_UTILS_HASHING_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_UTILS_HASHING_HPP

#include "algorithms/utils/hinting.hpp"
#include "iterable.hpp"
#include "tslintrin.hpp"

namespace tuddbs {

  namespace hints {
    namespace hashing {
      struct unique_keys {};
      struct size_exp_2 {};
      struct keys_may_contain_zero {};
      struct is_hull_for_merging {};

      struct linear_displacement {};
      struct refill {};
    }  // namespace hashing
  }    // namespace hints

  template <tsl::VectorProcessingStyle SimdStyle, class HintSet, tsl::ImplementationDegreeOfFreedom Idof>
  class normalizer {
   public:
    [[nodiscard]] TSL_FORCE_INLINE static auto normalize(typename SimdStyle::register_type position_hint,
                                                         typename SimdStyle::register_type bucket_count) {
      if constexpr (std::is_integral_v<typename SimdStyle::base_type>) {
         if constexpr (has_hint<HintSet, hints::hashing::size_exp_2>) {
           return tsl::binary_and<SimdStyle, Idof>(position_hint, bucket_count);
         } else {
           return tsl::mod<SimdStyle, Idof>(position_hint, bucket_count);
         }
      } else {
         // do the float magic here
         using IntegralSimdStyle = typename SimdStyle::template transform_extension<typename SimdStyle::offset_base_type>;
         auto casted_position_hint = tsl::reinterpret<SimdStyle, IntegralSimdStyle>(position_hint);
      }
    }
    [[nodiscard]] TSL_FORCE_INLINE static auto normalize_value(typename SimdStyle::base_type position_hint,
                                                               typename SimdStyle::base_type bucket_count) {
      if constexpr (std::is_integral_v<typename SimdStyle::base_type>) {
         if constexpr (has_hint<HintSet, hints::hashing::size_exp_2>) {
           return position_hint & (bucket_count - 1);
         } else {
           return position_hint % bucket_count;
         }
      } else {
         //auto casted_integral_position_hint = static_cast<typename SimdStyle::offset_base_type>(position_hint);
         auto casted_integral_position_hint = *(reinterpret_cast<typename SimdStyle::offset_base_type*>(&position_hint));
         //...

      }
    }
    [[nodiscard]] TSL_FORCE_INLINE static auto align_value(typename SimdStyle::base_type position_hint) {
      return position_hint - (position_hint & (SimdStyle::vector_element_count() - 1));
    }
  };

  template <tsl::VectorProcessingStyle SimdStyle, tsl::ImplementationDegreeOfFreedom Idof>
  class default_hasher {
   public:
    [[nodiscard]] TSL_FORCE_INLINE static auto hash(typename SimdStyle::register_type key) { return key; }
    [[nodiscard]] TSL_FORCE_INLINE static auto hash_value(typename SimdStyle::base_type key) { return key; }
  };

}  // namespace tuddbs
#endif
