
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
 * @file multiply.hpp
 * @brief
 */

#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_ARITHMETIC_MULTIPLY_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_ARITHMETIC_MULTIPLY_HPP

#include <climits>
#include <cstddef>
#include <deque>
#include <iostream>
#include <iterable.hpp>
#include <tuple>
#include <type_traits>

#include "algorithms/dbops/dbops_hints.hpp"
#include "generated/declarations/calc.hpp"
#include "tsl.hpp"

namespace tuddbs {
  /**
   * @brief A class for performing arithmetic operations on columns.
   * @details We assume that the data is stored in a columnar format.
   * Furthermore, the operation must not exceed the range of the base type.
   * Consequently, when working with uint8_t, the maximum possible result is 255.
   * @todo We should add a convert_up version to support arbitrary length columns.
   *
   * @tparam _SimdStyle
   * @tparam HintSet
   */
  template <tsl::VectorProcessingStyle _SimdStyle, class HintSet = OperatorHintSet<hints::arithmetic::add>>
  class Arithmetic {
   public:
    using SimdStyle = _SimdStyle;
    using reg_t = typename SimdStyle::register_type;
    using base_t = typename SimdStyle::base_type;

    explicit Arithmetic() {}

    /* Reducing Operations on a single column, e.g., sum, avg */
    template <class HS = HintSet>
    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end,
                    activate_for_position_list<HS> = {}) {
      const auto simd_end = tuddbs::simd_iter_end<SimdStyle>(p_data, p_end);
      const auto scalar_end = tuddbs::iter_end(p_data, p_end);

      reg_t res_vec = tsl::set1<SimdStyle>(0);
      if constexpr (std::is_floating_point_v<base_t>) {
        // This is a SIMDified version of the Kahan summation
        reg_t error_vec = tsl::set1<SimdStyle>(0);
        for (; p_data != simd_end; p_data += SimdStyle::vector_element_count()) {
          reg_t vals = tsl::loadu<SimdStyle>(p_data);
          vals = tsl::sub<SimdStyle>(vals, error_vec);
          reg_t buffer = tsl::add<SimdStyle>(res_vec, vals);
          error_vec = tsl::sub<SimdStyle>(tsl::sub<SimdStyle>(buffer, res_vec), vals);
          res_vec = buffer;
        }
      } else {
        for (; p_data != simd_end; p_data += SimdStyle::vector_element_count()) {
          res_vec = tsl::add<SimdStyle>(res_vec, tsl::loadu<SimdStyle>(p_data));
        }
      }

      base_t res_scalar = 0;

      if constexpr (std::is_floating_point_v<base_t>) {
        base_t error = static_cast<base_t>(0.0);
        for (; p_data != scalar_end; p_data++) {
          // This is a scalar version of the Kahan summation
          base_t y = *p_data - error;
          base_t t = res_scalar + y;
          error = (t - res_scalar) - y;
          res_scalar = t;
        }
        // Scalar remainder for Kahan
        const auto vec_res = tsl::hadd<SimdStyle>(res_vec);
        base_t y = vec_res - error;
        res_scalar += y;
      } else {
        for (; p_data != scalar_end; p_data++) {
          res_scalar += *p_data;
        }
        res_scalar += tsl::hadd<SimdStyle>(res_vec);
      }

      if constexpr (has_hint<HintSet, tuddbs::hints::arithmetic::sum>) {
        *p_result = res_scalar;
      } else if constexpr (has_hint<HintSet, tuddbs::hints::arithmetic::average>) {
        const size_t element_count = scalar_end - p_data;
        if constexpr (std::is_floating_point_v<base_t>) {
          *p_result = res_scalar / element_count;
        } else {
          *p_result = static_cast<double>(res_scalar) / element_count;
        }
      } else {
        throw std::runtime_error("Unknown single-column arithmetic. No suitable hint was provided.");
      }
    }

    template <class HS = HintSet>
    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end,
                    SimdOpsIterable auto p_valid_masks, activate_for_bit_mask<HS> = {}) {
      using CountSimdStyle = typename SimdStyle::template transform_extension<typename SimdStyle::offset_base_type>;

      const auto simd_end = tuddbs::simd_iter_end<SimdStyle>(p_data, p_end);
      const auto scalar_end = tuddbs::iter_end(p_data, p_end);
      auto valid_masks = reinterpret_iterable<typename SimdStyle::imask_type *>(p_valid_masks);
      auto valid_elements_count_reg = tsl::set1<CountSimdStyle>(0);
      auto const valid_elements_increment_reg = tsl::set1<CountSimdStyle>(1);
      reg_t res_vec = tsl::set1<SimdStyle>(static_cast<base_t>(0));

      if constexpr (std::is_floating_point_v<base_t>) {
        // This is a SIMDified version of the Kahan summation
        reg_t error_vec = tsl::set1<SimdStyle>(0);
        for (; p_data != simd_end; p_data += SimdStyle::vector_element_count(), ++valid_masks) {
          reg_t vals = tsl::loadu<SimdStyle>(p_data);
          auto valid_mask = tsl::load_mask<SimdStyle>(valid_masks);
          vals = tsl::sub<SimdStyle>(valid_mask, vals, error_vec);
          reg_t buffer = tsl::add<SimdStyle>(valid_mask, res_vec, vals);
          error_vec = tsl::blend<SimdStyle>(
            valid_mask, error_vec,
            tsl::sub<SimdStyle>(valid_mask, tsl::sub<SimdStyle>(valid_mask, buffer, res_vec), vals));

          res_vec = buffer;
          if constexpr (std::is_same_v<decltype(valid_mask), typename CountSimdStyle::mask_type>) {
            valid_elements_count_reg =
              tsl::add<CountSimdStyle>(valid_mask, valid_elements_count_reg, valid_elements_increment_reg);
          } else {
            valid_elements_count_reg = tsl::add<CountSimdStyle>(tsl::reinterpret<SimdStyle, CountSimdStyle>(valid_mask),
                                                                valid_elements_count_reg, valid_elements_increment_reg);
          }
        }
      } else {
        auto const p_data_start = p_data;
        for (; p_data != simd_end; p_data += SimdStyle::vector_element_count(), ++valid_masks) {
          auto valid_mask = tsl::load_mask<SimdStyle>(valid_masks);
          res_vec = tsl::add<SimdStyle>(valid_mask, res_vec, tsl::loadu<SimdStyle>(p_data));
          valid_elements_count_reg =
            tsl::add<CountSimdStyle>(valid_mask, valid_elements_count_reg, valid_elements_increment_reg);
        }
      }

      base_t res_scalar = 0;
      typename SimdStyle::offset_base_type valid_elements_count = tsl::hadd<CountSimdStyle>(valid_elements_count_reg);
      if constexpr (std::is_floating_point_v<base_t>) {
        base_t error = static_cast<base_t>(0.0);
        auto valid_mask = tsl::load_imask<SimdStyle>(valid_masks);
        for (; p_data != scalar_end; p_data++) {
          if ((valid_mask & 0b1) == 0b1) {
            // This is a scalar version of the Kahan summation
            base_t y = *p_data - error;
            base_t t = res_scalar + y;
            error = (t - res_scalar) - y;
            res_scalar = t;
            ++valid_elements_count;
          }
          valid_mask >>= 1;
        }
        // Scalar remainder for Kahan
        const auto vec_res = tsl::hadd<SimdStyle>(res_vec);
        base_t y = vec_res - error;
        res_scalar += y;
      } else {
        auto valid_mask = tsl::load_imask<SimdStyle>(valid_masks);
        // std::cout << "Remainder:" << std::endl;
        for (; p_data != scalar_end; p_data++) {
          if ((valid_mask & 0b1) == 0b1) {
            res_scalar += *p_data;
            ++valid_elements_count;
          }
          valid_mask >>= 1;
        }
        res_scalar += tsl::hadd<SimdStyle>(res_vec);
      }

      if constexpr (has_hint<HintSet, tuddbs::hints::arithmetic::sum>) {
        *p_result = res_scalar;
      } else if constexpr (has_hint<HintSet, tuddbs::hints::arithmetic::average>) {
        if constexpr (std::is_floating_point_v<base_t>) {
          *p_result = res_scalar / valid_elements_count;
        } else {
          *p_result = static_cast<double>(res_scalar) / valid_elements_count;
        }
      } else {
        throw std::runtime_error("Unknown single-column arithmetic. No suitable hint was provided.");
      }
    }

    /* Combining two columns element-wise, e.g., add, sub, div, mul
     * There is no need for bitmasks, as we assume columns to be materialized prior to calculation.
     */
    template <typename HS = HintSet>
    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_data1, SimdOpsIterableOrSizeT auto p_end1,
                    SimdOpsIterable auto p_data2, activate_for_position_list<HS> = {}) {
      const auto simd_end = tuddbs::simd_iter_end<SimdStyle>(p_data1, p_end1);
      const auto scalar_end = tuddbs::iter_end(p_data1, p_end1);
      for (; p_data1 != simd_end; p_data1 += SimdStyle::vector_element_count(),
                                  p_data2 += SimdStyle::vector_element_count(),
                                  p_result += SimdStyle::vector_element_count()) {
        const auto vals1 = tsl::loadu<SimdStyle>(p_data1);
        const auto vals2 = tsl::loadu<SimdStyle>(p_data2);
        tsl::storeu<SimdStyle>(p_result, calc(vals1, vals2));
      }
      // Process the scalar remainder
      for (; p_data1 != scalar_end; p_data1++, p_data2++) {
        *p_result++ = calc<tsl::simd<typename SimdStyle::base_type, tsl::scalar>>(*p_data1, *p_data2);
      }
    }

   private:
    template <class PS = SimdStyle>
    typename PS::register_type calc(typename PS::register_type vals1, typename PS::register_type vals2) const {
      if constexpr (has_hint<HintSet, tuddbs::hints::arithmetic::add>) {
        return tsl::add<PS>(vals1, vals2);
      } else if constexpr (has_hint<HintSet, tuddbs::hints::arithmetic::sub>) {
        return tsl::sub<PS>(vals1, vals2);
      } else if constexpr (has_hint<HintSet, tuddbs::hints::arithmetic::mul>) {
        return tsl::mul<PS>(vals1, vals2);
      } else if constexpr (has_hint<HintSet, tuddbs::hints::arithmetic::div>) {
        return tsl::div<PS>(vals1, vals2);
      } else {
        throw std::runtime_error("No supported arithmetic operation found");
      }
    }
  };

  template <typename SimdStyle>
  using col_adder_t = tuddbs::Arithmetic<
    SimdStyle, tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::add, tuddbs::hints::intermediate::position_list>>;

  template <typename SimdStyle>
  using col_subtractor_t = tuddbs::Arithmetic<
    SimdStyle, tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::sub, tuddbs::hints::intermediate::position_list>>;

  template <typename SimdStyle>
  using col_multiplier_t = tuddbs::Arithmetic<
    SimdStyle, tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::mul, tuddbs::hints::intermediate::position_list>>;

  template <typename SimdStyle>
  using col_divider_t = tuddbs::Arithmetic<
    SimdStyle, tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::div, tuddbs::hints::intermediate::position_list>>;

  template <typename SimdStyle>
  using col_sum_t = tuddbs::Arithmetic<
    SimdStyle, tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::sum, tuddbs::hints::intermediate::position_list>>;

  template <typename SimdStyle>
  using col_average_t = tuddbs::Arithmetic<
    SimdStyle, tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::average, tuddbs::hints::intermediate::position_list>>;

  template <typename SimdStyle>
  using col_bm_sum_t =
    tuddbs::Arithmetic<SimdStyle,
                       tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::sum, tuddbs::hints::intermediate::bit_mask>>;

  template <typename SimdStyle>
  using col_bm_average_t = tuddbs::Arithmetic<
    SimdStyle, tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::average, tuddbs::hints::intermediate::bit_mask>>;
}  // namespace tuddbs

#endif
