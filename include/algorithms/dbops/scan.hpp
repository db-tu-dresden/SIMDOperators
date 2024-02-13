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
 * @file scan.hpp
 * @brief
 */

#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SCAN_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_SCAN_HPP

#include "algorithms/dbops/simdops.hpp"
#include "iterable.hpp"
#include "tslintrin.hpp"

namespace tuddbs {

  template <tsl::VectorProcessingStyle _SimdStyle, template <class, class> class CompareFun,
            typename Idof = tsl::workaround>
  class Generic_Scan_Bitmask {
   public:
    using SimdStyle = _SimdStyle;
    using DataSinkType = typename _SimdStyle::imask_type *;
    using base_type = typename SimdStyle::base_type;
    using result_base_type = typename SimdStyle::imask_type;

   private:
    using ScalarT = tsl::simd<typename SimdStyle::base_type, tsl::scalar>;
    using UnsignedSimdT = typename SimdStyle::template transform_extension<typename SimdStyle::offset_base_type>;
    using CountSimdT = typename SimdStyle::template transform_extension<size_t>;
    using CountSimdRegisterT = typename SimdStyle::template transform_type<size_t>;

   private:
    typename SimdStyle::base_type const m_predicate_scalar;
    typename SimdStyle::register_type const m_predicate_reg;
    typename SimdStyle::offset_base_register_type const m_increment;

   public:
    Generic_Scan_Bitmask(typename SimdStyle::base_type const p_predicate)
      : m_predicate_scalar(p_predicate),
        m_predicate_reg(tsl::set1<SimdStyle>(p_predicate)),
        m_increment(tsl::set1<UnsignedSimdT>(1)) {}

    ~Generic_Scan_Bitmask() = default;

   public:
    /**
     * Applies a selection Scan to the data elements in the range [p_data, p_end).
     * The Scan is applied using SIMD operations for improved performance.
     * The selected elements are stored in the data sink.
     * The valid count of selected elements is updated accordingly.
     *
     * @param p_data The pointer to the start of the data range.
     * @param p_end The pointer to the end of the data range.
     *
     * @note This function assumes that the data range is aligned for SIMD operations.
     *
     * @note The data sink must have enough capacity to store the selected elements.
     *
     * @note The valid count is the total count of selected elements.
     *
     * @note The function uses SIMD operations for improved performance.
     *
     * @note The function updates the valid count using SIMD operations.
     *
     * @note The function updates the data sink with the selected elements.
     */
    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_data,
                    SimdOpsIterableOrSizeT auto p_end) noexcept -> std::tuple<DataSinkType, size_t> {
      // Get the end of the SIMD iteration
      auto const simd_end = simd_iter_end<SimdStyle>(p_data, p_end);
      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      auto result = reinterpret_iterable<DataSinkType>(p_result);

      auto valid_count = tsl::set1<CountSimdT, Idof>(0);
      // Iterate over the data simdified and apply the Scan for equality
      for (; p_data != simd_end; p_data += SimdStyle::vector_element_count(), ++result) {
        // Load data from the source
        auto data = tsl::load<SimdStyle, Idof>(p_data);
        // Compare the data with the predicate producing a mask type (either
        // register or integral type)
        auto mask = CompareFun<SimdStyle, Idof>::apply(data, m_predicate_reg);
        // Store the result as an integral value into the data sink
        tsl::store_imask<SimdStyle, Idof>(result, tsl::to_integral<SimdStyle>(mask));

        // Increment the valid count
        auto count_increment =
          tsl::reinterpret<SimdStyle, UnsignedSimdT, Idof>(tsl::maskz_mov<SimdStyle>(mask, m_increment));
        if constexpr (sizeof(typename SimdStyle::base_type) < sizeof(size_t)) {
          for (auto const &inc : tsl::convert_up<SimdStyle, CountSimdT, Idof>(count_increment)) {
            valid_count = tsl::add<CountSimdT, Idof>(valid_count, inc);
          }
        } else {
          valid_count = tsl::add<CountSimdT, Idof>(valid_count, count_increment);
        }
      }
      // Get the simdified valid elements count
      auto scalar_valid_count = tsl::hadd<CountSimdT, Idof>(valid_count);
      if (p_data != end) {
        size_t valid_count_scalar = 0;
        typename SimdStyle::imask_type remainder_result = 0;
        // Iterate over the remainder of the data
        typename SimdStyle::imask_type shift = 0;

        for (; p_data != end; ++p_data, ++shift) {
          auto const res =
            static_cast<typename SimdStyle::imask_type>(CompareFun<ScalarT, Idof>::apply(*p_data, m_predicate_scalar));
          remainder_result |= res << shift;
          valid_count_scalar += res;
        }
        // Store the remainder result
        *result++ = remainder_result;
        // Increment the valid count
        scalar_valid_count += valid_count_scalar;
      }
      return std::make_tuple(result, scalar_valid_count);
    }

    auto merge(Generic_Scan_Bitmask const &other) noexcept -> void {}

    auto finalize() const noexcept -> void {}
  };

  template <tsl::VectorProcessingStyle _SimdStyle, template <class, class> class CompareFun,
            SimdOpsIterable _DataSinkType = typename _SimdStyle::imask_type *, typename Idof = tsl::workaround>
  class Generic_Scan_Range_Bitmask {
   public:
    using SimdStyle = _SimdStyle;
    using DataSinkType = _DataSinkType;
    using base_type = typename SimdStyle::base_type;
    using result_base_type = typename SimdStyle::imask_type;

   private:
    using ScalarT = tsl::simd<typename SimdStyle::base_type, tsl::scalar>;
    using UnsignedSimdT =
      typename SimdStyle::template transform_extension<typename SimdStyle::offset_base_register_type>;
    using CountSimdT = typename SimdStyle::template transform_extension<size_t>;
    using CountSimdRegisterT = typename SimdStyle::template transform_type<size_t>;

   private:
    typename SimdStyle::base_type const m_predicate_lower_scalar;
    typename SimdStyle::base_type const m_predicate_upper_scalar;
    typename SimdStyle::register_type const m_predicate_lower_reg;
    typename SimdStyle::register_type const m_predicate_upper_reg;
    typename SimdStyle::offset_base_register_type const m_increment;

   public:
    Generic_Scan_Range_Bitmask(typename SimdStyle::base_type const &p_predicate_lower,
                               typename SimdStyle::base_type const &p_predicate_upper)
      : m_predicate_lower_scalar(p_predicate_lower),
        m_predicate_upper_scalar(p_predicate_upper),
        m_predicate_lower_reg(tsl::set1<SimdStyle>(p_predicate_lower)),
        m_predicate_upper_reg(tsl::set1<SimdStyle>(p_predicate_upper)),
        m_increment(tsl::set1<UnsignedSimdT>(1)) {}
    ~Generic_Scan_Range_Bitmask() = default;

   public:
    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end,
                    bit_mask) noexcept -> std::tuple<DataSinkType, size_t> {
      // Get the end of the SIMD iteration
      auto const simd_end = simd_iter_end<SimdStyle>(p_data, p_end);
      // Get the end of the data
      auto const end = iter_end(p_data, p_end);
      auto valid_count = tsl::set1<CountSimdT>(0);
      // Iterate over the data simdified and apply the Scan for equality
      for (; p_data != simd_end; p_data += SimdStyle::vector_element_count(), ++p_result) {
        // Load data from the source
        auto data = tsl::load<SimdStyle>(p_data);
        // Compare the data with the predicate producing a mask type (either
        // register or integral type)
        auto mask = CompareFun<SimdStyle, Idof>::apply(data, m_predicate_lower_reg, m_predicate_upper_reg);
        // Store the result as an integral value into the data sink
        tsl::store_imask<SimdStyle>(p_result, tsl::to_integral<SimdStyle>(mask));

        // Increment the valid count
        auto count_increment = tsl::reinterpret<SimdStyle, UnsignedSimdT>(tsl::maskz_mov<SimdStyle>(mask, m_increment));
        if constexpr (sizeof(typename SimdStyle::base_type) < sizeof(size_t)) {
          for (auto const &inc : tsl::convert_up<SimdStyle, CountSimdT>(count_increment)) {
            valid_count = tsl::add<CountSimdT>(valid_count, inc);
          }
        } else {
          valid_count = tsl::add<CountSimdT>(valid_count, count_increment);
        }
      }
      // Get the simdified valid elements count
      auto scalar_valid_count = tsl::hadd<CountSimdT>(valid_count);
      if (p_data != end) {
        size_t valid_count_scalar = 0;
        typename SimdStyle::imask_type remainder_result = 0;
        // Iterate over the remainder of the data
        typename SimdStyle::imask_type shift = 0;

        for (; p_data != end; ++p_data, ++shift) {
          auto const res = static_cast<typename SimdStyle::imask_type>(
            CompareFun<ScalarT, Idof>::apply(*p_data, m_predicate_lower_scalar, m_predicate_upper_scalar));
          remainder_result |= res << shift;
          valid_count_scalar += res;
        }
        // Store the remainder result
        *p_result = remainder_result;
        // Increment the valid count
        scalar_valid_count += valid_count_scalar;
      }
      return std::make_tuple(p_result, scalar_valid_count);
    }

    auto merge(Generic_Scan_Range_Bitmask const &other) noexcept -> void {}

    auto finalize() -> void {}
  };

  template <tsl::VectorProcessingStyle SimdStyle>
  using ScanEQ_BM = Generic_Scan_Bitmask<SimdStyle, tsl::functors::equal>;
  template <tsl::VectorProcessingStyle SimdStyle>
  using ScanNEQ_BM = Generic_Scan_Bitmask<SimdStyle, tsl::functors::nequal>;
  template <tsl::VectorProcessingStyle SimdStyle>
  using ScanLT_BM = Generic_Scan_Bitmask<SimdStyle, tsl::functors::less_than>;
  template <tsl::VectorProcessingStyle SimdStyle>
  using ScanGT_BM = Generic_Scan_Bitmask<SimdStyle, tsl::functors::greater_than>;
  template <tsl::VectorProcessingStyle SimdStyle>
  using ScanLE_BM = Generic_Scan_Bitmask<SimdStyle, tsl::functors::less_than_or_equal>;
  template <tsl::VectorProcessingStyle SimdStyle>
  using ScanGE_BM = Generic_Scan_Bitmask<SimdStyle, tsl::functors::greater_than_or_equal>;

  template <tsl::VectorProcessingStyle SimdStyle>
  using ScanBWI_BM = Generic_Scan_Range_Bitmask<SimdStyle, tsl::functors::between_inclusive>;

}  // namespace tuddbs

#endif