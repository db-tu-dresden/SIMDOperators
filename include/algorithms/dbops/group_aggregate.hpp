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
 * @file group.hpp
 * @brief
 */

#ifndef SIMDOPS_INLCUDE_ALGORITHMS_DBOPS_GROUP_AGGREGATE_HPP
#define SIMDOPS_INLCUDE_ALGORITHMS_DBOPS_GROUP_AGGREGATE_HPP

#include <cassert>
#include <type_traits>

#include "algorithms/dbops/hashing.hpp"
#include "algorithms/dbops/simdops.hpp"
#include "iterable.hpp"
#include "tslintrin.hpp"

namespace tuddbs {
  template <tsl::VectorProcessingStyle _KeySimdStyle, typename _ValueType, template <class, class> class _AggregateFun,
            class HintSet = OperatorHintSet<hints::hashing::size_exp_2>, typename Idof = tsl::workaround>
  class Group_Aggregate_Binary_SIMD_Linear_Displacement {
   public:
    using KeySimdStyle = _KeySimdStyle;
    using KeyType = typename KeySimdStyle::base_type;
    using KeySinkType = KeyType *;
    using ValueType = _ValueType;
    using ValidityType = bool;
    using AggregateFun = _AggregateFun<tsl::simd<ValueType, tsl::scalar>, Idof>;

   public:
    struct Value_Entry_With_Validity {
      ValueType value = (ValueType)0;
      ValidityType valid = false;
    };

   public:
    using ValueEntryType = std::conditional_t<has_hint<HintSet, hints::hashing::keys_may_contain_zero>,
                                              Value_Entry_With_Validity, ValueType>;
    using ValueEntrySinkType = ValueEntryType *;

   private:
    KeySinkType m_key_sink;
    ValueEntrySinkType m_value_sink;

    size_t const m_map_element_count;
    size_t m_group_id_count;

    size_t const m_empty_bucket_value = 0;

   public:
    explicit Group_Aggregate_Binary_SIMD_Linear_Displacement(SimdOpsIterable auto p_key_sink,
                                                             SimdOpsIterable auto p_value_sink,
                                                             size_t p_map_element_count, bool initialize = true)
      : m_key_sink(p_key_sink),
        m_value_sink(p_value_sink),
        m_map_element_count(p_map_element_count),
        m_group_id_count(0) {
      if constexpr (has_hint<HintSet, hints::hashing::size_exp_2>) {
        assert((m_map_element_count & (m_map_element_count - 1)) == 0);
      }
      if (initialize) {
        for (auto i = 0; i < m_map_element_count; ++i) {
          m_key_sink[i] = m_empty_bucket_value;
          m_value_sink[i] = {};
        }
      }
    }

   private:
    TSL_FORCE_INLINE auto insert(typename KeySimdStyle::base_type const key, ValueType const value,
                                 typename KeySimdStyle::imask_type const all_false_mask,
                                 typename KeySimdStyle::register_type const empty_bucket_reg) noexcept -> void {
      // broadcast the key to all lanes
      auto const keys_reg = tsl::set1<KeySimdStyle, Idof>(key);
      // calculate the position hint
      auto lookup_position =
        normalizer<KeySimdStyle, HintSet, Idof>::align_value(normalizer<KeySimdStyle, HintSet, Idof>::normalize_value(
          default_hasher<KeySimdStyle, Idof>::hash_value(key), m_map_element_count));

      while (true) {
        typename KeySimdStyle::register_type map_reg;
        if constexpr (has_hint<HintSet, hints::memory::aligned>) {
          // load N values from the map
          map_reg = tsl::load<KeySimdStyle, Idof>(m_key_sink + lookup_position);
        } else {
          // load N values from the map
          map_reg = tsl::loadu<KeySimdStyle, Idof>(m_key_sink + lookup_position);
        }
        auto const key_found_mask = tsl::equal_as_imask<KeySimdStyle, Idof>(map_reg, keys_reg);
        if (tsl::nequal<KeySimdStyle, Idof>(key_found_mask, all_false_mask)) {
          auto const found_position = tsl::tzc<KeySimdStyle, Idof>(key_found_mask);
          auto &value_entry = m_value_sink[lookup_position + found_position];
          if constexpr (has_any_hint<HintSet, hints::hashing::keys_may_contain_zero>) {
            if (!value_entry.valid) {
              value_entry.valid = true;
            }
            value_entry.value = AggregateFun::apply(value_entry.value, value);
          } else {
            value_entry = AggregateFun::apply(value_entry, value);
          }

          break;
        }
        auto const empty_bucket_found_mask = tsl::equal_as_imask<KeySimdStyle, Idof>(map_reg, empty_bucket_reg);
        if (tsl::nequal<KeySimdStyle, Idof>(key_found_mask, all_false_mask)) {
          auto const found_position = tsl::tzc<KeySimdStyle, Idof>(key_found_mask);
          auto &value_entry = m_value_sink[lookup_position + found_position];
          if constexpr (has_any_hint<HintSet, hints::hashing::keys_may_contain_zero>) {
            value_entry.valid = true;
            value_entry.value = AggregateFun::apply(value_entry.value, value);
          } else {
            value_entry = AggregateFun::apply(value_entry, value);
          }
          break;
        }
        lookup_position = normalizer<KeySimdStyle, HintSet, Idof>::normalize_value(
          lookup_position + KeySimdStyle::vector_element_count(), m_map_element_count);
      }
    }

   public:
    auto operator()(SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end,
                    SimdOpsIterable auto p_value) noexcept -> void {
      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      auto const all_false_mask = tsl::integral_all_false<KeySimdStyle, Idof>();
      auto const empty_bucket_reg = tsl::set1<KeySimdStyle, Idof>(m_empty_bucket_value);

      for (; p_data != end; ++p_data, ++p_value) {
        insert(*p_data, *p_value, all_false_mask, empty_bucket_reg);
      }
    }

    auto operator()(SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end, SimdOpsIterable auto p_valid_masks,
                    SimdOpsIterable auto p_value) noexcept -> void {
      // Get the end of the SIMD iteration
      auto const simd_end = simd_iter_end<KeySimdStyle>(p_data, p_end);
      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      auto valid_masks = reinterpret_iterable<typename KeySimdStyle::imask_type>(p_valid_masks);
      auto const all_false_mask = tsl::integral_all_false<KeySimdStyle, Idof>();
      auto const empty_bucket_reg = tsl::set1<KeySimdStyle, Idof>(m_empty_bucket_value);

      for (; p_data != simd_end; p_data += KeySimdStyle::vector_element_count(), ++valid_masks,
                                 p_value += KeySimdStyle::vector_element_count()) {
        auto valid_mask = tsl::load_mask<KeySimdStyle, Idof>(valid_masks);
        for (size_t i = 0; i < KeySimdStyle::vector_element_count(); ++i) {
          if (tsl::test_mask<KeySimdStyle, Idof>(valid_mask, i)) {
            insert(p_data[i], p_value[i], all_false_mask, empty_bucket_reg);
          }
        }
      }
      if (p_data != end) {
        auto valid_mask = tsl::load_mask<KeySimdStyle, Idof>(valid_masks);
        int i = 0;
        for (; p_data != end; ++p_data, ++i, ++p_value) {
          if (tsl::test_mask<KeySimdStyle, Idof>(valid_mask, i)) {
            insert(*p_data, *p_value, all_false_mask, empty_bucket_reg);
          }
        }
      }
    }

    template <class HS = HintSet, enable_if_has_hint_t<HS, hints::intermediate::dense_bit_mask>>
    auto operator()(SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end, SimdOpsIterable auto p_valid_masks,
                    SimdOpsIterable auto p_value) noexcept -> void {
      constexpr auto const bits_per_mask = sizeof(typename KeySimdStyle::imask_type) * CHAR_BIT;
      // Get the end of the SIMD iteration
      auto const batched_end_end = batched_iter_end<bits_per_mask>(p_data, p_end);
      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      auto valid_masks = reinterpret_iterable<typename KeySimdStyle::imask_type>(p_valid_masks);
      auto const all_false_mask = tsl::integral_all_false<KeySimdStyle, Idof>();
      auto const empty_bucket_reg = tsl::set1<KeySimdStyle, Idof>(m_empty_bucket_value);

      for (; p_data != batched_end_end; p_data += bits_per_mask, ++valid_masks, p_value += bits_per_mask) {
        auto valid_mask = tsl::load_mask<KeySimdStyle, Idof>(valid_masks);
        for (size_t i = 0; i < bits_per_mask; ++i) {
          if (tsl::test_mask<KeySimdStyle, Idof>(valid_mask, i)) {
            insert(p_data[i], p_value[i], all_false_mask, empty_bucket_reg);
          }
        }
      }
      if (p_data != end) {
        auto valid_mask = tsl::load_mask<KeySimdStyle, Idof>(valid_masks);
        int i = 0;
        for (; p_data != end; ++p_data, ++i, ++p_value) {
          if (tsl::test_mask<KeySimdStyle, Idof>(valid_mask, i)) {
            insert(*p_data, *p_value, all_false_mask, empty_bucket_reg);
          }
        }
      }
    }

    auto merge(Group_Aggregate_Binary_SIMD_Linear_Displacement const &other) noexcept -> void {
      auto const all_false_mask = tsl::integral_all_false<KeySimdStyle, Idof>();
      auto const empty_bucket_reg = tsl::set1<KeySimdStyle, Idof>(m_empty_bucket_value);

      auto const &other_key_sink = other.m_key_sink;
      auto const &other_value_sink = other.m_value_sink;
      for (size_t i = 0; i < m_map_element_count; ++i) {
        if constexpr (has_hint<HintSet, hints::hashing::keys_may_contain_zero>) {
          auto const &value_entry = other_value_sink[i];
          if (value_entry.valid) {
            insert(other_key_sink[i], value_entry.value, all_false_mask, empty_bucket_reg);
          }
        } else {
          auto const other_key = other_key_sink[i];
          if (other_key != m_empty_bucket_value) {
            insert(other_key, other_value_sink[i], all_false_mask, empty_bucket_reg);
          }
        }
      }
    }

    auto finalize() const noexcept -> void {}
  };

}  // namespace tuddbs
#endif