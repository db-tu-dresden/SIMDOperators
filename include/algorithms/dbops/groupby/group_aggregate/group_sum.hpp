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
 * @file groupby_sum.hpp
 * @brief
 */

#ifndef SIMDOPS_INLCUDE_ALGORITHMS_DBOPS_GROUPBY_AGGREGATE_GROUPBY_SUM_HPP
#define SIMDOPS_INLCUDE_ALGORITHMS_DBOPS_GROUPBY_AGGREGATE_GROUPBY_SUM_HPP

#include <cassert>
#include <type_traits>

#include "algorithms/dbops/dbops_hints.hpp"
#include "algorithms/utils/hashing.hpp"
#include "iterable.hpp"
#include "tslintrin.hpp"

namespace tuddbs {
  template <tsl::VectorProcessingStyle _KeySimdStyle, tsl::TSLArithmetic _ValueType = typename _KeySimdStyle::base_type,
            class HintSet = OperatorHintSet<hints::hashing::size_exp_2>, typename Idof = tsl::workaround>
  class Grouper_Aggregate_Sum_Build_Hash_SIMD_Linear_Displacement {
   public:
    using KeySimdStyle = _KeySimdStyle;
    using KeyType = typename KeySimdStyle::base_type;
    using KeySinkType = KeyType *;
    using ValueType = _ValueType;
    using ValueSinkType = ValueType *;
    using AddSimdStyle = tsl::simd<_ValueType, tsl::scalar>;

   private:
    KeySinkType m_key_sink;
    ValueSinkType m_value_sink;

    size_t const m_map_element_count;
    size_t m_groups_count;

    KeyType const m_empty_bucket_value;
    bool m_empty_bucket_seen_in_keys = false;

   public:
    auto distinct_key_count() const noexcept {
      if constexpr (has_hint<HintSet, hints::hashing::keys_may_contain_zero>) {
        return m_empty_bucket_seen_in_keys ? m_groups_count + 1 : m_groups_count;
      } else {
        return m_groups_count;
      }
    }
    auto empty_bucket_value() const noexcept { return m_empty_bucket_value; }

   public:
    explicit Grouper_Aggregate_Sum_Build_Hash_SIMD_Linear_Displacement(void) = delete;

    explicit Grouper_Aggregate_Sum_Build_Hash_SIMD_Linear_Displacement(SimdOpsIterable auto p_key_sink,
                                                                       SimdOpsIterable auto p_value_sink,
                                                                       size_t p_map_element_count,
                                                                       KeyType p_empty_bucket_value = 0,
                                                                       bool initialize = true)
      : m_key_sink(p_key_sink),
        m_value_sink(reinterpret_iterable<ValueSinkType>(p_value_sink)),
        m_map_element_count(p_map_element_count),
        m_groups_count(0),
        m_empty_bucket_value(p_empty_bucket_value) {
      if constexpr (has_hint<HintSet, hints::hashing::size_exp_2>) {
        assert((m_map_element_count & (m_map_element_count - 1)) == 0);
      }
      if (initialize) {
        for (auto i = 0; i < m_map_element_count; ++i) {
          m_key_sink[i] = m_empty_bucket_value;
          m_value_sink[i] = (ValueType)0;
        }
      }
    }

    ~Grouper_Aggregate_Sum_Build_Hash_SIMD_Linear_Displacement() = default;

   private:
    TSL_FORCE_INLINE auto insert(typename KeySimdStyle::base_type const key, ValueType const value,
                                 typename KeySimdStyle::imask_type const all_false_mask,
                                 typename KeySimdStyle::register_type const empty_bucket_reg) noexcept -> void {
      if constexpr (has_hint<HintSet, hints::hashing::keys_may_contain_zero>) {
        if (key == 0) {
          m_empty_bucket_seen_in_keys = true;
        }
      }

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
        // compare the current key with the N values, if the key is found, the mask will contain a 1 bit at the position
        // of the found key
        auto const key_found_mask = tsl::equal_as_imask<KeySimdStyle, Idof>(map_reg, keys_reg);
        if (tsl::nequal<KeySimdStyle, Idof>(key_found_mask, all_false_mask)) {
          auto const found_position = tsl::tzc<KeySimdStyle, Idof>(key_found_mask);
          auto &value_entry = m_value_sink[lookup_position + found_position];
          value_entry = tsl::add<AddSimdStyle, Idof>(value_entry, value);
          break;
        }
        auto const empty_bucket_found_mask = tsl::equal_as_imask<KeySimdStyle, Idof>(map_reg, empty_bucket_reg);
        if (tsl::nequal<KeySimdStyle, Idof>(empty_bucket_found_mask, all_false_mask)) {
          auto const empty_bucket_position = tsl::tzc<KeySimdStyle, Idof>(empty_bucket_found_mask);

          if constexpr (has_any_hint<HintSet, hints::hashing::keys_may_contain_zero>) {
            // At this position we know that key != m_empty_bucket_value, otherwise, we would have found an empty bucket
            // and breaked out of the loop before. However, it could be the case that if the keys contain zero, that we
            // found an "empty" bucket that isn't empty. Consequently, we check the corresponding value. If it is 0, we
            // can use the bucket, otherwise we have to continue the search. This behavior may lead to a 'formal'
            // override of the key that is equal to empty_bucket_value, but if the corresponding vlaue is 0 we don't
            // loose any information.
            auto &value_entry = m_value_sink[lookup_position + empty_bucket_position];
            if (value_entry == 0) {
              m_key_sink[lookup_position + empty_bucket_position] = key;
              value_entry = value;
              ++m_groups_count;
              break;
            } else {
              auto updated_empty_bucket_found_mask =
                tsl::shift_right<KeySimdStyle, false, Idof>(empty_bucket_found_mask, empty_bucket_position + 1);
              if (tsl::nequal<KeySimdStyle, Idof>(updated_empty_bucket_found_mask, all_false_mask)) {
                // As there can be only a single occurence of the key that equals an empty bucket, we only have to use
                // the first occurence
                auto updated_empty_bucket_position =
                  tsl::tzc<KeySimdStyle, Idof>(updated_empty_bucket_found_mask) + empty_bucket_position + 1;
                m_key_sink[lookup_position + updated_empty_bucket_position] = key;
                m_value_sink[lookup_position + updated_empty_bucket_position] = value;
                ++m_groups_count;
                break;
              }
            }
          } else {
            m_key_sink[lookup_position + empty_bucket_position] = key;
            m_value_sink[lookup_position + empty_bucket_position] = value;
            ++m_groups_count;
            break;
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

    template <tsl::VectorProcessingStyle OtherSimdStlye, tsl::TSLArithmetic OtherValueType, class OtherHintSet,
              typename OtherIdof>
    auto merge(Grouper_Aggregate_Sum_Build_Hash_SIMD_Linear_Displacement<OtherSimdStlye, OtherValueType, OtherHintSet,
                                                                         OtherIdof> const &other) noexcept -> void {
      auto const all_false_mask = tsl::integral_all_false<KeySimdStyle, Idof>();
      auto const empty_bucket_reg = tsl::set1<KeySimdStyle, Idof>(m_empty_bucket_value);

      auto const other_empty_bucket_value = other.empty_bucket_value();
      auto const &other_key_sink = other.m_key_sink;
      auto const &other_value_sink = other.m_value_sink;
      for (size_t i = 0; i < m_map_element_count; ++i) {
        if constexpr (has_hint<HintSet, hints::hashing::keys_may_contain_zero>) {
          auto const other_key = other_key_sink[i];
          if (other_key == other_empty_bucket_value) {
            auto const value_entry = other_value_sink[i];
            if (value_entry != 0) {
              insert(other_key, value_entry, all_false_mask, empty_bucket_reg);
            }
          } else {
            insert(other_key, other_value_sink[i], all_false_mask, empty_bucket_reg);
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

  template <tsl::VectorProcessingStyle _KeySimdStyle, typename _ValueType = typename _KeySimdStyle::base_type,
            class HintSet = OperatorHintSet<>, typename Idof = tsl::workaround>
  class Grouper_Aggregate_Sum_Hash_SIMD_Linear_Displacement {
   public:
    using KeySimdStyle = _KeySimdStyle;
    using KeyType = typename KeySimdStyle::base_type;
    using KeySinkType = KeyType *;
    using ValueType = _ValueType;
    using ValueSinkType = ValueType *;

   private:
    KeySinkType m_key_sink;
    ValueSinkType m_value_sink;

    size_t const m_map_element_count;
    KeyType const m_empty_bucket_value;
    bool m_empty_bucket_seen_in_keys = false;

   public:
    explicit Grouper_Aggregate_Sum_Hash_SIMD_Linear_Displacement(void) = delete;
    explicit Grouper_Aggregate_Sum_Hash_SIMD_Linear_Displacement(SimdOpsIterable auto p_key_sink,
                                                                 SimdOpsIterable auto p_value_sink,
                                                                 size_t p_map_element_count,
                                                                 KeyType p_empty_bucket_value = 0) noexcept
      : m_key_sink(p_key_sink),
        m_value_sink(reinterpret_iterable<ValueSinkType>(p_value_sink)),
        m_map_element_count(p_map_element_count),
        m_empty_bucket_value(p_empty_bucket_value) {}

    ~Grouper_Aggregate_Sum_Hash_SIMD_Linear_Displacement() = default;

   public:
    auto operator()(SimdOpsIterable auto p_group_key, SimdOpsIterable auto p_group_value) noexcept -> void {
      for (size_t i = 0; i < m_map_element_count; ++i) {
        if constexpr (has_hint<HintSet, hints::hashing::keys_may_contain_zero>) {
          if (m_key_sink[i] != m_empty_bucket_value) {
            *p_group_key = m_key_sink[i];
            *p_group_value = m_value_sink[i];
            ++p_group_key;
            ++p_group_value;
          } else {
            if (m_value_sink[i] != 0) {
              *p_group_key = m_key_sink[i];
              *p_group_value = m_value_sink[i];
              ++p_group_key;
              ++p_group_value;
              m_empty_bucket_seen_in_keys = true;
            }
          }
        } else {
          if (m_key_sink[i] != m_empty_bucket_value) {
            *p_group_key = m_key_sink[i];
            *p_group_value = m_value_sink[i];
            ++p_group_key;
            ++p_group_value;
          }
        }
      }
      if constexpr (has_hint<HintSet, hints::hashing::keys_may_contain_zero>) {
        if (!m_empty_bucket_seen_in_keys) {
          *p_group_key = 0;
          *p_group_value = 0;
        }
      }
    }
  };
  template <tsl::VectorProcessingStyle _SimdStyle, tsl::TSLArithmetic _ValueType = typename _SimdStyle::base_type,
            class HintSet = OperatorHintSet<hints::hashing::size_exp_2>, typename Idof = tsl::workaround>
  struct Grouper_Aggregate_SUM_SIMD_Linear_Displacement {
    using builder_t = Grouper_Aggregate_Sum_Build_Hash_SIMD_Linear_Displacement<_SimdStyle, _ValueType, HintSet, Idof>;
    using grouper_t = Grouper_Aggregate_Sum_Hash_SIMD_Linear_Displacement<_SimdStyle, _ValueType, HintSet, Idof>;
  };

}  // namespace tuddbs
#endif