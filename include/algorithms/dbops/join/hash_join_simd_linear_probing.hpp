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
 * @file hash_join_simd_linear_probing.hpp
 * @brief
 */

#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_JOIN_HASH_JOIN_SIMD_LINEAR_PROBING_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_JOIN_HASH_JOIN_SIMD_LINEAR_PROBING_HPP

#include <cassert>
#include <climits>
#include <type_traits>
#include <set>

#include "algorithms/dbops/dbops_hints.hpp"
#include "algorithms/dbops/join/hash_join_hints.hpp"
#include "algorithms/utils/hashing.hpp"
#include "iterable.hpp"
// #include "static/utils/type_concepts.hpp"
#include "tsl.hpp"

namespace tuddbs {

  /**
   * @brief A hash table implementation for grouping elements using SIMD and linear displacement.
   *
   * This class provides a hash table implementation for grouping elements using SIMD (Single Instruction, Multiple
   * Data) and linear displacement. It is designed to efficiently group elements based on their keys using a hash
   * function and linear probing to handle collisions. The class supports different SIMD processing styles and allows
   * customization through template parameters.
   *
   * @tparam _SimdStyle The SIMD processing style to use. This should be a type that implements the necessary SIMD
   * operations and provides a base type for storing keys and group IDs.
   * @tparam HintSet The set of operator hints to use for optimizing the hash table. This should be a type that provides
   * hints for hashing, such as the desired size of the hash table.
   * @tparam Idof The type used for indexing into the hash table. This should be a type that supports indexing
   * operations.
   */
  template <tsl::VectorProcessingStyle _SimdStyle, tsl::TSLArithmetic _PositionType = size_t,
            class HintSet = OperatorHintSet<hints::hashing::size_exp_2>, typename Idof = tsl::workaround>
  class Hash_Join_Build_SIMD_Linear_Probing {
   public:
    using SimdStyle = _SimdStyle;
    using KeyType = typename SimdStyle::base_type;
    using KeySinkType = KeyType *;

    using PositionType = _PositionType;
    using PositionSinkType = PositionType *;

    using BucketUsedType = typename SimdStyle::base_type;
    using BucketUsedSinkType = BucketUsedType *;

   private:
    KeySinkType m_key_sink;                      // keys
    PositionSinkType m_original_positions_sink;  // values
    BucketUsedSinkType m_used_bucket_sink;       // indicator for empty buckets

    size_t const m_bucket_count;  // fixed count of map size // table size
    size_t m_used_bucket_count;   // count of how many buckets are used

    KeyType const m_empty_bucket_value;         // empty bucket indicator
    PositionType const m_invalid_position;      //
    BucketUsedType const m_bucket_empty = 0x0;  // empty bucket indicator
    BucketUsedType const m_bucket_full = 0x1;   // indicator for a full bucket

   public:
    auto distinct_key_count() const noexcept { return m_used_bucket_count; }
    auto empty_bucket_value() const noexcept { return m_empty_bucket_value; }
    auto invalid_position() const noexcept { return m_invalid_position; }
    auto empty_bucket_indicator() const noexcept { return m_bucket_empty; }
    auto full_bucket_indicator() const noexcept { return m_bucket_full; }

   public:
    explicit Hash_Join_Build_SIMD_Linear_Probing(void) = delete;

    explicit Hash_Join_Build_SIMD_Linear_Probing(
      SimdOpsIterable auto p_key_sink, SimdOpsIterable auto p_used_bucket_sink, SimdOpsIterable auto p_position_sink,
      size_t p_map_element_count, KeyType p_empty_bucket_value = 0,
      PositionType p_invalid_position = std::numeric_limits<PositionType>::max(), bool initialize = true)
      : m_key_sink(reinterpret_iterable<KeySinkType>(p_key_sink)),
        m_original_positions_sink(reinterpret_iterable<PositionSinkType>(p_position_sink)),
        m_used_bucket_sink(reinterpret_iterable<BucketUsedSinkType>(p_used_bucket_sink)),
        m_bucket_count(p_map_element_count),
        m_used_bucket_count(0),
        m_empty_bucket_value(p_empty_bucket_value),
        m_invalid_position(p_invalid_position) {
      if constexpr (has_hint<HintSet, hints::hashing::size_exp_2>) {
        assert((m_bucket_count & (m_bucket_count - 1)) == 0);
      }
      if (initialize) {
        for (size_t i = 0; i < m_bucket_count; ++i) {
          m_key_sink[i] = m_empty_bucket_value;
          m_used_bucket_sink[i] = m_bucket_empty;
          m_original_positions_sink[i] = m_invalid_position;
        }
      }
    }

    auto getOrderedSet() {
      std::set<std::pair<KeyType,PositionType>> mySet;
      size_t pos = 0;
      for ( auto it = m_used_bucket_sink; it != m_used_bucket_sink + m_bucket_count; ++it, ++pos) {
        if ( *it != m_bucket_empty ) {
          const auto key = m_key_sink[pos];
          const auto val = m_original_positions_sink[pos];
          mySet.emplace( key, val );
        }
      }
      return mySet;
    }

    template <tsl::VectorProcessingStyle OtherSimdStlye, typename OtherPositionType, class OtherHintSet,
              typename OtherIdof, typename HS = HintSet, enable_if_has_hint_t<HS, hints::hashing::is_hull_for_merging>>
    explicit Hash_Join_Build_SIMD_Linear_Probing(
      SimdOpsIterable auto p_key_sink, SimdOpsIterable auto p_bucket_used_sink, SimdOpsIterable auto p_position_sink,
      size_t p_map_element_count,
      Hash_Join_Build_SIMD_Linear_Probing<OtherSimdStlye, OtherPositionType, OtherHintSet, OtherIdof> const &other)
      : m_key_sink(reinterpret_iterable<KeySinkType>(p_key_sink)),
        m_used_bucket_sink(reinterpret_iterable<BucketUsedSinkType>(p_bucket_used_sink)),
        m_original_positions_sink(reinterpret_iterable<PositionSinkType>(p_position_sink)),
        m_bucket_count(p_map_element_count),
        m_used_bucket_count(0),
        m_empty_bucket_value(other.empty_bucket_value()),
        m_invalid_position(other.invalid_position()) {
      if constexpr (has_hint<HintSet, hints::hashing::size_exp_2>) {
        assert((m_bucket_count & (m_bucket_count - 1)) == 0);
      }
      for (auto i = 0; i < m_bucket_count; ++i) {
        m_key_sink[i] = m_empty_bucket_value;
        m_used_bucket_sink[i] = m_bucket_empty;
        m_original_positions_sink[i] = m_invalid_position;
      }

      merge(other);
    }
    ~Hash_Join_Build_SIMD_Linear_Probing() = default;

   private:
    TSL_FORCE_INLINE auto probe_position(typename SimdStyle::base_type const key) {
      auto const keys_reg = tsl::set1<SimdStyle, Idof>(key);
      auto const all_false_mask = tsl::integral_all_false<SimdStyle, Idof>();
      auto lookup_position =
        normalizer<SimdStyle, HintSet, Idof>::align_value(normalizer<SimdStyle, HintSet, Idof>::normalize_value(
          default_hasher<SimdStyle, Idof>::hash_value(key), m_bucket_count));
      typename SimdStyle::register_type map_reg;
      typename SimdStyle::register_type bucket_used;
      int64_t lookup_position_helper;

      if constexpr (has_hint<HintSet,
                             hints::hash_join::keys_may_contain_empty_indicator>) {  // if the data does include the
                                                                                     // empty position indicator
        typename SimdStyle::register_type empty_reg = tsl::set1<SimdStyle, Idof>(m_bucket_empty);
        while (true) {
          lookup_position_helper = lookup_position + SimdStyle::vector_element_count() - m_bucket_count;

          if (lookup_position_helper < 1) {
            if (has_hint<HintSet, hints::memory::aligned>) {
              map_reg = tsl::load<SimdStyle, Idof>(m_key_sink + lookup_position);
              bucket_used = tsl::load<SimdStyle, Idof>(m_used_bucket_sink + lookup_position);
            } else {
              map_reg = tsl::loadu<SimdStyle, Idof>(m_key_sink + lookup_position);
              bucket_used = tsl::loadu<SimdStyle, Idof>(m_used_bucket_sink + lookup_position);
            }
          } else {
            lookup_position -= lookup_position_helper;
            map_reg = tsl::loadu<SimdStyle, Idof>(m_key_sink + lookup_position);
            bucket_used = tsl::loadu<SimdStyle, Idof>(m_used_bucket_sink + lookup_position);
          }

          // auto const key_found_mask = tsl::equal_as_imask<SimdStyle, Idof>(map_reg, keys_reg);
          auto const empty_found_mask = tsl::equal_as_imask<SimdStyle, Idof>(bucket_used, empty_reg);
          // auto const key_found = tsl::mask_binary_and<SimdStyle, Idof>(
            // key_found_mask, tsl::mask_binary_not<SimdStyle, Idof>(empty_found_mask));

          // if (tsl::nequal<SimdStyle, Idof>(key_found, all_false_mask)) {  // key found
            // auto const found_position = tsl::tzc<SimdStyle, Idof>(key_found);
            // return std::make_pair(lookup_position + found_position, true);
          // } else 
          if (tsl::nequal<SimdStyle, Idof>(empty_found_mask, all_false_mask)) {  // empty place found
            auto const found_position = tsl::tzc<SimdStyle, Idof>(empty_found_mask);
            return std::make_pair(lookup_position + found_position, false);
          } else {  // move to the next probing location
            lookup_position = normalizer<SimdStyle, HintSet, Idof>::normalize_value(
              lookup_position + SimdStyle::vector_element_count(), m_bucket_count);
          }
        }
      } else {  // if the data doesn't include the empty position key as a value we can simplify the probing
        auto const key_reg_empty = tsl::set1<SimdStyle, Idof>(m_empty_bucket_value);
        while (true) {
          lookup_position_helper = lookup_position + SimdStyle::vector_element_count() - m_bucket_count;

          if (lookup_position_helper < 1) {
            if (has_hint<HintSet, hints::memory::aligned>) {
              map_reg = tsl::load<SimdStyle, Idof>(m_key_sink + lookup_position);
            } else {
              map_reg = tsl::loadu<SimdStyle, Idof>(m_key_sink + lookup_position);
            }
          } else {
            lookup_position -= lookup_position_helper;
            map_reg = tsl::loadu<SimdStyle, Idof>(m_key_sink + lookup_position);
          }

          auto const key_found_mask = tsl::equal_as_imask<SimdStyle, Idof>(map_reg, keys_reg);
          auto const empty_found_mask = tsl::equal_as_imask<SimdStyle, Idof>(map_reg, key_reg_empty);

          if (tsl::nequal<SimdStyle, Idof>(key_found_mask, all_false_mask)) {  // key found
            auto const found_position = tsl::tzc<SimdStyle, Idof>(key_found_mask);
            return std::make_pair(lookup_position + found_position, true);
          } else if (tsl::nequal<SimdStyle, Idof>(empty_found_mask, all_false_mask)) {  // empty place found
            auto const found_position = tsl::tzc<SimdStyle, Idof>(empty_found_mask);
            return std::make_pair(lookup_position + found_position, false);
          } else {  // move to the next probing location
            lookup_position = normalizer<SimdStyle, HintSet, Idof>::normalize_value(
              lookup_position + SimdStyle::vector_element_count(), m_bucket_count);
          }
        }
      }
    }

    TSL_FORCE_INLINE auto insert(typename SimdStyle::base_type const key, PositionType const key_position_in_data,
                                 size_t position) {
      if constexpr (has_hint<HintSet, hints::hash_join::global_first_occurence_required>) {
        if (key_position_in_data < m_original_positions_sink[position]) {
          m_used_bucket_count += (m_used_bucket_sink[position] == m_bucket_empty);
          m_original_positions_sink[position] = key_position_in_data;
          m_key_sink[position] = key;
          m_used_bucket_sink[position] = m_bucket_full;
        }
      } else {
        m_used_bucket_count += (m_used_bucket_sink[position] == m_bucket_empty);
        m_original_positions_sink[position] = key_position_in_data;
        m_key_sink[position] = key;
        m_used_bucket_sink[position] = m_bucket_full;
      }
    }

    TSL_FORCE_INLINE auto single_insert(typename SimdStyle::base_type const key, size_t position) {
      auto lookup_position = probe_position(key);

      insert(key, position, lookup_position.first);
    }

   public:
    static auto calculate_bucket_count(size_t const key_count,
                                       float const max_load = 0.6f) noexcept -> std::tuple<size_t, size_t> {
      auto key_sink_min_size = (size_t)((float)key_count * (1.0f + max_load));
      if constexpr (has_hint<HintSet, hints::hashing::size_exp_2>) {
        auto const bucket_count = (1 << (int)std::ceil(std::log2(key_sink_min_size)));
        auto const empty_bucket_bitset_count = bucket_count >> 6;
        return std::make_tuple(bucket_count, empty_bucket_bitset_count);
      } else {
        auto const mutliple_of_64 = (key_sink_min_size + 63) & ~63;
        return std::make_tuple(key_sink_min_size, mutliple_of_64 >> 6);
      }
    }

    auto get_used_bucket_count() const noexcept -> size_t { return m_used_bucket_count; }

    auto operator()(SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end,
                    PositionType start_position = 0) noexcept -> size_t {
      auto const end = iter_end(p_data, p_end);
      auto iter = p_data;
      size_t insertion_count = 0;
      for (; p_data != end && m_used_bucket_count < m_bucket_count; ++p_data, ++start_position) {
        auto key = *p_data;
        single_insert(key, start_position);
        insertion_count++;
      }
      return insertion_count;
    }

    template <tsl::VectorProcessingStyle OtherSimdStlye, tsl::TSLArithmetic OtherPositionType, class OtherHintSet,
              typename OtherIdof>
    auto merge(Hash_Join_Build_SIMD_Linear_Probing<OtherSimdStlye, OtherPositionType, OtherHintSet, OtherIdof> const
                 &other) noexcept -> bool {
      auto const other_full_bucket_indicator = other.full_bucket_indicator();
      auto const &other_bucket_used_sink = other.m_used_bucket_sink;
      auto const &other_key_sink = other.m_key_sink;
      auto const &other_position_sink = other.m_original_positions_sink;
      if (other.get_used_bucket_count() + m_used_bucket_count > m_bucket_count) {
        return false;
      }
      for (auto i = 0; i < other.m_bucket_count; ++i) {
        auto const occupation = other_bucket_used_sink[i];
        if (occupation == other_full_bucket_indicator) {
          auto const key = other_key_sink[i];
          auto original_key_position = other_position_sink[i];
          single_insert(key, original_key_position);
        }
      }
      return true;
    }

    auto finalize() const noexcept -> void {}
  };

  template <tsl::VectorProcessingStyle _SimdStyle, tsl::TSLArithmetic _PositionType = size_t,
            class HintSet = OperatorHintSet<hints::hashing::size_exp_2>, typename Idof = tsl::workaround>
  class Hash_Join_Probe_SIMD_Linear_Probing {
   public:
    using SimdStyle = _SimdStyle;
    using KeyType = typename SimdStyle::base_type;
    using KeySinkType = KeyType *;

    using PositionType = _PositionType;
    using PositionSinkType = PositionType *;

    using BucketUsedType = typename SimdStyle::base_type;
    using BucketUsedSinkType = BucketUsedType *;

   private:
    KeySinkType m_key_sink;
    PositionSinkType m_original_positions_sink;  // values
    BucketUsedSinkType m_used_bucket_sink;       // indicator for empty buckets

    size_t const m_bucket_count;

    KeyType const m_empty_bucket_value;        // empty bucket indicator
    BucketUsedType const m_bucket_empty = 0;   // empty bucket indicator
    BucketUsedType const m_bucket_full = 0xF;  // indicator for a full bucket

   public:
    explicit Hash_Join_Probe_SIMD_Linear_Probing(SimdOpsIterable auto p_key_sink,
                                                 SimdOpsIterable auto p_used_bucket_sink,
                                                 SimdOpsIterable auto p_position_sink, size_t p_map_element_count,
                                                 KeyType p_empty_bucket_value = 0)
      : m_key_sink(reinterpret_iterable<KeySinkType>(p_key_sink)),
        m_original_positions_sink(reinterpret_iterable<PositionSinkType>(p_position_sink)),
        m_used_bucket_sink(reinterpret_iterable<BucketUsedSinkType>(p_used_bucket_sink)),
        m_bucket_count(p_map_element_count),
        m_empty_bucket_value(p_empty_bucket_value) {
      if constexpr (has_hint<HintSet, hints::hashing::size_exp_2>) {
        assert((m_bucket_count & (m_bucket_count - 1)) == 0);
      }
    }
    ~Hash_Join_Probe_SIMD_Linear_Probing() = default;

   private:
    TSL_FORCE_INLINE auto lookup(typename SimdStyle::base_type const key,
                                 bool &not_found) const noexcept -> PositionType {
      auto const keys_reg = tsl::set1<SimdStyle, Idof>(key);
      auto const all_false_mask = tsl::integral_all_false<SimdStyle, Idof>();
      auto lookup_position =
        normalizer<SimdStyle, HintSet, Idof>::align_value(normalizer<SimdStyle, HintSet, Idof>::normalize_value(
          default_hasher<SimdStyle, Idof>::hash_value(key), m_bucket_count));

      typename SimdStyle::register_type map_reg;
      typename SimdStyle::register_type bucket_used;
      // TODO!! change to fit with the lookup objectiv!
      int64_t lookup_position_helper;
      if (has_hint<HintSet, hints::hash_join::keys_may_contain_empty_indicator>) {
        typename SimdStyle::register_type empty_reg = tsl::set1<SimdStyle, Idof>(m_bucket_empty);
        while (true) {
          lookup_position_helper = lookup_position + SimdStyle::vector_element_count() - m_bucket_count;

          if (lookup_position_helper < 1) {
            if (has_hint<HintSet, hints::memory::aligned>) {
              map_reg = tsl::load<SimdStyle, Idof>(m_key_sink + lookup_position);
              bucket_used = tsl::load<SimdStyle, Idof>(m_used_bucket_sink + lookup_position);
            } else {
              map_reg = tsl::loadu<SimdStyle, Idof>(m_key_sink + lookup_position);
              bucket_used = tsl::loadu<SimdStyle, Idof>(m_used_bucket_sink + lookup_position);
            }
          } else {
            lookup_position -= lookup_position_helper;
            map_reg = tsl::loadu<SimdStyle, Idof>(m_key_sink + lookup_position);
            bucket_used = tsl::loadu<SimdStyle, Idof>(m_used_bucket_sink + lookup_position);
          }

          auto const key_found_mask = tsl::equal_as_imask<SimdStyle, Idof>(map_reg, keys_reg);
          auto const empty_found_mask = tsl::equal_as_imask<SimdStyle, Idof>(bucket_used, empty_reg);
          auto const key_found = tsl::mask_binary_and<SimdStyle, Idof>(
            key_found_mask, tsl::mask_binary_not<SimdStyle, Idof>(empty_found_mask));

          if (tsl::nequal<SimdStyle, Idof>(key_found, all_false_mask)) {  // key found
            auto const found_position = tsl::tzc<SimdStyle, Idof>(key_found);
            not_found = false;
            return m_original_positions_sink[lookup_position + found_position];
          } else if (tsl::nequal<SimdStyle, Idof>(empty_found_mask, all_false_mask)) {  // empty place found
            not_found = true;  // useing this as a additional back channel to communicate if a value was found or not.
            return 0;
          } else {  // move to the next probing location
            lookup_position = normalizer<SimdStyle, HintSet, Idof>::normalize_value(
              lookup_position + SimdStyle::vector_element_count(), m_bucket_count);
          }
        }
      } else {  // if the data doesn't include the empty position key as a value we can simplify the probing
        auto empty_reg = tsl::set1<SimdStyle, Idof>(m_empty_bucket_value);
        while (true) {
          lookup_position_helper = lookup_position + SimdStyle::vector_element_count() - m_bucket_count;

          if (lookup_position_helper < 1) {
            if (has_hint<HintSet, hints::memory::aligned>) {
              map_reg = tsl::load<SimdStyle, Idof>(m_key_sink + lookup_position);
            } else {
              map_reg = tsl::loadu<SimdStyle, Idof>(m_key_sink + lookup_position);
            }
          } else {
            lookup_position -= lookup_position_helper;
            map_reg = tsl::loadu<SimdStyle, Idof>(m_key_sink + lookup_position);
          }

          auto const key_found_mask = tsl::equal_as_imask<SimdStyle, Idof>(map_reg, keys_reg);
          auto const empty_found_mask = tsl::equal_as_imask<SimdStyle, Idof>(map_reg, empty_reg);

          if (tsl::nequal<SimdStyle, Idof>(key_found_mask, all_false_mask)) {  // key found
            auto const found_position = tsl::tzc<SimdStyle, Idof>(key_found_mask);
            not_found = false;
            return m_original_positions_sink[lookup_position + found_position];
          } else if (tsl::nequal<SimdStyle, Idof>(empty_found_mask, all_false_mask)) {  // empty place found
            not_found = true;  // useing this as a additional back channel to communicate if a value was found or not.
            return 0;
          } else {  // move to the next probing location
            lookup_position = normalizer<SimdStyle, HintSet, Idof>::normalize_value(
              lookup_position + SimdStyle::vector_element_count(), m_bucket_count);
          }
        }
      }
    }

   public:
    auto operator()(SimdOpsIterable auto p_output_ht, SimdOpsIterable auto p_output_data, SimdOpsIterable auto p_data,
                    SimdOpsIterableOrSizeT auto p_end, size_t position_offset = 0) const noexcept -> size_t {
      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      auto const all_false_mask = tsl::integral_all_false<SimdStyle, Idof>();
      size_t result_size = 0;
      size_t current_pos = position_offset;
      bool not_found = false;
      for (; p_data != end; ++p_data, ++current_pos) {
        auto key = *p_data;
        *p_output_ht = (size_t)lookup(key, not_found);
        *p_output_data = current_pos;

        if (!not_found) {
          ++p_output_ht;
          ++p_output_data;
          ++result_size;
        }
      }
      return result_size;
    }

    template <tsl::VectorProcessingStyle OtherSimdStlye, tsl::TSLArithmetic OtherPositionType, class OtherHintSet,
              typename OtherIdof>
    auto merge(Hash_Join_Probe_SIMD_Linear_Probing<OtherSimdStlye, OtherPositionType, OtherHintSet, OtherIdof> const
                 &other) const noexcept -> void {}

    auto finalize() const noexcept -> void {}
  };

  template <tsl::VectorProcessingStyle _SimdStyle, tsl::TSLArithmetic _PositionType,
            class HintSet = OperatorHintSet<hints::hashing::size_exp_2>, typename Idof = tsl::workaround>
  struct Hash_Join_SIMD_Linear_Probing {
    using builder_t = Hash_Join_Build_SIMD_Linear_Probing<_SimdStyle, _PositionType, HintSet, Idof>;
    using prober_t = Hash_Join_Probe_SIMD_Linear_Probing<_SimdStyle, _PositionType, HintSet, Idof>;
  };
}  // namespace tuddbs
#endif