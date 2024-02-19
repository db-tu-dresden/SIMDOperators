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

#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_GROUP_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_GROUP_HPP

#include <cassert>
#include <climits>
#include <type_traits>

#include "algorithms/dbops/hashing.hpp"
#include "algorithms/dbops/simdops.hpp"
#include "iterable.hpp"
#include "tslintrin.hpp"

namespace tuddbs {

  namespace hints {
    namespace grouping {
      struct global_first_occurence_required {};
    }  // namespace grouping
  }    // namespace hints

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
  template <tsl::VectorProcessingStyle _SimdStyle, class HintSet = OperatorHintSet<hints::hashing::size_exp_2>,
            typename Idof = tsl::workaround>
  class Grouping_Hash_Build_SIMD_Linear_Displacement {
   public:
    using SimdStyle = _SimdStyle;
    using KeyType = typename SimdStyle::base_type;
    using KeySinkType = KeyType *;
    using GroupIdType = typename SimdStyle::base_type;
    using GroupIdSinkType = GroupIdType *;
    using PositionType = size_t;
    using PositionSinkType = PositionType *;

   private:
    KeySinkType m_key_sink;
    GroupIdSinkType m_group_id_sink;
    PositionSinkType m_original_positions_sink;

    size_t const m_map_element_count;
    size_t m_group_id_count;

    size_t const m_empty_bucket_value = 0;

    constexpr static PositionType const m_invalid_position = std::numeric_limits<PositionType>::max();
    constexpr static GroupIdType const m_invalid_gid = std::numeric_limits<GroupIdType>::max();

   public:
    /**
     * @brief Constructs a Grouping_Hash_Build_SIMD_Linear_Displacement object.
     *
     * @param p_key_sink Pointer to the memory location where the keys will be stored.
     * @param p_group_id_sink Pointer to the memory location where the group IDs will be stored.
     * @param p_overall_key_count The total number of keys to be inserted into the hash table.
     * @param p_map_element_count The number of elements in the hash table.
     * @param initialize Flag indicating whether to initialize the hash table with empty values.
     */
    explicit Grouping_Hash_Build_SIMD_Linear_Displacement(SimdOpsIterable auto p_key_sink,
                                                          SimdOpsIterable auto p_group_id_sink,
                                                          SimdOpsIterable auto p_original_first_occurence_position_sink,
                                                          size_t p_map_element_count, bool initialize = true)
      : m_key_sink(reinterpret_iterable<KeySinkType>(p_key_sink)),
        m_group_id_sink(reinterpret_iterable<GroupIdSinkType>(p_group_id_sink)),
        m_original_positions_sink(reinterpret_iterable<PositionSinkType>(p_original_first_occurence_position_sink)),
        m_map_element_count(p_map_element_count),
        m_group_id_count(0) {
      if constexpr (has_hint<HintSet, hints::hashing::size_exp_2>) {
        assert((m_map_element_count & (m_map_element_count - 1)) == 0);
      }
      if (initialize) {
        for (auto i = 0; i < m_map_element_count; ++i) {
          m_key_sink[i] = m_empty_bucket_value;
          m_group_id_sink[i] = m_invalid_gid;
          m_original_positions_sink[i] = m_invalid_position;
        }
      }
    }
    template <tsl::VectorProcessingStyle OtherSimdStlye, class OtherHintSet, typename OtherIdof, typename HS = HintSet,
              enable_if_has_hint_t<HS, hints::hashing::is_hull_for_merging>>
    explicit Grouping_Hash_Build_SIMD_Linear_Displacement(
      SimdOpsIterable auto p_key_sink, SimdOpsIterable auto p_group_id_sink,
      SimdOpsIterable auto p_original_first_occurence_position_sink, size_t p_map_element_count,
      Grouping_Hash_Build_SIMD_Linear_Displacement<OtherSimdStlye, OtherHintSet, OtherIdof> const &other)
      : m_key_sink(reinterpret_iterable<KeySinkType>(p_key_sink)),
        m_group_id_sink(reinterpret_iterable<GroupIdSinkType>(p_group_id_sink)),
        m_original_positions_sink(reinterpret_iterable<PositionSinkType>(p_original_first_occurence_position_sink)),
        m_map_element_count(p_map_element_count),
        m_group_id_count(0) {
      if constexpr (has_hint<HintSet, hints::hashing::size_exp_2>) {
        assert((m_map_element_count & (m_map_element_count - 1)) == 0);
      }
      for (auto i = 0; i < m_map_element_count; ++i) {
        m_key_sink[i] = m_empty_bucket_value;
        m_group_id_sink[i] = m_invalid_gid;
        m_original_positions_sink[i] = m_invalid_position;
      }
      merge<true, false>(other);
    }

    /**
     * @brief Destructor for the Grouping_Hash_Build_SIMD_Linear_Displacement object.
     */
    ~Grouping_Hash_Build_SIMD_Linear_Displacement() = default;

   private:
    /**
     * @brief Inserts a key into the hash table using SIMD and linear displacement.
     *
     * This function inserts a key into the hash table using SIMD instructions and linear displacement to handle
     * collisions. It first broadcasts the key to all lanes, then calculates the position hint based on the key's hash
     * value. It then iteratively searches for an empty bucket in the hash table, and if found, inserts the key and
     * assigns a group ID to it.
     *
     * @param key The key to insert into the hash table.
     * @param all_false_mask The SIMD mask representing all false values.
     * @param empty_bucket_reg The SIMD register containing the value representing an empty bucket.
     */
    template <bool CalledFromMerge = false>
    TSL_FORCE_INLINE auto insert(typename SimdStyle::base_type const key, PositionType const key_position_in_data,
                                 typename SimdStyle::imask_type const all_false_mask,
                                 typename SimdStyle::register_type const empty_bucket_reg) noexcept -> void {
      // broadcast the key to all lanes
      auto const keys_reg = tsl::set1<SimdStyle, Idof>(key);
      // calculate the position hint
      auto lookup_position =
        normalizer<SimdStyle, HintSet, Idof>::align_value(normalizer<SimdStyle, HintSet, Idof>::normalize_value(
          default_hasher<SimdStyle, Idof>::hash_value(key), m_map_element_count));

      while (true) {
        typename SimdStyle::register_type map_reg;
        if constexpr (has_hint<HintSet, hints::memory::aligned>) {
          // load N values from the map
          map_reg = tsl::load<SimdStyle, Idof>(m_key_sink + lookup_position);
        } else {
          // load N values from the map
          map_reg = tsl::loadu<SimdStyle, Idof>(m_key_sink + lookup_position);
        }
        // compare the current key with the N values, if the key is found, the mask will contain a 1 bit at the position
        // of the found key
        auto const key_found_mask = tsl::equal_as_imask<SimdStyle, Idof>(map_reg, keys_reg);
        if (tsl::nequal<SimdStyle, Idof>(key_found_mask, all_false_mask)) {
          auto const found_position = tsl::tzc<SimdStyle, Idof>(key_found_mask);
          auto group_id = m_group_id_sink[lookup_position + found_position];
          if constexpr (has_hint<HintSet, hints::hashing::keys_may_contain_zero>) {
            // If the key can be the same value as an empty bucket, we have to make an additional check
            if (key == m_empty_bucket_value) {
              // If the current processed key equals the empty bucket value, we have check, whether the found position
              // already contains the key. This is done by looking into the correspondig group id. If the group id is
              // invalid, we did not see the key before and have to insert it. Otherwise we have to check, whether the
              // current key position is smaller than the original position of the key.
              if (group_id == m_invalid_gid) {
                m_group_id_sink[lookup_position + found_position] = m_group_id_count;
                m_original_positions_sink[m_group_id_count++] = key_position_in_data;
              } else {
                if constexpr (has_hint<HintSet, hints::grouping::global_first_occurence_required>) {
                  if (m_original_positions_sink[group_id] > key_position_in_data) {
                    m_original_positions_sink[group_id] = key_position_in_data;
                  }
                }
              }
            } else {
              if constexpr (has_hint<HintSet, hints::grouping::global_first_occurence_required>) {
                if (m_original_positions_sink[group_id] > key_position_in_data) {
                  m_original_positions_sink[group_id] = key_position_in_data;
                }
              }
            }
          } else {
            if constexpr (has_hint<HintSet, hints::grouping::global_first_occurence_required>) {
              if (m_original_positions_sink[group_id] > key_position_in_data) {
                m_original_positions_sink[group_id] = key_position_in_data;
              }
            }
          }
          break;
        }
        // if the key is not found, we have to check if there is an empty bucket in the map
        auto const empty_bucket_found_mask = tsl::equal_as_imask<SimdStyle, Idof>(map_reg, empty_bucket_reg);
        if (tsl::nequal<SimdStyle, Idof>(empty_bucket_found_mask, all_false_mask)) {
          size_t empty_bucket_position = tsl::tzc<SimdStyle, Idof>(empty_bucket_found_mask);

          if constexpr (has_hint<HintSet, hints::hashing::keys_may_contain_zero>) {
            auto group_id = m_group_id_sink[lookup_position + empty_bucket_position];
            if (group_id == m_invalid_gid) {
              auto updated_empty_bucket_found_mask =
                tsl::shift_right<SimdStyle, Idof>(empty_bucket_found_mask, empty_bucket_position + 1);
              if (tsl::nequal<SimdStyle, Idof>(updated_empty_bucket_found_mask, all_false_mask)) {
                // As there can be only a single occurence of the key that equals an empty bucket, we only have to use
                // the first occurence
                auto updated_empty_bucket_position =
                  tsl::tzc<SimdStyle, Idof>(updated_empty_bucket_found_mask) + empty_bucket_position + 1;
                m_key_sink[lookup_position + updated_empty_bucket_position] = key;
                m_group_id_sink[lookup_position + updated_empty_bucket_position] = m_group_id_count;
                m_original_positions_sink[m_group_id_count++] = key_position_in_data;
              }
            } else {
              m_key_sink[lookup_position + empty_bucket_position] = key;
              m_group_id_sink[lookup_position + empty_bucket_position] = m_group_id_count;
              m_original_positions_sink[m_group_id_count++] = key_position_in_data;
            }
          } else {
            m_key_sink[lookup_position + empty_bucket_position] = key;
            m_group_id_sink[lookup_position + empty_bucket_position] = m_group_id_count;
            m_original_positions_sink[m_group_id_count++] = key_position_in_data;
          }
          break;
        }
        lookup_position = normalizer<SimdStyle, HintSet, Idof>::normalize_value(
          lookup_position + SimdStyle::vector_element_count(), m_map_element_count);
      }
    }

   public:
    /**
     * @brief Inserts elements into the hash table.
     *
     * This function inserts elements into the hash table by iterating over the input data and calling the insert
     * function for each element. It uses SIMD instructions to process multiple elements in parallel.
     *
     * @param p_data The input data to insert into the hash table.
     * @param p_end The end iterator of the input data.
     */
    auto operator()(SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end,
                    PositionType start_position = 0) noexcept -> void {
      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      auto const all_false_mask = tsl::integral_all_false<SimdStyle, Idof>();
      auto const empty_bucket_reg = tsl::set1<SimdStyle, Idof>(m_empty_bucket_value);

      for (; p_data != end; ++p_data, ++start_position) {
        auto key = *p_data;
        insert(key, start_position, all_false_mask, empty_bucket_reg);
      }
    }

    auto operator()(SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end, SimdOpsIterable auto p_valid_masks,
                    PositionType start_position = 0) noexcept -> void {
      // Get the end of the SIMD iteration
      auto const simd_end = simd_iter_end<SimdStyle>(p_data, p_end);
      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      auto valid_masks = reinterpret_iterable<typename SimdStyle::imask_type>(p_valid_masks);
      auto const all_false_mask = tsl::integral_all_false<SimdStyle, Idof>();
      auto const empty_bucket_reg = tsl::set1<SimdStyle, Idof>(m_empty_bucket_value);

      for (; p_data != simd_end; p_data += SimdStyle::vector_element_count(), ++valid_masks) {
        auto valid_mask = tsl::load_mask<SimdStyle, Idof>(valid_masks);
        for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i) {
          if (tsl::test_mask<SimdStyle, Idof>(valid_mask, i)) {
            auto key = p_data[i];
            insert(key, start_position, all_false_mask, empty_bucket_reg);
            if constexpr (!has_hint<HintSet, hints::operators::preserve_original_positions>()) {
              ++start_position;
            }
          } else {
            if constexpr (has_hint<HintSet, hints::operators::preserve_original_positions>()) {
              ++start_position;
            }
          }
        }
      }
      if (p_data != end) {
        auto valid_mask = tsl::load_mask<SimdStyle, Idof>(valid_masks);
        int i = 0;
        for (; p_data != end; ++p_data, ++i) {
          if (tsl::test_mask<SimdStyle, Idof>(valid_mask, i)) {
            auto key = *p_data;
            insert(key, start_position, all_false_mask, empty_bucket_reg);
            if constexpr (!has_hint<HintSet, hints::operators::preserve_original_positions>()) {
              ++start_position;
            }
          } else {
            if constexpr (has_hint<HintSet, hints::operators::preserve_original_positions>()) {
              ++start_position;
            }
          }
        }
      }
    }

    template <class HS = HintSet, enable_if_has_hint_t<HS, hints::intermediate::dense_bit_mask>>
    auto operator()(SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end, SimdOpsIterable auto p_valid_masks,
                    PositionType start_position = 0) noexcept -> void {
      constexpr auto const bits_per_mask = sizeof(typename SimdStyle::imask_type) * CHAR_BIT;
      // Get the end of the SIMD iteration
      auto const batched_end_end = batched_iter_end<bits_per_mask>(p_data, p_end);
      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      auto valid_masks = reinterpret_iterable<typename SimdStyle::imask_type>(p_valid_masks);
      auto const all_false_mask = tsl::integral_all_false<SimdStyle, Idof>();
      auto const empty_bucket_reg = tsl::set1<SimdStyle, Idof>(m_empty_bucket_value);

      for (; p_data != batched_end_end; p_data += bits_per_mask, ++valid_masks) {
        auto valid_mask = tsl::load_mask<SimdStyle, Idof>(valid_masks);
        for (size_t i = 0; i < bits_per_mask; ++i) {
          if (tsl::test_mask<SimdStyle, Idof>(valid_mask, i)) {
            auto key = p_data[i];
            insert(key, start_position, all_false_mask, empty_bucket_reg);
            if constexpr (!has_hint<HintSet, hints::operators::preserve_original_positions>()) {
              ++start_position;
            }
          } else {
            if constexpr (has_hint<HintSet, hints::operators::preserve_original_positions>()) {
              ++start_position;
            }
          }
        }
      }
      if (p_data != end) {
        auto valid_mask = tsl::load_mask<SimdStyle, Idof>(valid_masks);
        int i = 0;
        for (; p_data != end; ++p_data, ++i) {
          if (tsl::test_mask<SimdStyle, Idof>(valid_mask, i)) {
            auto key = *p_data;
            insert(key, start_position, all_false_mask, empty_bucket_reg);
            if constexpr (!has_hint<HintSet, hints::operators::preserve_original_positions>()) {
              ++start_position;
            }
          } else {
            if constexpr (has_hint<HintSet, hints::operators::preserve_original_positions>()) {
              ++start_position;
            }
          }
        }
      }
    }

    /**
     * @brief Merges another hash table into this hash table.
     *
     * This function merges the contents of another hash table into this hash table. It iterates over the other hash
     * table and inserts each non-empty key into this hash table using SIMD instructions.
     *
     * @param other The other hash table to merge.
     */
    template <tsl::VectorProcessingStyle OtherSimdStlye, class OtherHintSet, typename OtherIdof,
              bool NeedsPosition = has_hint<HintSet, hints::grouping::global_first_occurence_required>,
              bool ExplicitMerge = true>
    auto merge(
      Grouping_Hash_Build_SIMD_Linear_Displacement<OtherSimdStlye, OtherHintSet, OtherIdof> const &other) noexcept
      -> void {
      auto const all_false_mask = tsl::integral_all_false<SimdStyle, Idof>();
      auto const empty_bucket_reg = tsl::set1<SimdStyle, Idof>(m_empty_bucket_value);

      for (auto i = 0; i < other.m_map_element_count; ++i) {
        auto const gid = other.m_group_id_sink[i];
        if (gid != m_invalid_gid) {
          auto const key = other.m_key_sink[i];
          if constexpr (NeedsPosition) {
            auto original_key_position = other.m_original_positions_sink[gid];
            insert<ExplicitMerge>(key, original_key_position, all_false_mask, empty_bucket_reg);
          } else {
            insert<ExplicitMerge>(key, 0, all_false_mask, empty_bucket_reg);
          }
        }
      }
    }

    /**
     * @brief Finalizes the hash table.
     *
     * This function performs any necessary finalization steps for the hash table. Currently, it does nothing.
     */
    auto finalize() const noexcept -> void {}

    auto distinct_key_count() const noexcept { return m_group_id_count; }
  };

  template <tsl::VectorProcessingStyle _SimdStyle, class HintSet = OperatorHintSet<hints::hashing::size_exp_2>,
            typename Idof = tsl::workaround>
  class Grouper_SIMD_Linear_Displacement {
   public:
    using SimdStyle = _SimdStyle;
    using KeyType = typename SimdStyle::base_type;
    using KeySinkType = KeyType *;
    using GroupIdType = typename SimdStyle::base_type;
    using GroupIdSinkType = GroupIdType *;
    using PositionType = size_t;
    using PositionSinkType = PositionType *;

   private:
    KeySinkType m_key_sink;
    GroupIdSinkType m_group_id_sink;
    size_t const m_map_element_count;

   public:
    explicit Grouper_SIMD_Linear_Displacement(KeySinkType p_key_sink, GroupIdSinkType p_group_id_sink,
                                              PositionSinkType p_original_positions_sink, size_t p_map_element_count)
      : m_key_sink(reinterpret_iterable<KeySinkType>(p_key_sink)),
        m_group_id_sink(reinterpret_iterable<GroupIdSinkType>(p_group_id_sink)),
        m_map_element_count(p_map_element_count) {
      if constexpr (has_hint<HintSet, hints::hashing::size_exp_2>) {
        assert((m_map_element_count & (m_map_element_count - 1)) == 0);
      }
    }
    ~Grouper_SIMD_Linear_Displacement() = default;

   private:
    TSL_FORCE_INLINE auto lookup(typename SimdStyle::base_type const key,
                                 typename SimdStyle::imask_type const all_false_mask) const noexcept -> GroupIdType {
      // broadcast the key to all lanes
      auto const keys_reg = tsl::set1<SimdStyle, Idof>(key);
      // calculate the position hint
      auto lookup_position =
        normalizer<SimdStyle, HintSet, Idof>::align_value(normalizer<SimdStyle, HintSet, Idof>::normalize_value(
          default_hasher<SimdStyle, Idof>::hash_value(key), m_map_element_count));

      while (true) {
        // load N values from the map
        auto map_reg = tsl::loadu<SimdStyle, Idof>(m_key_sink + lookup_position);
        // compare the current key with the N values, if the key is found, the mask will contain a 1 bit at the position
        auto const key_found_mask = tsl::equal_as_imask<SimdStyle, Idof>(map_reg, keys_reg);
        if (tsl::nequal<SimdStyle, Idof>(key_found_mask, all_false_mask)) {
          size_t position = tsl::tzc<SimdStyle, Idof>(key_found_mask);
          return m_group_id_sink[lookup_position + position];
        }
        lookup_position =
          normalizer<SimdStyle, HintSet, Idof>::align_value(normalizer<SimdStyle, HintSet, Idof>::normalize_value(
            lookup_position + SimdStyle::vector_element_count(), m_map_element_count));
      }
      // this should never be reached
      return 0;
    }

   public:
    auto operator()(SimdOpsIterable auto p_output_gids, SimdOpsIterable auto p_data,
                    SimdOpsIterableOrSizeT auto p_end) const noexcept -> void {
      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      auto const all_false_mask = tsl::integral_all_false<SimdStyle, Idof>();

      for (; p_data != end; ++p_data, ++p_output_gids) {
        auto key = *p_data;
        *p_output_gids = lookup(key, all_false_mask);
      }
    }

    auto operator()(SimdOpsIterable auto p_output_gids, SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end,
                    SimdOpsIterable auto p_valid_masks) const noexcept -> void {
      // Get the end of the SIMD iteration
      auto const simd_end = simd_iter_end<SimdStyle>(p_data, p_end);
      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      auto valid_masks = reinterpret_iterable<typename SimdStyle::imask_type>(p_valid_masks);

      auto const all_false_mask = tsl::integral_all_false<SimdStyle, Idof>();

      for (; p_data != simd_end; p_data += SimdStyle::vector_element_count(), ++valid_masks) {
        auto valid_mask = tsl::load_mask<SimdStyle, Idof>(valid_masks);
        for (size_t i = 0; i < SimdStyle::vector_element_count(); ++i, ++p_output_gids) {
          if (tsl::test_mask<SimdStyle, Idof>(valid_mask, i)) {
            auto key = *p_data;
            *p_output_gids = lookup(key, all_false_mask);
          }
        }
      }
      for (; p_data != end; ++p_data, ++p_output_gids) {
        auto key = *p_data;
        *p_output_gids = lookup(key, all_false_mask);
      }
    }

    template <class HS = HintSet, enable_if_has_hint_t<HS, hints::intermediate::dense_bit_mask>>
    auto operator()(SimdOpsIterable auto p_output_gids, SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end,
                    SimdOpsIterable auto p_valid_masks) const noexcept -> void {
      constexpr auto const bits_per_mask = sizeof(typename SimdStyle::imask_type) * CHAR_BIT;
      // Get the end of the SIMD iteration
      auto const batched_end_end = batched_iter_end<bits_per_mask>(p_data, p_end);
      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      auto valid_masks = reinterpret_iterable<typename SimdStyle::imask_type>(p_valid_masks);
      auto const all_false_mask = tsl::integral_all_false<SimdStyle, Idof>();

      for (; p_data != batched_end_end; p_data += bits_per_mask, ++valid_masks) {
        auto valid_mask = tsl::load_mask<SimdStyle, Idof>(valid_masks);
        for (size_t i = 0; i < bits_per_mask; ++i, ++p_output_gids) {
          if (tsl::test_mask<SimdStyle, Idof>(valid_mask, i)) {
            auto key = p_data[i];
            *p_output_gids = lookup(key, all_false_mask);
          }
        }
      }
      if (p_data != end) {
        auto valid_mask = tsl::load_mask<SimdStyle, Idof>(valid_masks);
        int i = 0;
        for (; p_data != end; ++p_data, ++i, ++p_output_gids) {
          if (tsl::test_mask<SimdStyle, Idof>(valid_mask, i)) {
            auto key = *p_data;
            *p_output_gids = lookup(key, all_false_mask);
          }
        }
      }
    }

    auto merge(Grouper_SIMD_Linear_Displacement const &other) const noexcept -> void {}

    auto finalize() const noexcept -> void {}
  };

  template <tsl::VectorProcessingStyle _SimdStyle, class HintSet = OperatorHintSet<hints::hashing::size_exp_2>,
            typename Idof = tsl::workaround>
  struct Group_SIMD_Linear_Displacement {
    using builder_t = Grouping_Hash_Build_SIMD_Linear_Displacement<_SimdStyle, HintSet, Idof>;
    using grouper_t = Grouper_SIMD_Linear_Displacement<_SimdStyle, HintSet, Idof>;
  };

  template <tsl::VectorProcessingStyle _SimdStyle,
            class HintSet = OperatorHintSet<hints::hashing::size_exp_2, hints::hashing::linear_displacement>,
            typename Idof = tsl::workaround>
  struct Group {
    using base_class = std::conditional_t<has_hints<HintSet, hints::hashing::linear_displacement> &&
                                            !has_hint<HintSet, hints::hashing::refill>,
                                          Group_SIMD_Linear_Displacement<_SimdStyle, HintSet, Idof>, void>;
    using builder_t = typename base_class::builder_t;
    using grouper_t = typename base_class::grouper_t;
  };
}  // namespace tuddbs
#endif