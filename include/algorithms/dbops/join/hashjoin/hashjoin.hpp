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
 * @file hashjoin.hpp
 * @brief
 */

#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_HASHJOIN_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_HASHJOIN_HPP

#include <cassert>

#include "algorithms/dbops/hashing.hpp"
#include "algorithms/dbops/simdops.hpp"
#include "generated/declarations/mask.hpp"
#include "iterable.hpp"
#include "tslintrin.hpp"

namespace tuddbs {
  template <tsl::VectorProcessingStyle _SimdStyle, class HintSet = OperatorHintSet<hints::hashing::size_exp_2>,
            typename Idof = tsl::workaround>
  class Hash_Build_SIMD_Linear_Displacement {
   public:
    using SimdStyle = _SimdStyle;
    using KeySinkType = typename SimdStyle::base_type *;
    using GroupIdSinkType = typename SimdStyle::base_type *;

   private:
    KeySinkType m_key_sink;
    GroupIdSinkType m_group_id_sink;
    size_t const m_map_element_count;
    size_t m_distinct_key_count;
    size_t m_group_id_count;

    size_t const m_empty_bucket_value = 0;

   public:
    /**
     * @brief Constructs a Grouping_Hash_Table_SIMD_Linear_Displacement object.
     *
     * @param p_key_sink Pointer to the memory location where the keys will be stored.
     * @param p_group_id_sink Pointer to the memory location where the group IDs will be stored.
     * @param p_overall_key_count The total number of keys to be inserted into the hash table.
     * @param p_map_element_count The number of elements in the hash table.
     * @param initialize Flag indicating whether to initialize the hash table with empty values.
     */
    explicit Grouping_Hash_Table_SIMD_Linear_Displacement(KeySinkType p_key_sink, GroupIdSinkType p_group_id_sink,
                                                          size_t p_overall_key_count, size_t p_map_element_count,
                                                          bool initialize = true)
      : m_key_sink(p_key_sink),
        m_group_id_sink(p_group_id_sink),
        m_map_element_count(p_map_element_count),
        m_distinct_key_count(0),
        m_group_id_count(0) {
      if constexpr (has_hint<HintSet, hints::hashing::size_exp_2>) {
        assert((m_map_element_count & (m_map_element_count - 1)) == 0);
      }
      if (initialize) {
        for (auto i = 0; i < m_map_element_count; ++i) {
          m_key_sink[i] = m_empty_bucket_value;
          m_group_id_sink[i] = m_empty_bucket_value;
        }
      }
    }

    /**
     * @brief Destructor for the Grouping_Hash_Table_SIMD_Linear_Displacement object.
     */
    ~Grouping_Hash_Table_SIMD_Linear_Displacement() = default;

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
    TSL_FORCE_INLINE auto insert(typename SimdStyle::base_type const key,
                                 typename SimdStyle::imask_type const all_false_mask,
                                 typename SimdStyle::register_type const empty_bucket_reg) noexcept -> void {
      // broadcast the key to all lanes
      auto const keys_reg = tsl::set1<SimdStyle, Idof>(key);
      // calculate the position hint
      auto lookup_position =
        normalizer<SimdStyle, HintSet, Idof>::align_value(normalizer<SimdStyle, HintSet, Idof>::normalize_value(
          default_hasher<SimdStyle, Idof>::hash_value(*key), m_map_element_count));

      while (true) {
        // load N values from the map
        auto map_reg = tsl::loadu<SimdStyle, Idof>(m_key_sink + lookup_position);
        // compare the current key with the N values, if the key is found, the mask will contain a 1 bit at the position
        // of the found key
        auto const key_found_mask = tsl::to_integral<SimdStyle, Idof>(tsl::equal<SimdStyle, Idof>(map_reg, keys_reg));
        // if the key is found, we can stop the search, since we already inserted it into the map
        if (tsl::nequal<SimdStyle, Idof>(key_found_mask, all_false_mask)) {
          break;
        }
        // if the key is not found, we have to check if there is an empty bucket in the map
        auto const empty_bucket_found_mask =
          tsl::to_integral<SimdStyle, Idof>(tsl::equal<SimdStyle, Idof>(map_reg, empty_bucket_reg));
        if (tsl::nequal<SimdStyle, Idof>(empty_bucket_found_mask, all_false_mask)) {
          size_t empty_bucket_position = tsl::tzc<SimdStyle, Idof>(empty_bucket_found_mask);
          m_key_sink[lookup_position + empty_bucket_position] = key;
          m_group_id_sink[lookup_position + empty_bucket_position] = m_group_id_count++;
          m_distinct_key_count++;
          break;
        }
        lookup_position = normalizer<SimdStyle, HintSet, Idof>::align_value(
          normalizer<SimdStyle, HintSet, Idof>::normalize_value(lookup_position + SimdStyle::vector_element_count()));
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
    auto operator()(SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end) noexcept -> void {
      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      auto const all_false_mask = tsl::integral_all_false<SimdStyle, Idof>();
      auto const empty_bucket_reg = tsl::set1<SimdStyle, Idof>(m_empty_bucket_value);

      for (; p_data != end; ++p_data) {
        auto key = *p_data;
        insert(key, all_false_mask, empty_bucket_reg);
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
    auto merge(Grouping_Hash_Table_SIMD_Linear_Displacement const &other) noexcept -> void {
      auto const not_found_mask = tsl::integral_all_false<SimdStyle, Idof>();
      auto const empty_bucket_reg = tsl::set1<SimdStyle, Idof>(m_empty_bucket_value);

      for (auto i = 0; i < other.m_map_element_count; ++i) {
        auto const key = other.m_key_sink[i];
        if (key != m_empty_bucket_value) {
          insert(key, not_found_mask, empty_bucket_reg);
        }
      }
    }

    /**
     * @brief Finalizes the hash table.
     *
     * This function performs any necessary finalization steps for the hash table. Currently, it does nothing.
     */
    auto finalize() const noexcept -> void {}
  };
}

#endif