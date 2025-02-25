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
#pragma once

#include <cassert>
#include <climits>
#include <type_traits>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <iomanip>

#include "algorithms/dbops/dbops_hints.hpp"
#include "algorithms/dbops/join/hash_join_hints.hpp"
#include "algorithms/utils/hashing.hpp"
#include "iterable.hpp"
// #include "static/utils/preprocessor.hpp"
// #include "static/utils/type_concepts.hpp"
#include "tsl.hpp"

namespace tuddbs {
  /**
   * @brief
   * A semi join gets a left and a right table-column as input and returns all elements (either as bitmask or as
   * positionlist) of the left table that have a match in the right table. It is inherently order preserving. The
   * fundamental idea of the implementation is as follows:
   *  - Build a hash set for the right table that only contains the unique values from the right side.
   *    Consequently, the build phase can be used by other operators as well, if UNIQUE(col) needs to be computed.
   *    However, the unique values will __NOT__ be ordered (neither by the order of the input nor by the value itself)
   *  - Probe the hash set with the values from the left side
   * @details Internally, the hash
   * @tparam _SimdStyle
   * @tparam HintSet
   * @tparam Idof
   */
  template <tsl::VectorProcessingStyle _SimdStyle, class HintSet = OperatorHintSet<hints::hashing::size_exp_2>,
            typename Idof = tsl::workaround>
  class Hash_Semi_Join_Build_RightSide_SIMD_Linear_Probing {
   public:
    using SimdStyle = _SimdStyle;
    using KeyType = typename SimdStyle::base_type;
    using KeySinkType = KeyType *;

    using BucketFreeBitMaskType = uint64_t;
    using BucketFreeSinkType = BucketFreeBitMaskType *;

   private:
    KeySinkType m_key_sink;
    BucketFreeSinkType m_free_bucket_slot_bitset_sink;
    size_t const m_bucket_count;
    size_t key_count = 0;
    BucketFreeBitMaskType const m_bucket_empty_check =
      ((BucketFreeBitMaskType)1 << (BucketFreeBitMaskType)SimdStyle::vector_element_count()) - 1;

   private:
    /**
     * @brief Checks whether the current search starting index has a free bucket.
     * @details
     * m_free_bucket_slot_bitset_sink contains a bitset that indicates, whether the associated bucket is empty
     * (corresponding bit set to 1) or not (corresponding bit set to 0). As we execute the linear probing (search) in a
     * vectorized fashion, this function is executed whenever a SIMD-compare equal for an empty bucket value is != 0.
     * Let's consider the following example:
     *  - We have a SIMD register size of 256 bits, and operate on 64-bit integers. Consequently, we have 4 elements in
     * a SIMD register.
     *  - The used bucket sink value at the given position is is 0b0010. This means, the second element in the
     * m_free_bucket_slot_bitset_sink is free. The return value will be 1. In contrast, if the value would be 0b0000,
     * the return value would be -1. The function will additionally set the bit in the m_free_bucket_slot_bitset_sink to
     * 1, indicating that the bucket is now used.
     * @param bucket_index Index of the bucket
     * @param to_check
     * @return TSL_FORCE_INLINE
     */
    auto check_next_empty(size_t const bucket_index) noexcept -> int {
      // std::cout << "Before check_next_empty" << std::endl;
      // print();
      // first we divde the bucket index by 64 to get the right bucket
      auto const bitmask_idx_base = bucket_index >> 6;
      // then we calculate the offset of the bucket index in the bitmask
      auto const bitmask_idx_offset = bucket_index & 63;
      // now we load the bitmask
      auto const bitmask = m_free_bucket_slot_bitset_sink[bitmask_idx_base];
      // now we shift the bitmask to the right to get the bit we want to check
      auto const empty_buckets_bitmask = (bitmask >> bitmask_idx_offset) & m_bucket_empty_check;
      if (empty_buckets_bitmask == 0) {
        // std::cout << "No empty bucket..." << std::endl;
        return -1;
      }
      // get the trailing zero count of the empty_buckets_bitmask
      auto const first_empty_bucket_pos = tsl::tzc<SimdStyle>(empty_buckets_bitmask);
      auto const updated_empty_buckets_bitmask = (BucketFreeBitMaskType)1
                                                 << (first_empty_bucket_pos + bitmask_idx_offset);
      m_free_bucket_slot_bitset_sink[bitmask_idx_base] = bitmask & ~updated_empty_buckets_bitmask;
      // std::cout << "After check_next_empty" << std::endl;
      // print();
      return first_empty_bucket_pos;
    }

    /**
     * @brief Set the occupied bucket object
     *
     * @param bucket_index
     * @param offset
     * @return int 0 if nothing was changed, 1 if the bucket was set to occupied
     */
    auto set_occupied_bucket(size_t const bucket_index, size_t offset) noexcept -> int {
      // std::cout << "Before set_occupied_bucket" << std::endl;
      // print();
      // first we divde the bucket index by 64 to get the right bucket
      auto const bitmask_idx_base = bucket_index >> 6;
      // then we calculate the offset of the bucket index in the bitmask
      auto const bitmask_idx_offset = bucket_index & 63;
      auto const bitmask = m_free_bucket_slot_bitset_sink[bitmask_idx_base];
      auto const updated_empty_buckets_bitmask = (BucketFreeBitMaskType)1 << (offset + bitmask_idx_offset);
      m_free_bucket_slot_bitset_sink[bitmask_idx_base] = bitmask & ~updated_empty_buckets_bitmask;
      // std::cout << "After set_occupied_bucket" << std::endl;
      // print();
      return (bitmask >> (bitmask_idx_base + offset)) & 1;
    }

    auto insert_key(KeyType const key) noexcept -> void {
      auto const keys_reg = tsl::set1<SimdStyle, Idof>(key);
      auto const all_false_mask = tsl::integral_all_false<SimdStyle, Idof>();
      auto lookup_position =
        normalizer<SimdStyle, HintSet, Idof>::align_value(normalizer<SimdStyle, HintSet, Idof>::normalize_value(
          default_hasher<SimdStyle, Idof>::hash_value(key), m_bucket_count));
      typename SimdStyle::register_type map_reg;
      auto const initial_lookup_position = lookup_position;
      // bool wrapped_around = false;
      while (true) {
        // if (lookup_position < initial_lookup_position) {
          // wrapped_around = true;
        // } else {
        //   if (wrapped_around) {
        //     std::cout << "Full iteration passed without being able to insert key. something went horribly wrong." << std::endl;
        //     std::cout << "Key: " << key << std::endl;
        //     std::cout << "Initial lookup position: " << initial_lookup_position << std::endl;
        //     std::cout << "Bucket count: " << m_bucket_count << std::endl;
        //     std::cout << "Unique values so far: " << key_count << std::endl;
        //     print();
        //     throw std::runtime_error("error while hash build");
        //   }
        // }
        // if (key == 19940121) {
        //   std::cout << "Key of interest! (" << key << ")" << std::endl;
        //   print();
        // }
        if (has_hint<HintSet, hints::memory::aligned>) {
          map_reg = tsl::load<SimdStyle, Idof>(m_key_sink + lookup_position);
        } else {
          map_reg = tsl::loadu<SimdStyle, Idof>(m_key_sink + lookup_position);
        }
        auto const key_found_mask = tsl::equal_as_imask<SimdStyle, Idof>(map_reg, keys_reg);

        if (key_found_mask == 0) {
          // Key not found.
          // Check if it can be inserted
          auto const empty_bucket_position = check_next_empty(lookup_position);
          if (empty_bucket_position == -1) {
            // Key can not be inserted, move to the next probing location
            // std::cout << "Key " << key << " not in Sink [" << lookup_position << ":" << lookup_position+ SimdStyle::vector_element_count() << "] Move forward." << std::endl;
            lookup_position = normalizer<SimdStyle, HintSet, Idof>::normalize_value(
              lookup_position + SimdStyle::vector_element_count(), m_bucket_count);
          } else {
            // Key can be inserted
            m_key_sink[lookup_position + empty_bucket_position] = key;
            // std::cout << "Sink [" << lookup_position + empty_bucket_position << "] = " << key << std::endl;
            ++key_count;
            return;
          }
        } else {
          // Key found
          auto const equal_key_offset = tsl::tzc<SimdStyle>(key_found_mask);
          key_count += set_occupied_bucket(lookup_position, equal_key_offset);
          // std::cout << "Sink [" << lookup_position + equal_key_offset << "] already contained " << key << std::endl;
          return;
        }
      }
    }

   public:

    auto print() const noexcept -> void {
      for (size_t i = 0; i < m_bucket_count; ++i) {
        auto const bitmask_idx_base = i >> 6;
        auto const bitmask_idx_offset = i & 63;
        auto const bitmask = m_free_bucket_slot_bitset_sink[bitmask_idx_base];

        size_t before_count = ((bitmask_idx_offset+SimdStyle::vector_element_count()) > 64) ? 0 : 64-(bitmask_idx_offset+SimdStyle::vector_element_count());
        size_t after_count = ((bitmask_idx_offset+SimdStyle::vector_element_count())> 64) ? (bitmask_idx_offset+SimdStyle::vector_element_count())-before_count-SimdStyle::vector_element_count() : 64-before_count-SimdStyle::vector_element_count();
        auto empty_buckets_bitmask = (bitmask >> bitmask_idx_offset) & m_bucket_empty_check;
        std::string before = (before_count > 0) ? std::string(before_count, '_') : "";
        std::string after = (after_count > 0) ? std::string(after_count, '_'): "";
        std::cout << "[" << std::setw(5) << i << "]: " << std::setw(15) << m_key_sink[i] << " | " << before;
        if((bitmask_idx_offset+SimdStyle::vector_element_count())> 64) {
          for (int i = bitmask_idx_offset; i < 64; ++i) {
            std::cout << (empty_buckets_bitmask & 0b1);
            empty_buckets_bitmask >>= 1;
          }
        } else {
          std::cout << std::bitset<SimdStyle::vector_element_count()>{empty_buckets_bitmask};
        }
        std::cout << after << std::endl;
      }
    }

    auto unique_keys_count() const noexcept -> size_t {
      return key_count;
    }

    Hash_Semi_Join_Build_RightSide_SIMD_Linear_Probing(KeySinkType p_key_sink,
                                                       BucketFreeSinkType p_free_bucket_slot_bitset_sink,
                                                       size_t p_bucket_count,
                                                       size_t p_bitset_value_count) noexcept
      : m_key_sink(reinterpret_iterable<KeySinkType>(p_key_sink)),
        m_free_bucket_slot_bitset_sink(reinterpret_iterable<BucketFreeSinkType>(p_free_bucket_slot_bitset_sink)),
        m_bucket_count(p_bucket_count) {
      if constexpr (has_hint<HintSet, hints::hashing::size_exp_2>) {
        assert((m_bucket_count & (m_bucket_count - 1)) == 0);
      }
      std::fill(m_key_sink, m_key_sink + m_bucket_count, (KeyType)0);
      std::fill(m_free_bucket_slot_bitset_sink, m_free_bucket_slot_bitset_sink+p_bitset_value_count, (BucketFreeBitMaskType)-1);

    }

    auto operator()(SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end) noexcept -> void {
      auto const end = iter_end(p_data, p_end);
      size_t insertion_count = 0;
      for (; p_data != end; ++p_data) {
        auto key = *p_data;
        insert_key(key);
      }
    }
  };

  template <tsl::VectorProcessingStyle _SimdStyle, tsl::TSLArithmetic _PositionType = size_t,
            class HintSet = OperatorHintSet<hints::hashing::size_exp_2>, typename Idof = tsl::workaround>
  class Hash_Semi_Join_Probe_RightSide_SIMD_Linear_Probing {
   public:
    using SimdStyle = _SimdStyle;
    using KeyType = typename SimdStyle::base_type;
    using KeySinkType = KeyType *;

    using BucketFreeBitMaskType = uint64_t;
    using BucketFreeSinkType = BucketFreeBitMaskType *;

    using PositionType = _PositionType;
    using PositionSinkType = PositionType *;

   private:
    KeySinkType m_key_sink;
    BucketFreeSinkType m_free_bucket_slot_bitset_sink;
    size_t const m_bucket_count;
    size_t key_count = 0;
    BucketFreeBitMaskType const m_bucket_empty_check =
      ((BucketFreeBitMaskType)1 << (BucketFreeBitMaskType)SimdStyle::vector_element_count()) - 1;

   public:
    explicit Hash_Semi_Join_Probe_RightSide_SIMD_Linear_Probing(SimdOpsIterable auto p_key_sink,
                                                                SimdOpsIterable auto p_free_bucket_slot_bitset_sink,
                                                                size_t p_bucket_count) noexcept
      : m_key_sink(reinterpret_iterable<KeySinkType>(p_key_sink)),
        m_free_bucket_slot_bitset_sink(reinterpret_iterable<BucketFreeSinkType>(p_free_bucket_slot_bitset_sink)),
        m_bucket_count(p_bucket_count) {
      if constexpr (has_hint<HintSet, hints::hashing::size_exp_2>) {
        assert((m_bucket_count & (m_bucket_count - 1)) == 0);
      }
    }

   private:
    auto check_for_empty_buckets_in_range(size_t const bucket_index) noexcept -> bool {
      // first we divde the bucket index by 64 to get the right bucket
      auto const bitmask_idx_base = bucket_index >> 6;
      // then we calculate the offset of the bucket index in the bitmask
      auto const bitmask_idx_offset = bucket_index & 63;
      // now we load the bitmask
      auto const bitmask = m_free_bucket_slot_bitset_sink[bitmask_idx_base];
      // now we shift the bitmask to the right to get the bit we want to check
      auto const empty_buckets_bitmask = (bitmask >> bitmask_idx_offset) & m_bucket_empty_check;
      if (empty_buckets_bitmask == 0) {
        return false;
      }
      return true;
    }

    // auto check_for_empty_bucket(size_t const bucket_index, size_t offset, bool log = false) noexcept -> int {
    auto check_for_empty_bucket(size_t const bucket_index, size_t offset) noexcept -> int {
      // first we divde the bucket index by 64 to get the right bucket
      auto const bitmask_idx_base = bucket_index >> 6;
      // then we calculate the offset of the bucket index in the bitmask
      auto const bitmask_idx_offset = bucket_index & 63;
      // now we load the bitmask
      auto const bitmask = m_free_bucket_slot_bitset_sink[bitmask_idx_base];
      // if (log) {
      //   std::cout << "\t\t\tBucket index: " << bucket_index << std::endl;
      //   std::cout << "\t\t\tOffset      : " << offset << std::endl;
      //   std::cout << "\t\t\tBitmask     : " << std::bitset<64>{bitmask} << std::endl;
      //   std::cout << "\t\t\tBIB+Off     : " << bitmask_idx_offset + offset << std::endl;
      //   std::cout << "\t\t\tBM>>(BIB+Off): " << std::bitset<64>{(bitmask>>(bitmask_idx_offset+offset))} << std::endl;
      //   std::cout << "\t\t\tResult      : " << ((bitmask >> (bitmask_idx_offset + offset)) & 1) << std::endl;
      // }
      return (bitmask >> (bitmask_idx_offset + offset)) & 1;
    }

    auto probe_key(KeyType const key) noexcept -> bool {
      auto const keys_reg = tsl::set1<SimdStyle, Idof>(key);
      auto const all_false_mask = tsl::integral_all_false<SimdStyle, Idof>();
      auto lookup_position =
        normalizer<SimdStyle, HintSet, Idof>::align_value(normalizer<SimdStyle, HintSet, Idof>::normalize_value(
          default_hasher<SimdStyle, Idof>::hash_value(key), m_bucket_count));
      typename SimdStyle::register_type map_reg;
      while (true) {
        if (has_hint<HintSet, hints::memory::aligned>) {
          map_reg = tsl::load<SimdStyle, Idof>(m_key_sink + lookup_position);
        } else {
          map_reg = tsl::loadu<SimdStyle, Idof>(m_key_sink + lookup_position);
        }
        auto const key_found_mask = tsl::equal_as_imask<SimdStyle, Idof>(map_reg, keys_reg);
        if (key_found_mask == 0) {
          // Key not found
          // std::cout << "\t\tNot found" << std::endl;
          if (check_for_empty_buckets_in_range(lookup_position)) {
            // std::cout << "\t\tEmpty bucket found --> not present" << std::endl;
            // std::cout << "Key not found, but empty bucket in the range Sink [" << lookup_position << ":" << lookup_position + SimdStyle::vector_element_count() << "]." << std::endl;
            // Empty bucket found, key is not in the set
            // std::cout << "Key " << key << " not found" << std::endl;
            return false;
          } else {
            // std::cout << "Key not found and no empty bucket in the range Sink [" << lookup_position << ":" << lookup_position + SimdStyle::vector_element_count() << "]." << std::endl;
            lookup_position = normalizer<SimdStyle, HintSet, Idof>::normalize_value(
              lookup_position + SimdStyle::vector_element_count(), m_bucket_count);
          }
        } else {
          // Key found
          auto const equal_key_offset = tsl::tzc<SimdStyle>(key_found_mask);
          if (check_for_empty_bucket(lookup_position, equal_key_offset) == 1) {
            // throw std::runtime_error("Key found, but bucket was marked empty");
            // std::cout << "Probing key " << key << std::endl;
            // std::cout << "\tPosition: " << lookup_position << std::endl;
            // std::cout << "\t\tFound" << std::endl;
            // std::cout << "\t\tEMPTY!!!" << std::endl;
            // check_for_empty_bucket(lookup_position, equal_key_offset, true);
            // std::cout << "Found Key " << key << " in Sink [" << lookup_position+equal_key_offset << "] BUT bucket is empty." << std::endl; 
            return false;
          }
          // std::cout << "Found key " << key << " in Sink [" << lookup_position+equal_key_offset << "]." << std::endl; 
          // std::cout << "Found key" << key << std::endl;
          return true;
        }
      }
    }

   public:
    auto operator()(SimdOpsIterable auto p_output_pos, SimdOpsIterable auto p_data,
                    SimdOpsIterableOrSizeT auto p_end) noexcept -> size_t {
      auto const end = iter_end(p_data, p_end);
      size_t result_size = 0;
      size_t current_pos = 0;
      for (; p_data != end; ++p_data, ++current_pos) {
        auto key = *p_data;
        if (probe_key(key)) {
          *p_output_pos = current_pos;
          ++result_size;
          ++p_output_pos;
        }
      }
      return result_size;
    }
  };

  template <tsl::VectorProcessingStyle _SimdStyle, tsl::TSLArithmetic _PositionType,
            class HintSet = OperatorHintSet<hints::hashing::size_exp_2>, typename Idof = tsl::workaround>
  struct Hash_Semi_Join_RightSide_SIMD_Linear_Probing {
    using builder_t = Hash_Semi_Join_Build_RightSide_SIMD_Linear_Probing<_SimdStyle, HintSet, Idof>;
    using prober_t = Hash_Semi_Join_Probe_RightSide_SIMD_Linear_Probing<_SimdStyle, _PositionType, HintSet, Idof>;
  };
}  // namespace tuddbs