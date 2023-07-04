//
// Created by jpietrzyk on 28.05.19.
//

#ifndef MORPHSTORE_LINEAR_PROBING_H
#define MORPHSTORE_LINEAR_PROBING_H

#include <SIMDOperators/utils/preprocessor.h>

#include <tslintrin.hpp>
#include <SIMDOperators/datastructures/hash_based/hash_utils.h>

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>


namespace vectorlib {

   /**
    * @brief Linear Probe Strategy for hash based data structures.
    * @details As every strategy for hash based data structures, different insert as well as lookup methods are provided.
    * @tparam VectorExtension Vector extension which is used for probing.
    * @tparam BiggestSupportedVectorExtension Biggest vector extension the linear search should be able to work with.
    * @tparam HashFunction Struct which provides an static apply function to hash a vector register (VectorExtension::vector_t).
    * @tparam SPH Size policy which is needed for vectorlib::index_resizer
    * (either size_policy_hash::ARBITRARY or size_policy_hash::EXPONENTIAL).
    */
   template<
      class VectorExtension,
      class BiggestSupportedVectorExtension,
      template<class> class HashFunction,
      size_policy_hash SPH
   >
   struct scalar_key_vectorized_linear_search {
      struct state_single_key_t {
            alignas(VectorExtension::vector_size_B()) typename VectorExtension::base_type m_KeyArray[VectorExtension::vector_element_count()];
            alignas(VectorExtension::vector_size_B()) typename VectorExtension::base_type m_ValueArray [VectorExtension::vector_element_count()] ;
            alignas(VectorExtension::vector_size_B()) typename VectorExtension::base_type m_IndexArray[VectorExtension::vector_element_count()];
            typename HashFunction<VectorExtension>::state_t m_HashState;
            typename index_aligner<VectorExtension>::template state_t<BiggestSupportedVectorExtension> m_AlignerState;
            typename index_resizer<VectorExtension, SPH>::state_t m_ResizerState;
            typename VectorExtension::base_type * const m_KeyContainerStartPtr;
            typename VectorExtension::base_type * const m_KeyContainerEndPtr;

         state_single_key_t(
            typename VectorExtension::base_type * const p_KeyContainerStartPtr,
            size_t const p_ContainerBucketCount
         ) :
            m_ResizerState{p_ContainerBucketCount},
            m_KeyContainerStartPtr{p_KeyContainerStartPtr},
            m_KeyContainerEndPtr{p_KeyContainerStartPtr + m_ResizerState.m_ResizeValue - VectorExtension::vector_element_count()} { }
      };
      struct state_single_key_single_value_t {
         alignas(VectorExtension::vector_size_B()) typename VectorExtension::base_type m_KeyArray[VectorExtension::vector_element_count()];
         alignas(VectorExtension::vector_size_B()) typename VectorExtension::base_type m_ValueArray [VectorExtension::vector_element_count()] ;
         alignas(VectorExtension::vector_size_B()) typename VectorExtension::base_type m_IndexArray[VectorExtension::vector_element_count()];
         typename HashFunction<VectorExtension>::state_t m_HashState;
         typename index_aligner<VectorExtension>::template state_t<BiggestSupportedVectorExtension> m_AlignerState;
         typename index_resizer<VectorExtension, SPH>::state_t m_ResizerState;
         typename VectorExtension::base_type * const m_KeyContainerStartPtr;
         typename VectorExtension::base_type * const m_ValueContainerStartPtr;
         typename VectorExtension::base_type * const m_KeyContainerEndPtr;

         state_single_key_single_value_t(
            typename VectorExtension::base_type * const p_KeyContainerStartPtr,
            typename VectorExtension::base_type * const p_ValueContainerStartPtr,
            size_t const p_ContainerBucketCount
         ) :
            m_ResizerState{p_ContainerBucketCount},
            m_KeyContainerStartPtr{p_KeyContainerStartPtr},
            m_ValueContainerStartPtr{p_ValueContainerStartPtr},
            m_KeyContainerEndPtr{p_KeyContainerStartPtr + m_ResizerState.m_ResizeValue - VectorExtension::vector_element_count()} { }
      };
      struct state_double_key_single_value_t {
         alignas(VectorExtension::vector_size_B()) typename VectorExtension::base_type m_FirstKeyArray[VectorExtension::vector_element_count()];
         alignas(VectorExtension::vector_size_B()) typename VectorExtension::base_type m_SecondKeyArray[VectorExtension::vector_element_count()];
         alignas(VectorExtension::vector_size_B()) typename VectorExtension::base_type m_ValueArray [VectorExtension::vector_element_count()] ;
         alignas(VectorExtension::vector_size_B()) typename VectorExtension::base_type m_IndexArray[VectorExtension::vector_element_count()];
         typename HashFunction<VectorExtension>::state_t m_HashState;
         typename index_aligner<VectorExtension>::template state_t<BiggestSupportedVectorExtension> m_AlignerState;
         typename index_resizer<VectorExtension, SPH>::state_t m_ResizerState;
         typename VectorExtension::base_type * const m_FirstKeyContainerStartPtr;
         typename VectorExtension::base_type * const m_SecondKeyContainerStartPtr;
         typename VectorExtension::base_type * const m_ValueContainerStartPtr;
         typename VectorExtension::base_type * const m_KeyContainerEndPtr;

         state_double_key_single_value_t(
            typename VectorExtension::base_type * const p_FirstKeyContainerStartPtr,
            typename VectorExtension::base_type * const p_SecondKeyContainerStartPtr,
            typename VectorExtension::base_type * const p_ValueContainerStartPtr,
            size_t const p_ContainerBucketCount
         ) :
            m_ResizerState{p_ContainerBucketCount},
            m_FirstKeyContainerStartPtr{p_FirstKeyContainerStartPtr},
            m_SecondKeyContainerStartPtr{p_SecondKeyContainerStartPtr},
            m_ValueContainerStartPtr{p_ValueContainerStartPtr},
            m_KeyContainerEndPtr{p_FirstKeyContainerStartPtr + m_ResizerState.m_ResizeValue - VectorExtension::vector_element_count()} { }
      };
      /**
       *
       * @param p_InKeyVector
       * @param p_SearchState
       * @return Tuple containing a vector register with the indices where either the corresponding key matched, or an
       * empty bucket was found, a vector register mask with a bit set to one if the corresponding bucket matched and a
       * vector register mask with a bit set to one if the corresponding bucket is empty.
       */
//      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      std::tuple<typename VectorExtension::register_type, typename VectorExtension::mask_type, uint8_t>
      lookup(
         typename VectorExtension::register_type const & p_InKeyVector,
         state_single_key_single_value_t & p_SearchState
      ) {
         typename VectorExtension::register_type const zeroVec = tsl::set1<VectorExtension>(0);
         typename VectorExtension::mask_type resultMaskFound = 0;
         uint8_t resultCount = 0;
         tsl::store<VectorExtension>(
            p_SearchState.m_IndexArray,
            index_aligner<VectorExtension>::apply(
               index_resizer<VectorExtension, SPH>::apply(
                  HashFunction<VectorExtension>::apply(
                     p_InKeyVector,
                     p_SearchState.m_HashState
                  ),
                  p_SearchState.m_ResizerState
               ),
               p_SearchState.m_AlignerState
            )
         );
         tsl::store<VectorExtension>(
            p_SearchState.m_KeyArray,
            p_InKeyVector
         );
         typename VectorExtension::register_type keyVec;
         typename VectorExtension::mask_type searchOffset;
         typename VectorExtension::mask_type currentMask = 1;
         for(size_t pos = 0; pos < VectorExtension::vector_element_count(); ++pos) {
            typename VectorExtension::base_type index = p_SearchState.m_IndexArray[pos];
            typename VectorExtension::base_type *currentSearchPtr = p_SearchState.m_KeyContainerStartPtr + index;
            keyVec = tsl::set1<VectorExtension>(p_SearchState.m_KeyArray[pos] + 1 );
            bool done = false;
            while(!done) {
               typename VectorExtension::register_type loadedBucketsVec = tsl::load<VectorExtension>(
                  currentSearchPtr);
               searchOffset = tsl::equal<VectorExtension>(loadedBucketsVec, keyVec);
               if(searchOffset != 0) {
                  p_SearchState.m_ValueArray[pos] =
                     p_SearchState.m_ValueContainerStartPtr[index + __builtin_ctz(searchOffset)];
                  resultMaskFound |= currentMask;
                  ++resultCount;
                  done = true;
               } else {
                  searchOffset = tsl::equal<VectorExtension>(loadedBucketsVec, zeroVec);
                  if(searchOffset != 0) {
                     done = true;
                  } else {
                     if(MSV_CXX_ATTRIBUTE_LIKELY(currentSearchPtr < p_SearchState.m_KeyContainerEndPtr)) {
                        currentSearchPtr += VectorExtension::vector_element_count();
                        index += VectorExtension::vector_element_count();
                     } else {
                        currentSearchPtr = p_SearchState.m_KeyContainerStartPtr;
                        index = 0;
                     }
                  }
               }
            }
            currentMask = currentMask << 1;
         }
         return std::make_tuple(
            tsl::load<VectorExtension>( p_SearchState.m_ValueArray),
            resultMaskFound,
            resultCount
         );
      }

      /**
       *
       * @param p_InKeyVector
       * @param p_SearchState
       * @return Tuple containing a vector register with the indices where either the corresponding key matched, or an
       * empty bucket was found, a vector register mask with a bit set to one if the corresponding bucket matched and a
       * vector register mask with a bit set to one if the corresponding bucket is empty.
       */
//      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      std::pair< typename VectorExtension::mask_type, uint8_t >
      lookup(
         typename VectorExtension::register_type const & p_InKeyVector,
         state_single_key_t & p_SearchState
      ) {
         uint8_t resultCount = 0;
         typename VectorExtension::register_type const zeroVec = tsl::set1<VectorExtension>(0);
         typename VectorExtension::mask_type resultMaskFound = 0;
         tsl::store<VectorExtension>(
            p_SearchState.m_IndexArray,
            index_aligner<VectorExtension>::apply(
               index_resizer<VectorExtension, SPH>::apply(
                  HashFunction<VectorExtension>::apply(
                     p_InKeyVector,
                     p_SearchState.m_HashState
                  ),
                  p_SearchState.m_ResizerState
               ),
               p_SearchState.m_AlignerState
            )
         );
         tsl::store<VectorExtension>(
            p_SearchState.m_KeyArray,
            p_InKeyVector
         );
         typename VectorExtension::register_type keyVec;
         typename VectorExtension::mask_type searchOffset;
         typename VectorExtension::mask_type currentMask = 1;
         for(size_t pos = 0; pos < VectorExtension::vector_element_count(); ++pos) {
            typename VectorExtension::base_type index = p_SearchState.m_IndexArray[pos];
            typename VectorExtension::base_type *currentSearchPtr = p_SearchState.m_KeyContainerStartPtr + index;
            keyVec = tsl::set1<VectorExtension>(p_SearchState.m_KeyArray[pos] + 1 );
            searchOffset = 0;
            bool done = false;
            while(!done) {
               typename VectorExtension::register_type loadedBucketsVec = tsl::load<VectorExtension>(
                  currentSearchPtr);
               searchOffset = tsl::equal<VectorExtension>(loadedBucketsVec, keyVec);
               if(searchOffset != 0) {
                  resultMaskFound |= currentMask;
                  ++resultCount;
                  done = true;
               } else {
                  searchOffset = tsl::equal<VectorExtension>(loadedBucketsVec, zeroVec);
                  if(searchOffset != 0) {
                     done = true;
                  } else {
                     if(MSV_CXX_ATTRIBUTE_LIKELY(currentSearchPtr < p_SearchState.m_KeyContainerEndPtr)) {
                        currentSearchPtr += VectorExtension::vector_element_count();
                        index += VectorExtension::vector_element_count();
                     } else {
                        currentSearchPtr = p_SearchState.m_KeyContainerStartPtr;
                        index = 0;
                     }
                  }
               }
            }
            currentMask = currentMask << 1;
         }
         return std::make_pair(resultMaskFound, resultCount);
      }


//      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      void
      insert(
         typename VectorExtension::register_type const & p_InKeyVector,
         state_single_key_t & p_SearchState
      ) {
         typename VectorExtension::register_type const zeroVec = tsl::set1<VectorExtension>(0);
         typename VectorExtension::mask_type resultMaskFound = 0;
         tsl::store<VectorExtension>(
            p_SearchState.m_IndexArray,
            index_aligner<VectorExtension>::apply(
               index_resizer<VectorExtension, SPH>::apply(
                  HashFunction<VectorExtension>::apply(
                     p_InKeyVector,
                     p_SearchState.m_HashState
                  ),
                  p_SearchState.m_ResizerState
               ),
               p_SearchState.m_AlignerState
            )
         );
         tsl::store<VectorExtension>(
            p_SearchState.m_KeyArray,
            p_InKeyVector
         );
         typename VectorExtension::register_type keyVec;
         typename VectorExtension::mask_type searchOffset;
         for(size_t pos = 0; pos < VectorExtension::vector_element_count(); ++pos) {
            typename VectorExtension::base_type index = p_SearchState.m_IndexArray[pos];
            typename VectorExtension::base_type key = p_SearchState.m_KeyArray[pos] + 1;
            typename VectorExtension::base_type *currentSearchPtr = p_SearchState.m_KeyContainerStartPtr + index;
            keyVec = tsl::set1<VectorExtension>(key);
            bool done = false;
            while(!done) {
               typename VectorExtension::register_type loadedBucketsVec = tsl::load<VectorExtension>(
                  currentSearchPtr);
               searchOffset = tsl::equal<VectorExtension>(loadedBucketsVec, keyVec);
               if(searchOffset != 0) {
                  done = true;
               } else {
                  searchOffset = tsl::equal<VectorExtension>(loadedBucketsVec, zeroVec);
                  if(searchOffset != 0) {
                     p_SearchState.m_KeyContainerStartPtr[ index + __builtin_ctz(searchOffset) ] = key;
                     done = true;
                  } else {
                     if(MSV_CXX_ATTRIBUTE_LIKELY(currentSearchPtr < p_SearchState.m_KeyContainerEndPtr)) {
                        currentSearchPtr += VectorExtension::vector_element_count();
                        index += VectorExtension::vector_element_count();
                     } else {
                        currentSearchPtr = p_SearchState.m_KeyContainerStartPtr;
                        index = 0;
                     }
                  }
               }
            }
         }
      }


//      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      void
      insert(
         typename VectorExtension::register_type const & p_InKeyVector,
         typename VectorExtension::register_type const & p_InValueVector,
         state_single_key_single_value_t & p_SearchState
      ) {
         typename VectorExtension::register_type const zeroVec = tsl::set1<VectorExtension>(0);
         typename VectorExtension::mask_type resultMaskFound = 0;
         tsl::store<VectorExtension>(
            p_SearchState.m_IndexArray,
            index_aligner<VectorExtension>::apply(
               index_resizer<VectorExtension, SPH>::apply(
                  HashFunction<VectorExtension>::apply(
                     p_InKeyVector,
                     p_SearchState.m_HashState
                  ),
                  p_SearchState.m_ResizerState
               ),
               p_SearchState.m_AlignerState
            )
         );
         tsl::store<VectorExtension>(
            p_SearchState.m_KeyArray,
            p_InKeyVector
         );
         tsl::store<VectorExtension>(
            p_SearchState.m_ValueArray,
            p_InValueVector
         );
         typename VectorExtension::register_type keyVec;
         typename VectorExtension::mask_type searchOffset;
         for(size_t pos = 0; pos < VectorExtension::vector_element_count(); ++pos) {
            typename VectorExtension::base_type index = p_SearchState.m_IndexArray[pos];
            typename VectorExtension::base_type key = p_SearchState.m_KeyArray[pos] + 1;
            typename VectorExtension::base_type value = p_SearchState.m_ValueArray[pos];
            typename VectorExtension::base_type *currentSearchPtr = p_SearchState.m_KeyContainerStartPtr + index;
            keyVec = tsl::set1<VectorExtension>(key);
            bool done = false;
            while(!done) {
               typename VectorExtension::register_type loadedBucketsVec = tsl::load<VectorExtension>(
                  currentSearchPtr);
               searchOffset = tsl::equal<VectorExtension>(loadedBucketsVec, keyVec);
               if(searchOffset != 0) {
                  done = true;
               } else {
                  searchOffset = tsl::equal<VectorExtension>(loadedBucketsVec, zeroVec);
                  if(searchOffset != 0) {
                     size_t targetIdx = index + __builtin_ctz(searchOffset);
                     p_SearchState.m_KeyContainerStartPtr[targetIdx] = key;
                     p_SearchState.m_ValueContainerStartPtr[targetIdx] = value;
                     done = true;
                  } else {
                     if(MSV_CXX_ATTRIBUTE_LIKELY(currentSearchPtr < p_SearchState.m_KeyContainerEndPtr)) {
                        currentSearchPtr += VectorExtension::vector_element_count();
                        index += VectorExtension::vector_element_count();
                     } else {
                        currentSearchPtr = p_SearchState.m_KeyContainerStartPtr;
                        index = 0;
                     }
                  }
               }
            }
         }
      }



//      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      std::tuple<
         typename VectorExtension::register_type,      // groupID vector register
         typename VectorExtension::register_type,      // groupExt vector register
         typename VectorExtension::mask_type, // active groupExt elements
         uint8_t        // Number of active groupExt elements
      >
      insert_and_lookup(
         typename VectorExtension::register_type const & p_InKeyVector,
         typename VectorExtension::base_type & p_InStartPosFromKey,
         typename VectorExtension::base_type & p_InStartValue,
         state_single_key_single_value_t & p_SearchState
      ) {
         typename VectorExtension::register_type const zeroVec = tsl::set1<VectorExtension>(0);
         typename VectorExtension::mask_type activeGroupExtMask = 0;
         typename VectorExtension::mask_type currentMaskForGroupExtMask = 1;
         uint8_t activeGroupExtCount = 0;

         tsl::store<VectorExtension>(
            p_SearchState.m_IndexArray,
            index_aligner<VectorExtension>::apply(
               index_resizer<VectorExtension, SPH>::apply(
                  HashFunction<VectorExtension>::apply(
                     p_InKeyVector,
                     p_SearchState.m_HashState
                  ),
                  p_SearchState.m_ResizerState
               ),
               p_SearchState.m_AlignerState
            )
         );
         tsl::store<VectorExtension>(
            p_SearchState.m_KeyArray,
            p_InKeyVector
         );
         typename VectorExtension::register_type keyVec;
         typename VectorExtension::mask_type searchOffset;

         for(size_t pos = 0; pos < VectorExtension::vector_element_count(); ++pos) {
            typename VectorExtension::base_type index = p_SearchState.m_IndexArray[pos];
            typename VectorExtension::base_type key = p_SearchState.m_KeyArray[pos] + 1;

            typename VectorExtension::base_type * currentSearchPtr = p_SearchState.m_KeyContainerStartPtr + index;
            keyVec = tsl::set1<VectorExtension>(key);
            bool done = false;
            while(!done) {
               typename VectorExtension::register_type loadedBucketsVec = tsl::load<VectorExtension>(
                  currentSearchPtr);
               searchOffset = tsl::equal<VectorExtension>(loadedBucketsVec, keyVec);
               if(searchOffset != 0) {
                  p_SearchState.m_ValueArray[ pos ] = p_SearchState.m_ValueContainerStartPtr[index + __builtin_ctz(searchOffset)];
                  done = true;
               } else {
                  searchOffset = tsl::equal<VectorExtension>(loadedBucketsVec, zeroVec);
                  if(searchOffset != 0) {
                     size_t targetIdx = index + __builtin_ctz(searchOffset);
                     p_SearchState.m_KeyContainerStartPtr[targetIdx] = key;
                     p_SearchState.m_ValueContainerStartPtr[targetIdx] = p_InStartValue;
                     p_SearchState.m_ValueArray[ pos ] = p_InStartValue++;
                     p_SearchState.m_IndexArray[ pos ] = p_InStartPosFromKey;
                     activeGroupExtMask |= currentMaskForGroupExtMask;
                     ++activeGroupExtCount;
                     done = true;
                  } else {
                     if(MSV_CXX_ATTRIBUTE_LIKELY(currentSearchPtr < p_SearchState.m_KeyContainerEndPtr)) {
                        currentSearchPtr += VectorExtension::vector_element_count();
                        index += VectorExtension::vector_element_count();
                     } else {
                        currentSearchPtr = p_SearchState.m_KeyContainerStartPtr;
                        index = 0;
                     }
                  }
               }
            }
            currentMaskForGroupExtMask = currentMaskForGroupExtMask << 1;
            ++p_InStartPosFromKey;
         }
         return
            std::make_tuple(
               tsl::load<VectorExtension>(p_SearchState.m_ValueArray),
               tsl::load<VectorExtension>(p_SearchState.m_IndexArray),
               activeGroupExtMask,
               activeGroupExtCount
            );
      }

//      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      std::tuple<
         typename VectorExtension::register_type,      // groupID vector register
         typename VectorExtension::register_type,      // groupExt vector register
         typename VectorExtension::mask_type, // active groupExt elements
         uint8_t        // Number of active groupExt elements
      >
      insert_and_lookup(
         typename VectorExtension::register_type const & p_InKeysFirstVector,
         typename VectorExtension::register_type const & p_InKeySecondVector,
         typename VectorExtension::base_type & p_InStartPosFromKey,
         typename VectorExtension::base_type & p_InStartValue,
         state_double_key_single_value_t & p_SearchState
      ) {
         typename VectorExtension::register_type const zeroVec = tsl::set1<VectorExtension>(0);
         typename VectorExtension::mask_type activeGroupExtMask = 0;
         typename VectorExtension::mask_type currentMaskForGroupExtMask = 1;
         uint8_t activeGroupExtCount = 0;

         tsl::store<VectorExtension>(
            p_SearchState.m_IndexArray,
            index_aligner<VectorExtension>::apply(
               index_resizer<VectorExtension, SPH>::apply(
                  HashFunction<VectorExtension>::apply(
                     p_InKeysFirstVector,
                     p_InKeySecondVector,
                     p_SearchState.m_HashState
                  ),
                  p_SearchState.m_ResizerState
               ),
               p_SearchState.m_AlignerState
            )
         );
         tsl::store<VectorExtension>(
            p_SearchState.m_FirstKeyArray,
            p_InKeysFirstVector
         );
         tsl::store<VectorExtension>(
            p_SearchState.m_SecondKeyArray,
            p_InKeySecondVector
         );
         typename VectorExtension::register_type keyFirstVec,keySecondVec;
         typename VectorExtension::mask_type searchOffset;

         for(size_t pos = 0; pos < VectorExtension::vector_element_count(); ++pos) {
            typename VectorExtension::base_type index = p_SearchState.m_IndexArray[pos];
            typename VectorExtension::base_type keyFirst = p_SearchState.m_FirstKeyArray[pos] + 1;
            typename VectorExtension::base_type keySecond = p_SearchState.m_SecondKeyArray[pos];

            typename VectorExtension::base_type * currentFirstKeySearchPtr = p_SearchState.m_FirstKeyContainerStartPtr + index;
            typename VectorExtension::base_type * currentSecondKeySearchPtr = p_SearchState.m_SecondKeyContainerStartPtr + index;
            keyFirstVec = tsl::set1<VectorExtension>(keyFirst);
            keySecondVec = tsl::set1<VectorExtension>(keySecond);
            bool done = false;
            while(!done) {
               typename VectorExtension::register_type loadedFirstBucketsVec = tsl::load<VectorExtension>(
                  currentFirstKeySearchPtr);
               typename VectorExtension::register_type loadedSecondBucketsVec = tsl::load<VectorExtension>(
                  currentSecondKeySearchPtr);
               searchOffset =
                  (
                     tsl::equal<VectorExtension>(loadedFirstBucketsVec, keyFirstVec)
                  &
                     tsl::equal<VectorExtension>(loadedSecondBucketsVec, keySecondVec)
                  );
               if(searchOffset != 0) {
                  p_SearchState.m_ValueArray[ pos ] = p_SearchState.m_ValueContainerStartPtr[index + __builtin_ctz(searchOffset)];
                  done = true;
               } else {
                  searchOffset =
                     (
                        tsl::equal<VectorExtension>(loadedFirstBucketsVec, zeroVec)
                     &
                        tsl::equal<VectorExtension>(loadedSecondBucketsVec, zeroVec)
                     );
                  if(searchOffset != 0) {
                     size_t targetIdx = index + __builtin_ctz(searchOffset);
                     p_SearchState.m_FirstKeyContainerStartPtr[targetIdx] = keyFirst;
                     p_SearchState.m_SecondKeyContainerStartPtr[targetIdx] = keySecond;
                     p_SearchState.m_ValueContainerStartPtr[targetIdx] = p_InStartValue;
                     p_SearchState.m_ValueArray[pos] = p_InStartValue++;
                     p_SearchState.m_IndexArray[pos] = p_InStartPosFromKey;
                     activeGroupExtMask |= currentMaskForGroupExtMask;
                     ++activeGroupExtCount;
                     done = true;
                  } else {
                     if(MSV_CXX_ATTRIBUTE_LIKELY(currentFirstKeySearchPtr < p_SearchState.m_KeyContainerEndPtr)) {
                        currentFirstKeySearchPtr += VectorExtension::vector_element_count();
                        currentSecondKeySearchPtr += VectorExtension::vector_element_count();
                        index += VectorExtension::vector_element_count();
                     } else {
                        currentFirstKeySearchPtr = p_SearchState.m_FirstKeyContainerStartPtr;
                        currentSecondKeySearchPtr = p_SearchState.m_SecondKeyContainerStartPtr;
                        index = 0;
                     }
                  }
               }
            }
            currentMaskForGroupExtMask = currentMaskForGroupExtMask << 1;
            ++p_InStartPosFromKey;
         }
         return
            std::make_tuple(
               tsl::load<VectorExtension>(p_SearchState.m_ValueArray),
               tsl::load<VectorExtension>(p_SearchState.m_IndexArray),
               activeGroupExtMask,
               activeGroupExtCount
            );
      }
   };

}
#endif //MORPHSTORE_LINEAR_PROBING_H
